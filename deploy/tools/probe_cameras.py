#!/usr/bin/env python3
"""
probe_cameras.py — NX Computing AI | DVR Channel Probe

Discovers which channels on a DVR actually have cameras connected
by sending a raw RTSP DESCRIBE request to each channel URL.
No GStreamer or ffmpeg needed — uses plain sockets.

Usage (from repo root, credentials already in clients/<name>/.env):
    python3 tools/probe_cameras.py
    python3 tools/probe_cameras.py --client demo
    python3 tools/probe_cameras.py --client demo --max-ch 32 --update-config

The --update-config flag writes the discovered channels back to
clients/<name>/config.yaml so you don't have to edit it manually.
"""

import argparse
import os
import re
import socket
import sys
from pathlib import Path

import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT   = 4   # seconds per channel probe


# ── RTSP DESCRIBE probe ───────────────────────────────────────────────────────

def _rtsp_describe(host: str, port: int, path: str,
                   user: str, password: str) -> int:
    """
    Send RTSP DESCRIBE to one URL. Returns the HTTP-style status code:
      200 → camera present and accessible
      401 → auth required (camera likely exists, wrong credentials)
      404 → no stream at this path (channel empty / not configured)
      0   → connection refused / timeout (DVR unreachable)
    """
    url = f"rtsp://{host}:{port}{path}"
    request = (
        f"DESCRIBE {url} RTSP/1.0\r\n"
        f"CSeq: 1\r\n"
        f"User-Agent: NX-probe/1.0\r\n"
        f"Accept: application/sdp\r\n"
        f"\r\n"
    )
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT) as s:
            s.sendall(request.encode())
            response = s.recv(256).decode(errors="ignore")
        m = re.search(r"RTSP/1\.\d\s+(\d{3})", response)
        return int(m.group(1)) if m else 0
    except (OSError, ConnectionRefusedError):
        return 0


def probe_channel(dvr_ip: str, dvr_port: int, pattern: str,
                  ch: int, user: str, password: str) -> str:
    """
    Returns 'ok', 'auth_error', 'empty', or 'unreachable'.
    """
    path = pattern.replace("{dvr_ip}", dvr_ip) \
                  .replace("{port}", str(dvr_port)) \
                  .replace("{user}", user) \
                  .replace("{password}", password) \
                  .replace("{ch:02d}", f"{ch:02d}")

    # Extract just the path component for the DESCRIBE request
    # (strip rtsp://host:port prefix)
    url_path = re.sub(r"^rtsp://[^/]+", "", path)

    code = _rtsp_describe(dvr_ip, dvr_port, url_path, user, password)

    if code == 200:
        return "ok"
    elif code in (401, 403):
        return "auth_error"
    elif code == 404:
        return "empty"
    else:
        return "unreachable"


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_runtime(client_name: str):
    """Load DVR IP + credentials + current config for a client."""
    dvr_ip = (
        os.environ.get("NX_DVR_IP", "").strip()
        or Path("/etc/nx_dvr_ip").read_text().strip()
    )

    client_dir  = REPO_ROOT / "clients" / client_name
    config_path = client_dir / "config.yaml"
    env_path    = client_dir / ".env"

    if not config_path.exists():
        sys.exit(f"[ERR] No config found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = dotenv_values(env_path) if env_path.exists() else {}
    user     = env.get("DVR_USER", os.environ.get("DVR_USER", ""))
    password = env.get("DVR_PASS",  os.environ.get("DVR_PASS",  ""))

    if not user or not password:
        sys.exit(
            f"[ERR] DVR credentials missing.\n"
            f"      Copy {client_dir}/.env.example to {client_dir}/.env and fill it in."
        )

    return dvr_ip, cfg, user, password, config_path


def _update_config(config_path: Path, channels: list[int]):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["channels"] = channels
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\n[OK] Updated {config_path}  channels: {channels}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Probe DVR channels via RTSP DESCRIBE")
    ap.add_argument("--client",        default=None,
                    help="Client name (default: reads /etc/nx_client)")
    ap.add_argument("--max-ch",        type=int, default=16,
                    help="Highest channel number to probe (default: 16)")
    ap.add_argument("--update-config", action="store_true",
                    help="Write discovered channels to clients/<name>/config.yaml")
    ap.add_argument("--dvr-ip",        default=None,
                    help="Override DVR IP (default: /etc/nx_dvr_ip)")
    args = ap.parse_args()

    # Resolve client name
    client_name = args.client
    if not client_name:
        try:
            client_name = (
                os.environ.get("NX_CLIENT", "").strip()
                or Path("/etc/nx_client").read_text().strip()
            )
        except FileNotFoundError:
            sys.exit("[ERR] Client name not set. Use --client <name> or run setup.sh --client <name>")

    dvr_ip, cfg, user, password, config_path = _load_runtime(client_name)

    if args.dvr_ip:
        dvr_ip = args.dvr_ip

    dvr_port = int(cfg.get("dvr_port", 554))
    pattern  = cfg.get(
        "rtsp_url_pattern",
        "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/main/av_stream",
    )

    print(f"\n  Client  : {client_name}")
    print(f"  DVR     : {dvr_ip}:{dvr_port}")
    print(f"  Probing : channels 1–{args.max_ch}  (timeout {TIMEOUT}s each)\n")

    found      = []
    auth_error = False

    for ch in range(1, args.max_ch + 1):
        result = probe_channel(dvr_ip, dvr_port, pattern, ch, user, password)

        if result == "ok":
            print(f"  ch{ch:02d}  ✓  camera present")
            found.append(ch)
        elif result == "auth_error":
            print(f"  ch{ch:02d}  ✗  auth error (wrong DVR_USER / DVR_PASS?)")
            auth_error = True
            break
        elif result == "empty":
            print(f"  ch{ch:02d}  —  no camera")
        else:
            print(f"  ch{ch:02d}  ?  unreachable (DVR offline or wrong URL pattern)")
            if ch == 1:
                print("\n[WARN] First channel unreachable — check DVR IP and URL pattern.")
                break

    print(f"\n  Found {len(found)} camera(s): {found}")

    if auth_error:
        print("\n[ERR] Authentication failed — fix DVR_USER / DVR_PASS in .env and retry.")
        sys.exit(1)

    if found and args.update_config:
        _update_config(config_path, found)
    elif found:
        print(f"\n  To save this to config, re-run with --update-config")
        print(f"  Or add manually to clients/{client_name}/config.yaml:")
        print(f"    channels: {found}")


if __name__ == "__main__":
    main()
