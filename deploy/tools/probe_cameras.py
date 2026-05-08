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
import hashlib
import os
import re
import socket
import sys
from pathlib import Path
from typing import List

import yaml
from ruamel.yaml import YAML as _YAML
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT   = 4   # seconds per channel probe


# ── RTSP Digest auth helper ───────────────────────────────────────────────────

def _digest_auth_header(user: str, password: str, method: str,
                        uri: str, www_auth: str) -> str:
    """
    Build RFC 2617 no-qop Digest Authorization header.
    uri MUST be the full absolute rtsp:// URL (RFC 2326 + RFC 2617 §3.2.2).
    """
    www_auth = www_auth.strip()
    realm_m = re.search(r'realm="([^"]+)"', www_auth)
    nonce_m = re.search(r'nonce="([^"]+)"', www_auth)
    if not realm_m or not nonce_m:
        return ""
    realm = realm_m.group(1).strip()
    nonce = nonce_m.group(1).strip()
    ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    return (f'Digest username="{user}", realm="{realm}", '
            f'nonce="{nonce}", uri="{uri}", '
            f'algorithm="MD5", response="{response}"')


def _recv_response(s: socket.socket) -> str:
    data = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data:
                break
    except (socket.timeout, OSError):
        pass
    return data.decode(errors="ignore")


# ── RTSP DESCRIBE probe ───────────────────────────────────────────────────────

def _rtsp_describe(host: str, port: int, path: str,
                   user: str, password: str) -> int:
    """
    Send RTSP DESCRIBE with full Digest auth support.
    Returns the RTSP status code:
      200 → camera present and authenticated
      401 → persistent 401 after digest auth attempt → bad credentials
      403 → forbidden
      404 → no stream at this path (channel empty / not configured)
      0   → connection refused / timeout (DVR unreachable)

    Dahua DVRs always respond with 401+challenge on the first unauthenticated
    request, even for channels that exist. This is NOT an auth error — it is
    the normal challenge/response flow. We therefore always attempt digest
    auth when we receive a 401, and only report auth_error if the second
    authenticated request also returns 401.
    """
    url = f"rtsp://{host}:{port}{path}"

    def _make_req(auth_header: str = "", cseq: int = 1) -> bytes:
        req = (f"DESCRIBE {url} RTSP/1.0\r\n"
               f"CSeq: {cseq}\r\n"
               f"User-Agent: NX-probe/1.0\r\n"
               f"Accept: application/sdp\r\n")
        if auth_header:
            req += f"Authorization: {auth_header}\r\n"
        req += "\r\n"
        return req.encode()

    try:
        with socket.create_connection((host, port), timeout=TIMEOUT) as s:
            s.settimeout(TIMEOUT)

            # Step 1 — unauthenticated probe
            s.sendall(_make_req(cseq=1))
            text = _recv_response(s)

            m = re.search(r"RTSP/1\.\d\s+(\d{3})", text)
            code = int(m.group(1)) if m else 0

            if code != 401 or not user or not password:
                return code

            # Step 2 — digest auth on the SAME connection
            # Dahua binds the nonce to the TCP session.
            www_auth_m = re.search(r"WWW-Authenticate:\s*(.+)", text)
            if not www_auth_m:
                return code

            auth = _digest_auth_header(user, password, "DESCRIBE", url,
                                       www_auth_m.group(1))
            if not auth:
                return code

            s.sendall(_make_req(auth, cseq=2))
            text2 = _recv_response(s)

        m2 = re.search(r"RTSP/1\.\d\s+(\d{3})", text2)
        return int(m2.group(1)) if m2 else 0

    except (OSError, ConnectionRefusedError, socket.timeout):
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
                  .replace("{ch:02d}", f"{ch:02d}") \
                  .replace("{ch}", str(ch))

    # Extract just the path component for the socket connection
    # (strip rtsp://host:port prefix); the full URL is reconstructed inside
    # _rtsp_describe for use in the DESCRIBE request line and digest URI.
    url_path = re.sub(r"^rtsp://[^/]+", "", path)

    code = _rtsp_describe(dvr_ip, dvr_port, url_path, user, password)

    if code == 200:
        return "ok"
    elif code in (401, 403):
        # 401 here means digest auth was attempted and still rejected → bad creds
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


def _update_config(config_path: Path, channels: List[int]):
    ryaml = _YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        cfg = ryaml.load(f) or {}
    cfg["channels"] = channels
    with open(config_path, "w") as f:
        ryaml.dump(cfg, f)
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
