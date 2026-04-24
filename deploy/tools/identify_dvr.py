#!/usr/bin/env python3
"""
identify_dvr.py — NX Computing AI | DVR Auto-Identification

Tries all known RTSP URL patterns against channel 1 of the DVR,
finds the one that works, then uses gst-discoverer-1.0 to read
the actual stream resolution.

Prints the exact values to paste into clients/<name>/config.yaml.

Usage (run inside the Docker container on the Jetson):
    python3 tools/identify_dvr.py
    python3 tools/identify_dvr.py --client demo
    python3 tools/identify_dvr.py --dvr-ip 192.168.10.68
"""

import argparse
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT   = 5   # seconds per pattern attempt

# ── Known RTSP URL patterns (most common first) ───────────────────────────────
# {ch}     = channel number as-is  (1, 2, 3 ...)
# {ch:02d} = zero-padded           (01, 02, 03 ...)

PATTERNS = [
    # Name,                          Pattern
    ("Generic / QSee / Swann / Annke",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/main/av_stream"),

    ("Hikvision",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/Streaming/Channels/{ch:02d}01"),

    ("Dahua / Amcrest / Lorex",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=0"),

    ("Reolink",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/h264Preview_{ch:02d}_main"),

    ("Uniview / Unv",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/media/video{ch}"),

    ("Axis",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/axis-media/media.amp?videocodec=h264&camera={ch}"),

    ("Hanwha / Samsung",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/profile{ch}/media.smp"),

    ("Generic variant — h264 path",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/h264/ch{ch:02d}/main/av_stream"),

    ("Generic variant — live path",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/live/ch{ch:02d}"),

    ("Generic variant — stream path",
     "rtsp://{user}:{password}@{dvr_ip}:{port}/stream{ch}"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_url(pattern: str, dvr_ip: str, port: int,
               user: str, password: str, ch: int) -> str:
    return (
        pattern
        .replace("{user}",     user)
        .replace("{password}", password)
        .replace("{dvr_ip}",   dvr_ip)
        .replace("{port}",     str(port))
        .replace("{ch:02d}",   f"{ch:02d}")
        .replace("{ch}",       str(ch))
    )


def _rtsp_describe(host: str, port: int, url_path: str) -> Tuple[int, str]:
    """
    Send RTSP DESCRIBE. Returns (status_code, body).
    body contains the SDP on 200 OK — useful for future parsing.
    """
    url = f"rtsp://{host}:{port}{url_path}"
    request = (
        f"DESCRIBE {url} RTSP/1.0\r\n"
        f"CSeq: 1\r\n"
        f"User-Agent: NX-identify/1.0\r\n"
        f"Accept: application/sdp\r\n"
        f"\r\n"
    )
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT) as s:
            s.settimeout(TIMEOUT)
            s.sendall(request.encode())
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\r\n\r\n" in data:
                    break
        text = data.decode(errors="ignore")
        m = re.search(r"RTSP/1\.\d\s+(\d{3})", text)
        code = int(m.group(1)) if m else 0
        return code, text
    except (OSError, ConnectionRefusedError, TimeoutError):
        return 0, ""


def _probe_pattern(pattern: str, dvr_ip: str, port: int,
                   user: str, password: str) -> Tuple[bool, str]:
    """Try one pattern on channel 1. Returns (success, full_url)."""
    url = _build_url(pattern, dvr_ip, port, user, password, ch=1)
    url_path = re.sub(r"^rtsp://[^/]+", "", url)
    code, _ = _rtsp_describe(dvr_ip, port, url_path)

    if code == 200:
        return True, url
    if code in (401, 403):
        return None, url   # None = auth error
    return False, url


def _get_resolution(rtsp_url: str) -> Optional[Tuple[int, int]]:
    """
    Use gst-discoverer-1.0 to get stream resolution.
    Returns (width, height) or None if it fails.
    """
    try:
        result = subprocess.run(
            ["gst-discoverer-1.0", "-v", rtsp_url],
            capture_output=True, text=True, timeout=15,
        )
        output = result.stdout + result.stderr
        # Look for "Width: 1920" and "Height: 1080"
        w = re.search(r"[Ww]idth[:\s]+(\d+)", output)
        h = re.search(r"[Hh]eight[:\s]+(\d+)", output)
        if w and h:
            return int(w.group(1)), int(h.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _load_credentials(client_name: str) -> Tuple[str, str, int, str]:
    """Returns (dvr_ip, user, password, dvr_port)."""
    dvr_ip = (
        os.environ.get("NX_DVR_IP", "").strip()
        or Path("/etc/nx_dvr_ip").read_text().strip()
    )

    client_dir = REPO_ROOT / "clients" / client_name
    env_path   = client_dir / ".env"
    cfg_path   = client_dir / "config.yaml"

    env = dotenv_values(env_path) if env_path.exists() else {}
    user     = env.get("DVR_USER", os.environ.get("DVR_USER", ""))
    password = env.get("DVR_PASS",  os.environ.get("DVR_PASS",  ""))

    if not user or not password:
        sys.exit(
            f"[ERR] Credentials missing.\n"
            f"      Copy {client_dir}/.env.example → {client_dir}/.env and fill it in."
        )

    port = 554
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        port = int(cfg.get("dvr_port", 554))

    return dvr_ip, user, password, port


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Auto-identify DVR RTSP URL pattern and stream resolution")
    ap.add_argument("--client",        default=None, help="Client name (default: /etc/nx_client)")
    ap.add_argument("--dvr-ip",        default=None, help="Override DVR IP")
    ap.add_argument("--port",          type=int, default=None, help="Override DVR port (default: 554)")
    ap.add_argument("--update-config", action="store_true", help="Write detected values directly into config.yaml")
    args = ap.parse_args()

    # Resolve client
    client_name = args.client
    if not client_name:
        try:
            client_name = (
                os.environ.get("NX_CLIENT", "").strip()
                or Path("/etc/nx_client").read_text().strip()
            )
        except FileNotFoundError:
            sys.exit("[ERR] Client not set. Use --client <name> or run setup.sh --client <name>")

    dvr_ip, user, password, dvr_port = _load_credentials(client_name)

    if args.dvr_ip:
        dvr_ip = args.dvr_ip
    if args.port:
        dvr_port = args.port

    print(f"\n  Client  : {client_name}")
    print(f"  DVR     : {dvr_ip}:{dvr_port}")
    print(f"  Trying {len(PATTERNS)} known URL patterns on channel 1…\n")

    working_pattern  = None
    working_url      = None
    auth_failed      = False

    for name, pattern in PATTERNS:
        result, url = _probe_pattern(pattern, dvr_ip, dvr_port, user, password)
        masked = url.replace(password, "***") if password else url

        if result is True:
            print(f"  ✓  {name}")
            print(f"     {masked}")
            working_pattern = pattern
            working_url     = url
            break
        elif result is None:
            print(f"  ✗  Auth error on: {masked}")
            print("     DVR responded but rejected credentials — fix DVR_USER / DVR_PASS in .env")
            auth_failed = True
            break
        else:
            print(f"  —  {name}")

    if auth_failed or not working_pattern:
        if not auth_failed:
            print("\n[WARN] No known pattern worked.")
            print("       Check DVR brand/model and add its pattern manually to config.yaml.")
            print("       Or try with VLC: Media → Open Network Stream → test URLs above.")
        sys.exit(1)

    # ── Detect resolution ─────────────────────────────────────────────────────
    print("\n  Detecting stream resolution via gst-discoverer-1.0…")
    resolution = _get_resolution(working_url)

    width, height = resolution if resolution else (1920, 1080)
    res_note = "" if resolution else "  (gst-discoverer not available — defaulting to 1920x1080, verify manually)"

    # ── Write or print result ─────────────────────────────────────────────────
    if args.update_config:
        cfg_path = REPO_ROOT / "clients" / client_name / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        cfg["dvr_port"]         = dvr_port
        cfg["rtsp_url_pattern"] = working_pattern
        cfg["stream_width"]     = width
        cfg["stream_height"]    = height
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"\n  ✓  config.yaml updated: {cfg_path}")
        if res_note:
            print(f"  {res_note.strip()}")
    else:
        print(f"\n{'─'*55}")
        print(f"  DVR identified. Paste this into clients/{client_name}/config.yaml:")
        print(f"{'─'*55}\n")
        print(f"  dvr_port: {dvr_port}")
        print(f"  rtsp_url_pattern: \"{working_pattern}\"")
        print(f"  stream_width: {width}")
        print(f"  stream_height: {height}{res_note}")
        print(f"\n{'─'*55}")

    print("\n  Next: run probe_cameras.py --update-config to discover active channels.")


if __name__ == "__main__":
    main()
