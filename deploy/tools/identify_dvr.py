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
    python3 tools/identify_dvr.py --stream-type sub --update-config
"""

import argparse
import hashlib
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import yaml
from dotenv import dotenv_values
from ruamel.yaml import YAML as _YAML

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT   = 5   # seconds per pattern attempt

# ── Known RTSP URL patterns (most common first) ───────────────────────────────
# {ch}     = channel number as-is  (1, 2, 3 ...)
# {ch:02d} = zero-padded           (01, 02, 03 ...)
#
# Detection always probes the "main" pattern (more reliable).
# Once the brand is identified, "sub" is used when --stream-type sub is set.
# sub=None means the sub-stream path is device-specific; operator must set it manually.

PATTERNS = [
    {
        "name": "Generic / QSee / Swann / Annke",
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/main/av_stream",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/sub/av_stream",
    },
    {
        "name": "Hikvision",
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/Streaming/Channels/{ch:02d}01",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/Streaming/Channels/{ch:02d}02",
    },
    {
        "name": "Dahua / Amcrest / Lorex",
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=0",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=1",
    },
    {
        "name": "Reolink",
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/h264Preview_{ch:02d}_main",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/h264Preview_{ch:02d}_sub",
    },
    {
        "name": "Uniview / Unv",
        # Detection via /media/video{ch} (legacy firmware).
        # Sub-stream uses the newer unicast path; may need manual verification.
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/media/video{ch}",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/unicast/c{ch}/s1/live",
    },
    {
        "name": "Axis",
        # Axis sub-stream requires named stream profiles configured in the camera web UI.
        # No universal path exists — must be set manually.
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/axis-media/media.amp?videocodec=h264&camera={ch}",
        "sub":  None,
    },
    {
        "name": "Hanwha / Samsung",
        # Hanwha profile numbers are device-dependent (profile1=main, profile2=sub is common
        # but not guaranteed). Verify in the device web UI before deploying.
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/profile{ch}/media.smp",
        "sub":  None,
    },
    {
        "name": "Generic variant — h264 path",
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/h264/ch{ch:02d}/main/av_stream",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/h264/ch{ch:02d}/sub/av_stream",
    },
    {
        "name": "Generic variant — live path",
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/live/ch{ch:02d}",
        "sub":  "rtsp://{user}:{password}@{dvr_ip}:{port}/live/ch{ch:02d}_1",
    },
    {
        "name": "Generic variant — stream path",
        # No known sub-stream convention for this format.
        "main": "rtsp://{user}:{password}@{dvr_ip}:{port}/stream{ch}",
        "sub":  None,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_url(pattern: str, dvr_ip: str, port: int,
               user: str, password: str, ch: int) -> str:
    """Interpola un patrón de URL RTSP con las credenciales y el número de canal.

    Los placeholders soportados son: {user}, {password}, {dvr_ip}, {port}, {ch:02d}, {ch}.
    El orden de reemplazo importa: {ch:02d} debe reemplazarse antes que {ch} para evitar
    que '02d' quede parcialmente en la URL si el patrón usa ambos.
    """
    return (
        pattern
        .replace("{user}",     user)
        .replace("{password}", password)
        .replace("{dvr_ip}",   dvr_ip)
        .replace("{port}",     str(port))
        .replace("{ch:02d}",   f"{ch:02d}")
        .replace("{ch}",       str(ch))
    )


def _recv_response(s: socket.socket) -> str:
    """Read until we see the end of headers (\\r\\n\\r\\n) or the socket closes/times out."""
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


def _digest_auth_header(user: str, password: str, method: str,
                        uri: str, www_auth: str) -> str:
    """
    Build an RFC 2617 Digest Authorization header (no-qop / RFC 2069 variant).

    Dahua DVRs issue WWW-Authenticate with no qop field, so the response
    formula is simply: MD5(HA1 : nonce : HA2).

    The uri MUST be the full absolute RTSP URL (e.g. rtsp://host:554/path)
    because RFC 2326 §10.4 mandates absolute URLs in the RTSP Request-Line,
    and RFC 2617 §3.2.2 requires digest-uri to match the Request-URI exactly.

    The www_auth string may contain a trailing \\r — strip it to avoid
    corrupting the extracted nonce value.
    """
    www_auth = www_auth.strip()
    realm = re.search(r'realm="([^"]+)"', www_auth)
    nonce = re.search(r'nonce="([^"]+)"', www_auth)
    if not realm or not nonce:
        return ""
    realm = realm.group(1).strip()
    nonce = nonce.group(1).strip()
    ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    # Include algorithm="MD5" — required by some Dahua firmware revisions.
    return (f'Digest username="{user}", realm="{realm}", '
            f'nonce="{nonce}", uri="{uri}", '
            f'algorithm="MD5", response="{response}"')


def _rtsp_describe(host: str, port: int, url_path: str,
                   user: str = "", password: str = "") -> Tuple[int, str]:
    """
    Send RTSP DESCRIBE with Digest auth support.
    Returns (status_code, body).

    Protocol flow for Dahua (and most DVRs):
      1. Open TCP connection → send unauthenticated DESCRIBE.
      2. DVR replies 401 + WWW-Authenticate: Digest realm=..., nonce=...
         Dahua closes (or may keep) the connection here.
      3. Open a NEW TCP connection → send authenticated DESCRIBE.
         The nonce from step 2 is valid for this new connection.

    Using a fresh socket for the authenticated request is the safest
    approach; some Dahua firmware mark a nonce as used after the first
    401 response and will reject a retry on the same TCP session.

    The uri in the Authorization header and in HA2 MUST be the full
    absolute rtsp:// URL — RFC 2326 uses absolute URLs in the Request-Line,
    and RFC 2617 requires digest-uri to match the Request-URI.
    """
    url = f"rtsp://{host}:{port}{url_path}"

    def _make_request(auth_header: str = "", cseq: int = 1) -> str:
        """Construye la petición RTSP DESCRIBE en formato texto. auth_header vacío = sin autenticación."""
        req = (f"DESCRIBE {url} RTSP/1.0\r\n"
               f"CSeq: {cseq}\r\n"
               f"User-Agent: NX-identify/1.0\r\n"
               f"Accept: application/sdp\r\n")
        if auth_header:
            req += f"Authorization: {auth_header}\r\n"
        req += "\r\n"
        return req

    try:
        with socket.create_connection((host, port), timeout=TIMEOUT) as s:
            s.settimeout(TIMEOUT)

            # ── Step 1: unauthenticated probe ─────────────────────────────────
            s.sendall(_make_request(cseq=1).encode())
            text = _recv_response(s)

            m = re.search(r"RTSP/1\.\d\s+(\d{3})", text)
            code = int(m.group(1)) if m else 0

            if code == 200:
                return code, text

            # ── Step 2: digest auth on the SAME connection ────────────────────
            # Dahua binds the nonce to the TCP session — a new connection
            # causes the DVR to reject the nonce even with correct credentials.
            if code == 401 and user and password:
                www_auth_m = re.search(r"WWW-Authenticate:\s*(.+)", text)
                if www_auth_m:
                    auth = _digest_auth_header(
                        user, password, "DESCRIBE", url, www_auth_m.group(1))
                    if auth:
                        s.sendall(_make_request(auth, cseq=2).encode())
                        text = _recv_response(s)
                        m = re.search(r"RTSP/1\.\d\s+(\d{3})", text)
                        code = int(m.group(1)) if m else 0

        return code, text
    except (OSError, ConnectionRefusedError, TimeoutError, socket.timeout):
        return 0, ""


def _probe_pattern(pattern: str, dvr_ip: str, port: int,
                   user: str, password: str) -> Tuple[bool, str]:
    """Try one pattern on channel 1. Returns (success, full_url)."""
    url = _build_url(pattern, dvr_ip, port, user, password, ch=1)
    url_path = re.sub(r"^rtsp://[^/]+", "", url)
    code, _ = _rtsp_describe(dvr_ip, port, url_path, user, password)

    if code == 200:
        return True, url
    if code == 403:
        return None, url   # Forbidden — wrong credentials
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
    """Punto de entrada: prueba patrones RTSP conocidos contra el DVR y opcionalmente actualiza config.yaml.

    Si --update-config se pasa, escribe el patrón detectado, la resolución y el stream_type
    directamente en clients/<cliente>/config.yaml con ruamel.yaml (preservando comentarios).
    Útil desde setup.sh para configurar automáticamente el cliente sin intervención manual.
    """
    ap = argparse.ArgumentParser(description="Auto-identify DVR RTSP URL pattern and stream resolution")
    ap.add_argument("--client",        default=None, help="Client name (default: /etc/nx_client)")
    ap.add_argument("--dvr-ip",        default=None, help="Override DVR IP")
    ap.add_argument("--port",          type=int, default=None, help="Override DVR port (default: 554)")
    ap.add_argument("--update-config", action="store_true", help="Write detected values directly into config.yaml")
    ap.add_argument("--stream-type",   choices=["main", "sub"], default="main",
                    help="Stream type to configure: main (1920×1080, default) or sub (960×544, for 16+ cameras)")
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

    stream_type = args.stream_type

    print(f"\n  Client      : {client_name}")
    print(f"  DVR         : {dvr_ip}:{dvr_port}")
    print(f"  Stream type : {stream_type}")
    print(f"  Trying {len(PATTERNS)} known URL patterns on channel 1…\n")

    working_entry = None
    working_url   = None
    auth_failed   = False

    for entry in PATTERNS:
        result, url = _probe_pattern(entry["main"], dvr_ip, dvr_port, user, password)
        masked = url.replace(password, "***") if password else url

        if result is True:
            print(f"  ✓  {entry['name']}")
            print(f"     {masked}")
            working_entry = entry
            working_url   = url
            break
        elif result is None:
            print(f"  ✗  Auth error on: {masked}")
            print("     DVR responded but rejected credentials — fix DVR_USER / DVR_PASS in .env")
            auth_failed = True
            break
        else:
            print(f"  —  {entry['name']}")

    if auth_failed or not working_entry:
        if not auth_failed:
            print("\n[WARN] No known pattern worked.")
            print("       Check DVR brand/model and add its pattern manually to config.yaml.")
            print("       Or try with VLC: Media → Open Network Stream → test URLs above.")
        sys.exit(1)

    # ── Select main or sub pattern ────────────────────────────────────────────
    sub_unknown = False
    if stream_type == "sub":
        if working_entry["sub"] is not None:
            chosen_pattern = working_entry["sub"]
        else:
            # Sub-stream path unknown for this brand — fall back to main with warning
            chosen_pattern = working_entry["main"]
            sub_unknown    = True
    else:
        chosen_pattern = working_entry["main"]

    # ── Detect resolution from main stream ────────────────────────────────────
    print("\n  Detecting stream resolution via gst-discoverer-1.0…")
    resolution = _get_resolution(working_url)  # always connect to main for reliability

    detected_width, detected_height = resolution if resolution else (1920, 1080)
    res_note = "" if resolution else "  (gst-discoverer not available — defaulting to 1920x1080)"

    # For sub-stream, use 960×544 regardless of what the main stream reports
    if stream_type == "sub":
        width, height = 960, 544
    else:
        width, height = detected_width, detected_height

    # ── Write or print result ─────────────────────────────────────────────────
    if args.update_config:
        cfg_path = REPO_ROOT / "clients" / client_name / "config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        ryaml = _YAML()
        ryaml.preserve_quotes = True
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = ryaml.load(f) or {}
        cfg["dvr_port"]         = dvr_port
        cfg["rtsp_url_pattern"] = chosen_pattern
        cfg["stream_type"]      = stream_type
        cfg["stream_width"]     = width
        cfg["stream_height"]    = height
        with open(cfg_path, "w") as f:
            ryaml.dump(cfg, f)
        print(f"\n  ✓  config.yaml updated: {cfg_path}")
        if res_note:
            print(f"     {res_note.strip()}")
        print(f"     stream_type : {stream_type}  ({width}×{height})")
        if sub_unknown:
            print(f"\n  [WARN] Sub-stream path not known for {working_entry['name']}.")
            print(f"         rtsp_url_pattern was set to the main stream URL.")
            print(f"         Update it manually in {cfg_path}")
            print(f"         Consult your DVR's web interface for the sub-stream RTSP path.")
    else:
        masked_pattern = chosen_pattern.replace(password, "***") if password else chosen_pattern
        print(f"\n{'─'*55}")
        print(f"  DVR identified. Paste this into clients/{client_name}/config.yaml:")
        print(f"{'─'*55}\n")
        print(f"  dvr_port: {dvr_port}")
        print(f"  rtsp_url_pattern: \"{masked_pattern}\"")
        print(f"  stream_type: {stream_type}")
        print(f"  stream_width: {width}")
        print(f"  stream_height: {height}")
        if res_note:
            print(f"  {res_note.strip()}")
        if sub_unknown:
            print(f"\n  [WARN] Sub-stream path not known for {working_entry['name']}.")
            print(f"         rtsp_url_pattern above is the main stream URL.")
            print(f"         Update it manually after checking your DVR's web interface.")
        print(f"\n{'─'*55}")

    print("\n  Next: run probe_cameras.py --update-config to discover active channels.")


if __name__ == "__main__":
    main()
