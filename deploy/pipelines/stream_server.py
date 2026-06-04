"""
stream_server.py — NX Computing AI

HTTP MJPEG server for stream mode (NX_STREAM_ENABLED=true).
Serves one stream per camera with bounding-box overlays drawn by the probe.
Only instantiated when NX_STREAM_ENABLED=true. Zero impact in production.

Activated via ./stream.sh. Accessible at:
  /stream/<camera_id>  — MJPEG live stream with bboxes
  /viewer/<camera_id>  — HTML page that embeds the stream and auto-reconnects
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty, Queue
from socketserver import ThreadingMixIn

import cv2
import numpy as np


class StreamServer(threading.Thread):
    """Daemon thread that serves per-camera MJPEG streams over HTTP.

    Two-thread architecture:
      - MjpegEncoder thread: drains per-camera queues, encodes to JPEG, stores in _jpegs dict.
      - StreamServer thread (run()): handles HTTP. Each /stream/<cam_id> client gets the
        latest JPEG for that camera in a multipart/x-mixed-replace loop.

    Encoding runs in the background; HTTP handler only reads the most recent JPEG
    so the GStreamer pipeline is never blocked by HTTP clients.
    """

    def __init__(self, camera_queues: dict, port: int = 8080, quality: int = 72):
        """
        Args:
            camera_queues: camera_id → Queue of BGR frames from the stream probe.
            port: HTTP port to bind. Must match the port exposed in docker-compose.yml.
            quality: JPEG encoding quality (1–100). 72 balances bandwidth and clarity.
        """
        super().__init__(daemon=True, name="StreamServer")
        self._port = port
        self._quality = quality
        self._cam_queues = camera_queues
        self._lock = threading.Lock()
        self._jpegs: dict = {}  # camera_id → most recent JPEG bytes
        self._enc_thread = threading.Thread(
            target=self._encode_loop, daemon=True, name="StreamEncoder"
        )

    def start(self) -> None:
        self._enc_thread.start()
        super().start()

    def _to_jpeg(self, frame: np.ndarray) -> bytes:
        """Convert a BGR numpy frame to JPEG bytes."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return buf.tobytes()

    def _encode_loop(self) -> None:
        """Background thread: drain per-camera queues and keep _jpegs up to date."""
        while True:
            any_frame = False
            for cam_id, q in list(self._cam_queues.items()):
                try:
                    frame = q.get_nowait()
                    any_frame = True
                    try:
                        jpeg = self._to_jpeg(frame)
                        with self._lock:
                            self._jpegs[cam_id] = jpeg
                    except Exception:
                        # Corrupt or incompatible frame — skip silently, next frame replaces it.
                        pass
                except Empty:
                    pass
            if not any_frame:
                # No frames available — sleep briefly to avoid busy-waiting
                time.sleep(0.01)

    def run(self) -> None:
        """Start the HTTP server. Uses ThreadingMixIn for simultaneous clients."""

        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        server = _ThreadingHTTPServer(("0.0.0.0", self._port), self._make_handler())
        server.serve_forever()

    def _make_handler(self):
        """Build the HTTP handler with a closure reference to this server."""
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            """HTTP handler for /viewer/<cam_id> and /stream/<cam_id>."""

            def do_GET(self) -> None:
                path = self.path.strip("/")
                parts = path.split("/", 1)
                if len(parts) != 2:
                    self.send_response(404)
                    self.end_headers()
                    return

                prefix, key = parts[0], parts[1]

                if prefix == "viewer":
                    self._serve_viewer(key)
                elif prefix == "stream":
                    self._serve_stream(key)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_viewer(self, key: str) -> None:
                """Serve an HTML page with an auto-reconnecting MJPEG image.

                img.onerror retries every 2 s with a cache-buster so the stream
                resumes automatically if the pipeline restarts.
                """
                html = (
                    "<!DOCTYPE html><html><head>"
                    "<style>*{margin:0;padding:0;box-sizing:border-box}"
                    "body{background:#111}"
                    "img{width:100%;display:block}</style></head>"
                    f"<body><img id='s' src='/stream/{key}'>"
                    "<script>(function(){"
                    "var img=document.getElementById('s');"
                    f"function retry(){{setTimeout(function(){{img.src='/stream/{key}?t='+Date.now();}},2000);}}"
                    "img.onerror=retry;"
                    "})();</script>"
                    "</body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _serve_stream(self, key: str) -> None:
                """Stream MJPEG frames for the given camera until the client disconnects."""
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=nxframe"
                )
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    while True:
                        with srv._lock:
                            jpeg = srv._jpegs.get(key, b"")
                        if jpeg:
                            self.wfile.write(
                                b"--nxframe\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                        time.sleep(1 / 25)  # cap at 25 fps
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # client disconnected — normal end of stream

            def log_message(self, *_) -> None:
                pass  # silence per-request access logs

        return _Handler
