"""
mjpeg_server.py — NX Computing AI | QA Visual

HTTP MJPEG server for the QA dashboard. Exposes:
  /stream/all      — 640×360 tiled view with bounding boxes for all cameras
  /stream/<cam_id> — full-resolution crop for a single camera
  /viewer/<key>    — minimal HTML page that embeds the stream and auto-reconnects

Only instantiated when NX_QA_ENABLED=true. Zero impact in production.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty, Queue
from socketserver import ThreadingMixIn
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from recording_manager import RecordingManager


class MjpegServer(threading.Thread):
    """Daemon thread that serves MJPEG video streams over HTTP.

    Two-thread architecture:
      - MjpegEncoder thread: drains frame queues, encodes to JPEG, stores in _jpegs dict.
      - MjpegServer thread (run()): handles HTTP. Each /stream/<key> client gets the
        latest JPEG for that key in a multipart/x-mixed-replace loop.

    Encoding always runs in the background; the HTTP handler only reads the most recent
    JPEG (minimal lock contention — the pipeline is never blocked by HTTP clients).
    """

    def __init__(
        self,
        tiled_frame_queue: Queue,
        camera_queues: dict,
        port: int = 8080,
        quality: int = 72,
        recorder: "RecordingManager | None" = None,
    ):
        """
        Args:
            tiled_frame_queue: 640×360 frames from the tiler (served as "all" stream).
            camera_queues: camera_id → Queue of full-res crops (one stream per camera).
            port: HTTP port to bind. Must match the port exposed in docker-compose.qa.yml.
            quality: JPEG encoding quality (1–100). 72 balances bandwidth and clarity.
            recorder: if provided, each tiled frame is forwarded to push_tiled_frame()
                      so the RecordingManager can write tiled.mp4.
        """
        super().__init__(daemon=True, name="MjpegServer")
        self._port = port
        self._quality = quality
        self._tiled_queue = tiled_frame_queue
        self._cam_queues = camera_queues
        self._recorder = recorder
        self._lock = threading.Lock()
        self._jpegs: dict = {}  # "all" | camera_id → most recent JPEG bytes
        self._enc_thread = threading.Thread(
            target=self._encode_loop, daemon=True, name="MjpegEncoder"
        )

    def start(self) -> None:
        self._enc_thread.start()
        super().start()

    def _to_jpeg(self, frame: np.ndarray) -> bytes:
        """Convert a BGR or RGBA numpy frame to JPEG bytes."""
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return buf.tobytes()

    def _encode_loop(self) -> None:
        """Background thread: drain frame queues and keep _jpegs up to date.

        Processes the tiled queue first (with a 40 ms timeout to pace at ~25 fps),
        then drains per-camera queues without blocking.
        """
        while True:
            # Tiled frame → stored under "all" key
            try:
                frame = self._tiled_queue.get(timeout=0.04)
                try:
                    jpeg = self._to_jpeg(frame)
                    with self._lock:
                        self._jpegs["all"] = jpeg
                except Exception:
                    # Corrupt or incompatible frame — skip silently rather than crashing
                    # the MJPEG server. The next frame will replace it.
                    pass
                # Forward to RecordingManager without making an extra copy —
                # the frame is already a copy from the probe.
                if self._recorder is not None:
                    self._recorder.push_tiled_frame(frame)
            except Empty:
                pass

            # Per-camera crops
            for cam_id, q in list(self._cam_queues.items()):
                try:
                    frame = q.get_nowait()
                    try:
                        jpeg = self._to_jpeg(frame)
                        with self._lock:
                            self._jpegs[cam_id] = jpeg
                    except Exception:
                        pass  # same rationale as tiled frame above
                except Empty:
                    pass

    def run(self) -> None:
        """Start the HTTP server and serve requests until the process exits.

        Uses ThreadingMixIn so each MJPEG connection runs in its own thread,
        allowing multiple simultaneous clients (e.g. laptop + phone on the same network).
        """
        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            # daemon_threads=True: client threads don't block process shutdown
            daemon_threads = True

        server = _ThreadingHTTPServer(("0.0.0.0", self._port), self._make_handler())
        server.serve_forever()

    def _make_handler(self):
        """Build the HTTP handler class with a closure reference to this server.

        A closure is used instead of passing self directly because BaseHTTPRequestHandler
        instantiates a new object per request — there's no clean way to inject a
        dependency without inheritance or a closure.
        """
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            """HTTP handler for /viewer/<key> and /stream/<key> endpoints.

            One instance is created per connection by ThreadingHTTPServer.
            Reads JPEG frames from the enclosing MjpegServer via the `srv` closure.
            """

            def do_GET(self) -> None:
                """Route GET requests by URL prefix.

                /viewer/<key> → Serve an HTML page that embeds the stream.
                /stream/<key> → Stream MJPEG via multipart/x-mixed-replace at ~25 fps.

                The MJPEG loop runs until the client disconnects (BrokenPipeError,
                ConnectionResetError, or OSError on wfile.write).
                """
                path = self.path.strip("/")
                parts = path.split("/", 1)
                if len(parts) != 2:
                    self.send_response(404)
                    self.end_headers()
                    return

                prefix, key = parts[0], parts[1]

                if prefix == "viewer":
                    self._serve_viewer(key)
                    return

                if prefix != "stream":
                    self.send_response(404)
                    self.end_headers()
                    return

                self._serve_stream(key)

            def _serve_viewer(self, key: str) -> None:
                """Serve an HTML page with an auto-reconnecting MJPEG image.

                The JavaScript onerror handler retries the image src every 2 s with a
                cache-buster (?t=timestamp) so the stream resumes automatically after
                the pipeline restarts for playback mode — without requiring a page reload.
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
                """Stream MJPEG frames for the given key until the client disconnects."""
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
                pass  # silence per-request access logs to keep container output clean

        return _Handler
