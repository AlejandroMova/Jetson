"""
mjpeg_server.py — NX Computing AI | Stream Mode MJPEG Server

Sirve el frame tileado (640×360, todas las cámaras compuestas con bboxes/labels)
como MJPEG sobre HTTP. Solo se instancia cuando NX_STREAM_ENABLED=true.

Endpoints:
  GET /stream/all   — MJPEG multipart/x-mixed-replace a ~25 fps
  GET /viewer/all   — HTML mínimo con <img> + auto-reconexión JS

Arquitectura de dos threads:
  MjpegEncoder (daemon): drena tiled_frame_queue, codifica JPEG, guarda en _jpeg.
  MjpegServer  (daemon): corre el HTTP server; cada cliente lee _jpeg en loop.

El probe GStreamer hace put_nowait() en tiled_frame_queue — nunca bloquea el pipeline.
El encoder hace get() con timeout — si no llegan frames, simplemente espera.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty, Queue
from socketserver import ThreadingMixIn

import cv2
import numpy as np


class MjpegServer(threading.Thread):
    """Daemon thread que sirve el stream tileado de todas las cámaras como MJPEG.

    Solo se usa cuando NX_STREAM_ENABLED=true. En producción este objeto no existe.

    Args:
        tiled_queue: Queue(maxsize=1) poblada por tiled_overlay_probe con frames BGR 640×360.
        port:        Puerto HTTP (default 8080, debe coincidir con docker-compose).
        quality:     Calidad JPEG 1-100. 72 balancea ancho de banda y nitidez.
    """

    def __init__(self, tiled_queue: Queue, port: int = 8080, quality: int = 72):
        super().__init__(daemon=True, name="MjpegServer")
        self._queue   = tiled_queue
        self._port    = port
        self._quality = quality
        self._lock    = threading.Lock()
        self._jpeg: bytes | None = None   # último frame codificado
        self._enc_thread = threading.Thread(
            target=self._encode_loop, daemon=True, name="MjpegEncoder"
        )

    def start(self) -> None:
        self._enc_thread.start()
        super().start()

    # ── Encoder ───────────────────────────────────────────────────────────────

    def _encode_loop(self) -> None:
        """Hilo de fondo: drena tiled_queue, codifica JPEG, actualiza self._jpeg."""
        while True:
            try:
                frame = self._queue.get(timeout=0.04)  # 40 ms → ~25 fps máximo
            except Empty:
                continue
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
            )
            with self._lock:
                self._jpeg = buf.tobytes()

    # ── HTTP server ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Arranca el HTTP server en self._port."""
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # silenciar logs de acceso

            def do_GET(self):
                if self.path in ("/stream/all", "/stream/all/"):
                    self._serve_stream()
                elif self.path in ("/viewer/all", "/viewer/all/"):
                    self._serve_viewer()
                else:
                    self.send_error(404)

            def _serve_stream(self):
                """Multipart MJPEG — compatible con <img src="..."> y VLC."""
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=--nxframe",
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        with server._lock:
                            jpeg = server._jpeg
                        if jpeg is None:
                            time.sleep(0.04)
                            continue
                        header = (
                            "--nxframe\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n"
                        ).encode()
                        self.wfile.write(header + jpeg + b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.04)  # ~25 fps
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _serve_viewer(self):
                """Página HTML mínima con auto-reconexión si el stream cae."""
                html = (
                    "<!DOCTYPE html><html><head>"
                    "<meta charset='utf-8'>"
                    "<title>NX Stream</title>"
                    "<style>body{margin:0;background:#111;display:flex;"
                    "justify-content:center;align-items:center;height:100vh}"
                    "img{max-width:100%;max-height:100vh}</style>"
                    "</head><body>"
                    "<img id='s' src='/stream/all'>"
                    "<script>"
                    "var img=document.getElementById('s');"
                    "img.onerror=function(){setTimeout(function(){"
                    "img.src='/stream/all?t='+Date.now();},2000);};"
                    "</script>"
                    "</body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

        class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        httpd = _ThreadedHTTPServer(("0.0.0.0", self._port), _Handler)
        httpd.serve_forever()
