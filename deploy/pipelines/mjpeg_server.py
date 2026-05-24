"""
mjpeg_server.py — NX Computing AI | QA Visual

Servidor HTTP MJPEG que sirve:
  /stream/all      → frame tileado completo (640×360) con todos los bboxes dibujados
  /stream/<cam_id> → crop individual de esa cámara (e.g. /stream/jetson-nx-001-ch01)

Solo se instancia cuando NX_QA_ENABLED=true (arrancado desde app.py).
Cero impacto en producción cuando NX_QA_ENABLED no está activo.
"""

import threading
import time
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from queue import Queue, Empty
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recording_manager import RecordingManager


class MjpegServer(threading.Thread):
    """
    Hilo daemon que expone streams MJPEG por HTTP.

    Arquitectura de dos hilos:
      - _enc_thread: consume las queues de frames, encoda a JPEG y guarda en _jpegs dict.
      - run() (este hilo): sirve HTTP. Cada cliente que pide /stream/<key> recibe el
        latest_jpeg para esa key en un bucle multipart/x-mixed-replace.

    El encoding es siempre en background; el HTTP handler solo lee el último JPEG
    disponible (lock mínimo, sin bloquear el pipeline).
    """

    def __init__(
        self,
        tiled_frame_queue: Queue,
        camera_queues: dict,        # camera_id (str) → Queue
        port: int = 8080,
        quality: int = 72,
        recorder: "RecordingManager | None" = None,
    ):
        super().__init__(daemon=True, name="MjpegServer")
        self._port = port
        self._quality = quality
        self._tiled_queue = tiled_frame_queue
        self._cam_queues = camera_queues        # {"jetson-nx-001-ch01": Queue, ...}
        self._recorder = recorder               # RecordingManager (QA recording, optional)
        self._lock = threading.Lock()
        self._jpegs: dict = {}                  # "all" | camera_id → bytes
        self._enc_thread = threading.Thread(
            target=self._encode_loop, daemon=True, name="MjpegEncoder"
        )

    # ------------------------------------------------------------------
    def start(self):
        self._enc_thread.start()
        super().start()

    # ------------------------------------------------------------------
    def _to_jpeg(self, frame: np.ndarray) -> bytes:
        """Convert BGR or RGBA numpy frame to JPEG bytes."""
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return buf.tobytes()

    def _encode_loop(self):
        """Background thread: drains frame queues and updates _jpegs."""
        while True:
            # Tiled frame → "all"
            try:
                frame = self._tiled_queue.get(timeout=0.04)
                try:
                    jpeg = self._to_jpeg(frame)
                    with self._lock:
                        self._jpegs["all"] = jpeg
                except Exception:
                    pass
                # Pasar al recorder si está grabando (no copia extra: el frame ya es copia del probe)
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
                        pass
                except Empty:
                    pass

    # ------------------------------------------------------------------
    def run(self):
        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True  # no esperar threads de cliente al shutdown
        server = _ThreadingHTTPServer(("0.0.0.0", self._port), self._make_handler())
        server.serve_forever()

    def _make_handler(self):
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.strip("/")
                parts = path.split("/", 1)
                if len(parts) != 2:
                    self.send_response(404); self.end_headers(); return

                prefix, key = parts[0], parts[1]

                # /viewer/<key> → HTML page con <img> que apunta al stream
                # (mismo origen → sin CORS; el browser maneja MJPEG nativamente)
                if prefix == "viewer":
                    html = (
                        "<!DOCTYPE html><html><head>"
                        "<style>*{margin:0;padding:0;box-sizing:border-box}"
                        "body{background:#111}"
                        "img{width:100%;display:block}</style></head>"
                        f"<body><img src='/stream/{key}'></body></html>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return

                if prefix != "stream":
                    self.send_response(404); self.end_headers(); return

                # /stream/<key> → MJPEG multipart
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
                        time.sleep(1 / 25)    # cap at ~25 fps
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            def log_message(self, *_):
                pass    # silenciar access logs del HTTP server

        return _Handler
