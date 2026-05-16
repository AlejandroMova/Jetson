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
from queue import Queue, Empty


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
    ):
        super().__init__(daemon=True, name="MjpegServer")
        self._port = port
        self._quality = quality
        self._tiled_queue = tiled_frame_queue
        self._cam_queues = camera_queues        # {"jetson-nx-001-ch01": Queue, ...}
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
                jpeg = self._to_jpeg(frame)
                with self._lock:
                    self._jpegs["all"] = jpeg
            except Empty:
                pass

            # Per-camera crops
            for cam_id, q in list(self._cam_queues.items()):
                try:
                    frame = q.get_nowait()
                    jpeg = self._to_jpeg(frame)
                    with self._lock:
                        self._jpegs[cam_id] = jpeg
                except Empty:
                    pass

    # ------------------------------------------------------------------
    def run(self):
        server = HTTPServer(("0.0.0.0", self._port), self._make_handler())
        server.serve_forever()

    def _make_handler(self):
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Normalize path: /stream/all → key="all"
                #                 /stream/jetson-nx-001-ch01 → key="jetson-nx-001-ch01"
                path = self.path.strip("/")          # "stream/all" or "stream/<cam_id>"
                parts = path.split("/", 1)
                if len(parts) != 2 or parts[0] != "stream":
                    self.send_response(404)
                    self.end_headers()
                    return

                key = parts[1]   # "all" or camera_id
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
                        time.sleep(1 / 25)    # cap at ~25 fps
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass    # silenciar access logs del HTTP server

        return _Handler
