"""
recording_manager.py — NX Computing AI | QA Recording

Graba clips de video automáticamente cuando se detectan personas en el pipeline QA.
Solo se instancia cuando NX_QA_ENABLED=true. Cero impacto en producción.

Fuentes de frames:
  - Tiled (640×360)   : push_tiled_frame()   → llamado desde MjpegServer._encode_loop
  - Por cámara full-res: push_camera_frame() → llamado desde Probe A (pre_tiler_analytics_probe)

Trigger: suscripción a nx:qa:detections vía Redis pub/sub.
"""

import json
import logging
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


class RecordingManager:
    """
    Graba clips de video cuando se detectan personas en el pipeline QA.

    Estado interno: IDLE → (detección) → RECORDING → (cooldown/maxdur) → IDLE

    Al superar max_storage_gb, elimina los clips más antiguos automáticamente.
    """

    def __init__(
        self,
        recordings_dir: str,
        redis_client,
        fps: float = 25.0,
        max_clip_minutes: float = 5.0,
        min_clip_seconds: float = 5.0,
        cooldown_seconds: float = 10.0,
        max_storage_gb: float = 10.0,
    ):
        self._dir = Path(recordings_dir)
        self._redis = redis_client
        self._fps = fps
        self._max_clip_secs = max_clip_minutes * 60.0
        self._min_clip_secs = min_clip_seconds
        self._cooldown_secs = cooldown_seconds
        self._max_bytes = int(max_storage_gb * 1024 ** 3)

        # Queues para recibir frames de otros hilos (maxsize=2: descarta si el loop es lento)
        self._tiled_q: queue.Queue = queue.Queue(maxsize=2)
        self._cam_qs: Dict[str, queue.Queue] = {}

        # Timestamp de la última detección — escrito por el hilo watcher, leído por record loop.
        # La asignación de float a atributo es atómica en CPython (GIL + STORE_ATTR).
        self._last_det_time: Optional[float] = None
        self._max_people: int = 0

        # Estado del clip actual (solo escrito por _record_loop)
        self._clip_dir: Optional[Path] = None
        self._clip_start: Optional[float] = None
        self._clip_ts: str = ""
        self._tiled_writer: Optional[cv2.VideoWriter] = None
        self._cam_writers: Dict[str, cv2.VideoWriter] = {}
        self._frame_count: int = 0
        self._thumbnail_saved: bool = False

        self._recording = False
        self._running = False

        self._rec_thread = threading.Thread(
            target=self._record_loop, daemon=True, name="RecordLoop"
        )

    # ── API pública (llamada desde otros hilos) ───────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    def push_tiled_frame(self, frame: np.ndarray) -> None:
        """Llamado desde MjpegServer._encode_loop cuando hay un frame tileado listo."""
        if not self._recording:
            return
        try:
            self._tiled_q.put_nowait(frame)
        except queue.Full:
            pass

    def push_camera_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Llamado desde Probe A cuando hay un frame full-res por cámara."""
        if not self._recording:
            return
        if camera_id not in self._cam_qs:
            self._cam_qs[camera_id] = queue.Queue(maxsize=2)
        try:
            self._cam_qs[camera_id].put_nowait(frame)
        except queue.Full:
            pass

    def notify_detection(self, people_count: int = 1) -> None:
        """Llamado por _detection_watcher cuando se detectan personas."""
        self._last_det_time = time.monotonic()
        if people_count > self._max_people:
            self._max_people = people_count

    def start(self) -> None:
        self._running = True
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rec_thread.start()
        logger.info("[Recording] Iniciado — directorio: %s", self._dir)

    def stop(self) -> None:
        self._running = False
        if self._recording:
            self._finish()

    # ── Loop de grabación ─────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        while self._running:
            if not self._recording:
                if (self._last_det_time is not None
                        and time.monotonic() - self._last_det_time < 1.0):
                    self._start()
                else:
                    time.sleep(0.05)
                continue

            # RECORDING — evaluar condiciones de corte
            now = time.monotonic()
            idle_secs = now - (self._last_det_time or 0)
            elapsed_secs = now - (self._clip_start or now)

            if idle_secs > self._cooldown_secs or elapsed_secs > self._max_clip_secs:
                if elapsed_secs >= self._min_clip_secs:
                    self._finish()
                else:
                    self._cancel()
                continue

            # Drenar queue tileado
            try:
                frame = self._tiled_q.get(timeout=0.04)
                self._write_tiled(frame)
            except queue.Empty:
                pass

            # Drenar queues por cámara
            for cam_id, q in list(self._cam_qs.items()):
                try:
                    frame = q.get_nowait()
                    self._write_camera(cam_id, frame)
                except queue.Empty:
                    pass

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _start(self) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._clip_dir = self._dir / ts
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        self._clip_start = time.monotonic()
        self._clip_ts = ts
        self._frame_count = 0
        self._max_people = 0
        self._thumbnail_saved = False
        self._tiled_writer = None
        self._cam_writers = {}
        self._cam_qs = {}
        self._recording = True
        self._publish_state(active=True, clip_ts=ts)
        logger.info("[Recording] Clip iniciado: %s", ts)

    def _write_tiled(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._tiled_writer is None:
            path = str(self._clip_dir / "tiled.mp4")
            self._tiled_writer = cv2.VideoWriter(path, _FOURCC, self._fps, (w, h))
        if not self._thumbnail_saved:
            cv2.imwrite(
                str(self._clip_dir / "thumbnail.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            self._thumbnail_saved = True
        self._tiled_writer.write(frame)
        self._frame_count += 1

    def _write_camera(self, cam_id: str, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if cam_id not in self._cam_writers:
            path = str(self._clip_dir / f"{cam_id}.mp4")
            self._cam_writers[cam_id] = cv2.VideoWriter(path, _FOURCC, self._fps, (w, h))
        self._cam_writers[cam_id].write(frame)

    def _finish(self) -> None:
        self._recording = False
        elapsed = time.monotonic() - (self._clip_start or 0)
        if self._tiled_writer:
            self._tiled_writer.release()
        for w in self._cam_writers.values():
            w.release()
        meta = {
            "timestamp": self._clip_ts,
            "duration_s": round(elapsed, 1),
            "channels": list(self._cam_writers.keys()),
            "frame_count": self._frame_count,
            "max_people": self._max_people,
        }
        try:
            with open(self._clip_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning("[Recording] Error guardando metadata: %s", e)
        logger.info(
            "[Recording] Clip terminado: %s (%.1f s, %d frames tileados)",
            self._clip_ts, elapsed, self._frame_count,
        )
        self._publish_state(active=False)
        self._prune()
        self._reset_writers()

    def _cancel(self) -> None:
        self._recording = False
        elapsed = time.monotonic() - (self._clip_start or 0)
        if self._tiled_writer:
            self._tiled_writer.release()
        for w in self._cam_writers.values():
            w.release()
        if self._clip_dir and self._clip_dir.exists():
            shutil.rmtree(self._clip_dir, ignore_errors=True)
        logger.info(
            "[Recording] Clip descartado (%.1f s < mínimo %.1f s)",
            elapsed, self._min_clip_secs,
        )
        self._publish_state(active=False)
        self._reset_writers()

    def _reset_writers(self) -> None:
        self._tiled_writer = None
        self._cam_writers = {}
        self._cam_qs = {}
        self._clip_dir = None
        self._clip_start = None

    def _publish_state(self, active: bool, clip_ts: str = "") -> None:
        if not self._redis:
            return
        try:
            self._redis.set("nx:qa:recording_active", "1" if active else "0")
            if active:
                self._redis.set("nx:qa:recording_info", json.dumps({
                    "started": clip_ts,
                    "clip_name": clip_ts,
                    "people_count": self._max_people,
                }))
            else:
                self._redis.delete("nx:qa:recording_info")
        except Exception:
            pass

    def _prune(self) -> None:
        """Elimina los clips más antiguos si el total supera max_storage_gb."""
        try:
            clips = sorted(d for d in self._dir.iterdir() if d.is_dir())
            total = sum(
                f.stat().st_size
                for clip in clips
                for f in clip.rglob("*")
                if f.is_file()
            )
            for clip in clips:
                if total <= self._max_bytes:
                    break
                size = sum(f.stat().st_size for f in clip.rglob("*") if f.is_file())
                shutil.rmtree(clip, ignore_errors=True)
                total -= size
                logger.info("[Recording] Clip eliminado por espacio: %s", clip.name)
        except Exception as e:
            logger.warning("[Recording] Error en prune de almacenamiento: %s", e)

