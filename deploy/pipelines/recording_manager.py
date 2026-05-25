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
        """Configura el grabador. El hilo de grabación se arranca en start().

        redis_client se usa solo para publicar estado (nx:qa:recording_active/info).
        Puede ser None en producción sin QA — la grabación funciona igual pero sin publicar a Redis.
        Los VideoWriters se crean lazily en _write_tiled() / _write_camera() al recibir el primer frame,
        porque necesitan las dimensiones reales del frame para inicializarse correctamente.
        """
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
        """True si hay un clip en progreso (entre _start() y _finish()/_cancel()).

        Leído por Probe A y MjpegServer para decidir si pasar frames al recorder.
        La asignación de bool a atributo es atómica en CPython (GIL protege la lectura).
        """
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
        """Crea el directorio de grabaciones y arranca el hilo de grabación."""
        self._running = True
        self._dir.mkdir(parents=True, exist_ok=True)  # crear /nx_tech/recordings/ si no existe
        self._rec_thread.start()
        logger.info("[Recording] Iniciado — directorio: %s", self._dir)

    def stop(self) -> None:
        """Detiene el hilo de grabación, finalizando el clip actual si hay uno en progreso."""
        self._running = False
        if self._recording:
            self._finish()  # cerrar VideoWriters y guardar metadata antes de salir

    # ── Loop de grabación ─────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        """Loop principal del hilo de grabación.

        Máquina de estados simple: IDLE ↔ RECORDING.
        En IDLE: espera a que notify_detection() indique presencia de personas.
        En RECORDING: drena las queues de frames y evalúa condiciones de corte:
          - cooldown_seconds sin detecciones → _finish() (guarda el clip)
          - max_clip_minutes superados → _finish() (clip máximo alcanzado)
          - duración < min_clip_seconds → _cancel() (descarta el clip por muy corto)
        """
        while self._running:
            if not self._recording:
                # IDLE — verificar si hay una detección reciente para iniciar grabación
                if (self._last_det_time is not None
                        and time.monotonic() - self._last_det_time < 1.0):
                    self._start()       # arrancar nuevo clip
                else:
                    time.sleep(0.05)   # polling liviano para no quemar CPU
                continue

            # ── RECORDING — evaluar condiciones de corte ──────────────────────
            now = time.monotonic()
            idle_secs    = now - (self._last_det_time or 0)   # tiempo sin detecciones
            elapsed_secs = now - (self._clip_start or now)    # duración del clip actual

            if idle_secs > self._cooldown_secs or elapsed_secs > self._max_clip_secs:
                if elapsed_secs >= self._min_clip_secs:
                    self._finish()   # clip válido — guardar
                else:
                    self._cancel()  # clip demasiado corto — descartar
                continue

            # ── Drenar queue del frame tileado (640×360) ──────────────────────
            # timeout=0.04 s ≈ 25 fps — ritmo de grabación del tiled.mp4
            try:
                frame = self._tiled_q.get(timeout=0.04)
                self._write_tiled(frame)
            except queue.Empty:
                pass

            # ── Drenar queues de frames full-res por cámara ───────────────────
            for cam_id, q in list(self._cam_qs.items()):
                try:
                    frame = q.get_nowait()      # no bloquear — si no hay frame, seguir
                    self._write_camera(cam_id, frame)
                except queue.Empty:
                    pass

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _start(self) -> None:
        """Inicia un nuevo clip: crea el directorio con timestamp y resetea contadores.

        El timestamp en el nombre del directorio es la fuente de verdad para la UI de Streamlit
        (se muestra en la tab Grabaciones y en Redis nx:qa:recording_info).
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")      # formato legible para la UI
        self._clip_dir = self._dir / ts                          # directorio único por clip
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        self._clip_start = time.monotonic()                      # para medir duración del clip
        self._clip_ts = ts
        self._frame_count = 0
        self._max_people = 0
        self._thumbnail_saved = False
        self._tiled_writer = None                                # se crea lazily en _write_tiled
        self._cam_writers = {}                                   # se crean lazily en _write_camera
        self._cam_qs = {}                                        # resetear queues del clip anterior
        self._recording = True                                   # activar flag ANTES de publicar estado
        self._publish_state(active=True, clip_ts=ts)
        logger.info("[Recording] Clip iniciado: %s", ts)

    def _write_tiled(self, frame: np.ndarray) -> None:
        """Escribe un frame al video tileado (640×360). Crea el VideoWriter lazily en el primer frame.

        El primer frame también se guarda como thumbnail.jpg para la preview en Streamlit.
        """
        h, w = frame.shape[:2]
        if self._tiled_writer is None:
            # Crear VideoWriter en el primer frame — necesitamos las dimensiones reales
            path = str(self._clip_dir / "tiled.mp4")
            self._tiled_writer = cv2.VideoWriter(path, _FOURCC, self._fps, (w, h))

        # Guardar el primer frame como thumbnail para la galería de Streamlit
        if not self._thumbnail_saved:
            cv2.imwrite(
                str(self._clip_dir / "thumbnail.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            self._thumbnail_saved = True

        self._tiled_writer.write(frame)
        self._frame_count += 1  # contar frames para metadata.json

    def _write_camera(self, cam_id: str, frame: np.ndarray) -> None:
        """Escribe un frame full-res al video de una cámara específica. Crea el VideoWriter lazily."""
        h, w = frame.shape[:2]
        if cam_id not in self._cam_writers:
            # Crear VideoWriter específico para esta cámara — nombre = camera_id.mp4
            path = str(self._clip_dir / f"{cam_id}.mp4")
            self._cam_writers[cam_id] = cv2.VideoWriter(path, _FOURCC, self._fps, (w, h))
        self._cam_writers[cam_id].write(frame)

    def _finish(self) -> None:
        """Finaliza el clip actual: cierra VideoWriters, guarda metadata.json, poda almacenamiento."""
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
        """Descarta el clip actual: cierra VideoWriters y elimina el directorio completo.

        Se llama cuando el clip terminó antes de alcanzar min_clip_seconds.
        Clips muy cortos (ej. < 5 s) suelen ser falsos positivos del detector.
        """
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
        """Limpia todos los VideoWriters y el estado del clip actual tras _finish() o _cancel()."""
        self._tiled_writer = None   # será recreado lazily en el próximo _start()
        self._cam_writers = {}      # idem — un writer por cámara
        self._cam_qs = {}           # limpiar queues del clip anterior
        self._clip_dir = None
        self._clip_start = None

    def _publish_state(self, active: bool, clip_ts: str = "") -> None:
        """Publica el estado de grabación a Redis para que Streamlit lo muestre en tiempo real.

        nx:qa:recording_active → "1" si grabando, "0" si no.
        nx:qa:recording_info   → JSON con metadata del clip en progreso (solo cuando active=True).
        """
        if not self._redis:
            return   # sin Redis (producción sin QA) — silencioso
        try:
            self._redis.set("nx:qa:recording_active", "1" if active else "0")
            if active:
                # Publicar metadata del clip para mostrar en el sidebar de Streamlit
                self._redis.set("nx:qa:recording_info", json.dumps({
                    "started": clip_ts,
                    "clip_name": clip_ts,
                    "people_count": self._max_people,
                }))
            else:
                # Al terminar la grabación, eliminar la key de info del clip
                self._redis.delete("nx:qa:recording_info")
        except Exception:
            pass  # silencioso — Redis caído no debe detener la grabación

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

