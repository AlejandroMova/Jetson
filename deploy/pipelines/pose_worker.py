"""
pose_worker.py — NX Computing AI | Fall Detection via MoveNet ONNX
Async worker that runs MoveNet pose estimation on person crops and classifies falls.

Architecture: same non-blocking queue pattern as NxApiClient.
  probe → enqueue(crop, track_id, frame_num, bbox)   ← O(1), never blocks
  worker thread → MoveNet → 17 keypoints → fall rules → result_cache[track_id]
  probe (next frame) → get_result(track_id) → draw OSD / emit event
"""
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Fall detection thresholds ─────────────────────────────────────────────────
FALL_TORSO_ANGLE_MAX: float = 45.0   # degrees from vertical; <45° = torso horizontal
FALL_SCORE_THRESHOLD: int   = 2      # how many of 3 rules must fire
FALL_COOLDOWN_SECS: float   = 4.0    # min seconds between fall events per track
FALL_MIN_KP_CONF: float     = 0.25   # keypoints below this conf are ignored

# COCO keypoint indices used for fall detection
_KP_LEFT_SHOULDER  = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_HIP       = 11
_KP_RIGHT_HIP      = 12
_KP_LEFT_ANKLE     = 15
_KP_RIGHT_ANKLE    = 16


@dataclass
class PoseResult:
    """Pose classification result for a single track. Produced by _run_inference()."""
    is_falling: bool  # True when ≥ FALL_SCORE_THRESHOLD rules fired
    fall_score: int   # number of rules that fired (0–3)
    avg_conf: float   # mean confidence across the 6 relevant keypoints


class PoseWorker:
    """
    Runs MoveNet SinglePose Lightning in a background thread.
    Thread-safe enqueue/get_result interface for use from GStreamer probes.
    """

    def __init__(self, model_path: str, queue_size: int = 64):
        """Configura el worker. El modelo se carga en start() por la misma razón que AppearanceWorker.

        _fall_timestamps lleva el tiempo del último evento de caída por track para el cooldown.
        _new_falls es un set: pop_new_fall() lo consume una vez, evitando alertas repetidas.
        """
        self._model_path = model_path
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._results: Dict[int, PoseResult] = {}         # track_id → último resultado de pose
        self._fall_timestamps: Dict[int, float] = {}      # track_id → timestamp último evento fall
        self._new_falls: set = set()                       # tracks con caída nueva no consumida aún
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session = None  # se carga en start() después de que TRT inicialice CUDA

    def start(self):
        """Carga MoveNet ONNX y arranca el hilo worker. Llamar después de pipeline.set_state(PLAYING)."""
        self._session = self._load_model()  # debe correr después de set_state(PLAYING)
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="pose-worker"
        )
        self._thread.start()
        logger.info("PoseWorker started — model: %s", self._model_path)

    def stop(self):
        """Señaliza al worker que pare y espera a que el hilo termine (máximo 5 s)."""
        self._running = False
        self._queue.put(None)  # sentinel para desbloquear el get() en _worker_loop
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PoseWorker stopped.")

    def enqueue(self, crop_bgr: np.ndarray, track_id: int, frame_num: int, bbox: dict):
        """Encola un crop para clasificación de caída. No bloqueante — descarta si la cola está llena."""
        try:
            self._queue.put_nowait((crop_bgr.copy(), track_id, frame_num, bbox))
        except queue.Full:
            pass  # descarte silencioso — el siguiente frame POSE_SAMPLE_INTERVAL lo intentará de nuevo

    def get_result(self, track_id: int) -> Optional[PoseResult]:
        with self._lock:
            return self._results.get(track_id)

    def pop_new_fall(self, track_id: int) -> bool:
        """Returns True once per fall event (debounced by FALL_COOLDOWN_SECS)."""
        with self._lock:
            if track_id not in self._new_falls:
                return False
            self._new_falls.discard(track_id)
            return True

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        """Loop principal del hilo worker: consume la cola y actualiza _results y _new_falls.

        Usa get(timeout=1.0) para poder verificar _running periódicamente sin bloqueo indefinido.
        Al detectar caída: aplica cooldown por track_id antes de añadir a _new_falls, para que
        pop_new_fall() solo retorne True una vez por evento real (no en cada frame donde sigue cayendo).
        """
        if self._session is None:
            logger.error("PoseWorker: failed to load model — worker inactive.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break  # sentinel enviado por stop()
            crop_bgr, track_id, frame_num, bbox = item
            try:
                result = self._run_inference(crop_bgr, bbox)
                with self._lock:
                    self._results[track_id] = result
                    if result.is_falling:
                        now = time.monotonic()
                        last = self._fall_timestamps.get(track_id, 0.0)
                        if now - last >= FALL_COOLDOWN_SECS:  # solo registrar si pasó el cooldown
                            self._fall_timestamps[track_id] = now
                            self._new_falls.add(track_id)      # marcado para consumo único por pop_new_fall()
            except Exception as e:
                logger.warning("PoseWorker inference error track=%d: %s", track_id, e)
            self._queue.task_done()

    def _load_model(self):
        """Carga MoveNet Lightning ONNX en CPU. Retorna la sesión ONNX o None si falla.

        Usa solo CPUExecutionProvider para no competir con TensorRT por el contexto CUDA.
        La importación de onnxruntime es diferida (dentro del método) para que el módulo
        cargue sin error si onnxruntime no está instalado y fall_detection no está activo.
        """
        try:
            import onnxruntime as ort
            path = Path(self._model_path)
            if not path.exists():
                logger.error("MoveNet model not found: %s", path)
                logger.error("Run: python tools/download_models.py --fall-detection")
                return None
            providers = ["CPUExecutionProvider"]
            sess = ort.InferenceSession(str(path), providers=providers)
            logger.info("MoveNet loaded (%s)", sess.get_providers()[0])
            return sess
        except Exception as e:
            logger.error("Failed to load MoveNet: %s", e)
            return None

    def _run_inference(self, crop_bgr: np.ndarray, bbox: dict) -> PoseResult:
        """Run MoveNet on a person crop and classify whether the person is falling.

        Preprocessing: resize to 192×192, BGR→RGB, normalise to [0, 1], NCHW.
        MoveNet output: tensor (1, 1, 17, 3) — 17 COCO keypoints, each (y_norm, x_norm, conf).
        Classification uses 3 geometric rules on the resulting keypoints.
        """
        img = cv2.resize(crop_bgr, (192, 192))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2, 0, 1))[np.newaxis]  # HWC → NCHW: (1, 3, 192, 192)

        inp_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {inp_name: inp})[0]  # (1, 1, 17, 3)

        keypoints = out[0, 0]  # (17, 3): each row = (y_norm, x_norm, confidence)
        score, avg_conf = self._classify_fall(keypoints, bbox)
        return PoseResult(
            is_falling=(score >= FALL_SCORE_THRESHOLD),  # fall if ≥2 of 3 rules fire
            fall_score=score,
            avg_conf=avg_conf,
        )

    @staticmethod
    def _classify_fall(kps: np.ndarray, bbox: dict) -> tuple:
        """
        3-rule geometric fall classifier.
        Returns (fall_score 0-3, avg_keypoint_confidence).
        """
        def kp(idx):
            y, x, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
            return x, y, c

        left_shoulder_x,  left_shoulder_y,  left_shoulder_conf  = kp(_KP_LEFT_SHOULDER)
        right_shoulder_x, right_shoulder_y, right_shoulder_conf = kp(_KP_RIGHT_SHOULDER)
        left_hip_x,       left_hip_y,       left_hip_conf       = kp(_KP_LEFT_HIP)
        right_hip_x,      right_hip_y,      right_hip_conf      = kp(_KP_RIGHT_HIP)
        left_ankle_x,     left_ankle_y,     left_ankle_conf     = kp(_KP_LEFT_ANKLE)
        right_ankle_x,    right_ankle_y,    right_ankle_conf    = kp(_KP_RIGHT_ANKLE)

        confs = [left_shoulder_conf, right_shoulder_conf,
                 left_hip_conf,      right_hip_conf,
                 left_ankle_conf,    right_ankle_conf]
        avg_conf = float(np.mean(confs))

        def has_confidence(c): return c >= FALL_MIN_KP_CONF

        score = 0

        # Rule 1: torso angle from vertical — mid-shoulder to mid-hip vector
        if has_confidence(left_shoulder_conf) and has_confidence(right_shoulder_conf) \
                and has_confidence(left_hip_conf) and has_confidence(right_hip_conf):
            shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
            shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2
            hip_mid_x = (left_hip_x + right_hip_x) / 2
            hip_mid_y = (left_hip_y + right_hip_y) / 2
            dx = hip_mid_x - shoulder_mid_x
            dy = hip_mid_y - shoulder_mid_y
            if dy != 0 or dx != 0:
                angle_from_vertical = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
                if angle_from_vertical > (90 - FALL_TORSO_ANGLE_MAX):
                    score += 1

        # Rule 2: bounding box aspect ratio (width > height means person is lying down)
        w = bbox.get("width", 1)
        h = bbox.get("height", 1)
        if h > 0 and w / h > 1.0:
            score += 1

        # Rule 3: hip Y close to ankle Y (hips near ground level)
        if has_confidence(left_hip_conf) and has_confidence(right_hip_conf) \
                and has_confidence(left_ankle_conf) and has_confidence(right_ankle_conf):
            hip_mid_y   = (left_hip_y   + right_hip_y)   / 2
            ankle_mid_y = (left_ankle_y + right_ankle_y) / 2
            if ankle_mid_y > 0 and hip_mid_y >= ankle_mid_y * 0.80:
                score += 1

        return score, avg_conf
