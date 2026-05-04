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
    is_falling: bool
    fall_score: int
    avg_conf: float


class PoseWorker:
    """
    Runs MoveNet SinglePose Lightning in a background thread.
    Thread-safe enqueue/get_result interface for use from GStreamer probes.
    """

    def __init__(self, model_path: str, queue_size: int = 64):
        self._model_path = model_path
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._results: Dict[int, PoseResult] = {}
        self._fall_timestamps: Dict[int, float] = {}
        self._new_falls: set = set()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="pose-worker"
        )
        self._thread.start()
        logger.info("PoseWorker started — model: %s", self._model_path)

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PoseWorker stopped.")

    def enqueue(self, crop_bgr: np.ndarray, track_id: int, frame_num: int, bbox: dict):
        try:
            self._queue.put_nowait((crop_bgr.copy(), track_id, frame_num, bbox))
        except queue.Full:
            pass  # drop silently — next frame will enqueue again

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
        self._session = self._load_model()
        if self._session is None:
            logger.error("PoseWorker: failed to load model — worker inactive.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            crop_bgr, track_id, frame_num, bbox = item
            try:
                result = self._run_inference(crop_bgr, bbox)
                with self._lock:
                    self._results[track_id] = result
                    if result.is_falling:
                        now = time.monotonic()
                        last = self._fall_timestamps.get(track_id, 0.0)
                        if now - last >= FALL_COOLDOWN_SECS:
                            self._fall_timestamps[track_id] = now
                            self._new_falls.add(track_id)
            except Exception as e:
                logger.warning("PoseWorker inference error track=%d: %s", track_id, e)
            self._queue.task_done()

    def _load_model(self):
        try:
            import onnxruntime as ort
            path = Path(self._model_path)
            if not path.exists():
                logger.error("MoveNet model not found: %s", path)
                logger.error("Run: python tools/download_models.py --fall-detection")
                return None
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(str(path), providers=providers)
            logger.info("MoveNet loaded (%s)", sess.get_providers()[0])
            return sess
        except Exception as e:
            logger.error("Failed to load MoveNet: %s", e)
            return None

    def _run_inference(self, crop_bgr: np.ndarray, bbox: dict) -> PoseResult:
        # Resize crop to 192×192 and normalize to float32 [0, 1]
        img = cv2.resize(crop_bgr, (192, 192))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2, 0, 1))[np.newaxis]  # 1×3×192×192

        inp_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {inp_name: inp})[0]  # 1×1×17×3

        keypoints = out[0, 0]  # 17×3: (y_norm, x_norm, conf)
        score, avg_conf = self._classify_fall(keypoints, bbox)
        return PoseResult(
            is_falling=(score >= FALL_SCORE_THRESHOLD),
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

        # Gather keypoints with confidence check
        lsh_x, lsh_y, lsh_c = kp(_KP_LEFT_SHOULDER)
        rsh_x, rsh_y, rsh_c = kp(_KP_RIGHT_SHOULDER)
        lhi_x, lhi_y, lhi_c = kp(_KP_LEFT_HIP)
        rhi_x, rhi_y, rhi_c = kp(_KP_RIGHT_HIP)
        lan_x, lan_y, lan_c = kp(_KP_LEFT_ANKLE)
        ran_x, ran_y, ran_c = kp(_KP_RIGHT_ANKLE)

        confs = [lsh_c, rsh_c, lhi_c, rhi_c, lan_c, ran_c]
        avg_conf = float(np.mean(confs))

        # Filter out low-confidence keypoints by zeroing their contribution
        def valid(c): return c >= FALL_MIN_KP_CONF

        score = 0

        # Rule 1: torso angle from vertical
        # Mid-shoulder → mid-hip vector; angle from vertical axis
        if valid(lsh_c) and valid(rsh_c) and valid(lhi_c) and valid(rhi_c):
            sh_x = (lsh_x + rsh_x) / 2
            sh_y = (lsh_y + rsh_y) / 2
            hi_x = (lhi_x + rhi_x) / 2
            hi_y = (lhi_y + rhi_y) / 2
            dx = hi_x - sh_x
            dy = hi_y - sh_y
            if dy != 0 or dx != 0:
                angle_from_vertical = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
                if angle_from_vertical > (90 - FALL_TORSO_ANGLE_MAX):
                    score += 1

        # Rule 2: bounding box aspect ratio (width > height = person lying down)
        w = bbox.get("width", 1)
        h = bbox.get("height", 1)
        if h > 0 and w / h > 1.0:
            score += 1

        # Rule 3: hip Y close to ankle Y (hips near ground level)
        if valid(lhi_c) and valid(rhi_c) and valid(lan_c) and valid(ran_c):
            hi_y = (lhi_y + rhi_y) / 2
            an_y = (lan_y + ran_y) / 2
            if an_y > 0 and hi_y >= an_y * 0.80:
                score += 1

        return score, avg_conf
