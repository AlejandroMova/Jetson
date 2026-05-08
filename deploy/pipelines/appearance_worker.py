"""
appearance_worker.py — NX Computing AI | Cross-Camera Re-ID Worker

Runs OSNet-x0.25 ONNX in a background thread to generate 512-dim L2-normalized
appearance embeddings for each detected person. Used for cross-camera re-ID by the backend.

Architecture: same non-blocking queue pattern as PoseWorker / NxApiClient.
  probe → enqueue(crop_bgr, track_id, frame_num)   ← O(1)
  worker thread → resize + normalize + OSNet ONNX → 512-dim L2-norm vector
  probe (next frame) → get_result(track_id) → np.ndarray | None

Input: BGR crop of any size (resized internally to 128×256)
Output: 512-dim float32 vector, L2-normalized (cosine sim = dot product)

Model: OSNet-x0.25 from torchreid (KaiyangZhou), Apache 2.0 license
       Input: NCHW float32, RGB, ImageNet-normalized, 3×256×128
       Output: (1, 512) float32
Download: python3 tools/download_models.py --reid
"""
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_HEIGHT = 256
INPUT_WIDTH  = 128


class AppearanceWorker:
    """
    Non-blocking OSNet ONNX worker for appearance embeddings.
    Once a result is ready for a track_id it is cached until get_result() is called.
    """

    def __init__(self, model_path: str, queue_size: int = 64):
        self._model_path = model_path
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._results: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session = self._load_model()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="appearance-worker"
        )
        self._thread.start()
        logger.info("AppearanceWorker started — model: %s", self._model_path)

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("AppearanceWorker stopped.")

    def enqueue(self, crop_bgr: np.ndarray, track_id: int, frame_num: int) -> None:
        try:
            self._queue.put_nowait((crop_bgr.copy(), track_id, frame_num))
        except queue.Full:
            pass

    def get_result(self, track_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._results.get(track_id)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        if self._session is None:
            logger.error("AppearanceWorker: ONNX model failed to load — worker inactive.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            crop_bgr, track_id, frame_num = item
            try:
                vec = self._infer(crop_bgr)
                if vec is not None:
                    with self._lock:
                        self._results[track_id] = vec
                    logger.debug("Appearance vector computed track=%d frame=%d", track_id, frame_num)
            except Exception as e:
                logger.warning("AppearanceWorker error track=%d: %s", track_id, e)
            self._queue.task_done()

    def _load_model(self):
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            sess = ort.InferenceSession(self._model_path, providers=providers)
            logger.info("OSNet ONNX loaded (providers: %s)", sess.get_providers())
            return sess
        except Exception as e:
            logger.error("Failed to load OSNet ONNX from %s: %s", self._model_path, e)
            return None

    def _infer(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        # Resize to 128×256 (W×H), BGR→RGB, float32, /255
        img = cv2.resize(crop_bgr, (INPUT_WIDTH, INPUT_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalize
        img = (img - _MEAN) / _STD

        # HWC → NCHW
        inp = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 256, 128)

        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: inp})[0]  # (1, 512)
        vec = output[0].astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec
