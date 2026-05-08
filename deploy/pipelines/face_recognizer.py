"""
face_recognizer.py — NX Computing AI | ArcFace Identity Matching
Async worker: receives face crops from the FaceDetectIR SGIE, extracts 512-dim
ArcFace embeddings via InsightFace buffalo_l, and matches against known_faces.json.

Architecture: same non-blocking queue pattern as NxApiClient / PoseWorker.
  probe → enqueue(face_crop, track_id, frame_num, camera_id)   ← O(1)
  worker thread → InsightFace align+embed → cosine similarity → vote cache
  probe (next frame) → get_result(track_id) → (name, confidence) | None
"""
import json
import logging
import queue
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FACE_SIMILARITY_THRESHOLD: float = 0.50  # cosine sim minimum for a positive match
FACE_VOTES_REQUIRED: int         = 3     # votes before locking identity per track


class FaceRecognizer:
    """
    Runs InsightFace ArcFace embedding in a background thread.
    Matches crops against a per-client known_faces.json database.
    """

    def __init__(self, db_path: str, model_root: str = "/nx_tech/models/insightface",
                 queue_size: int = 64):
        self._db_path = db_path
        self._model_root = model_root
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._locked: Dict[int, Tuple[str, float]] = {}   # track_id → (name, conf)
        self._votes: Dict[int, List[str]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._app = self._load_model()
        self._db = self._load_db()

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="face-recognizer"
        )
        self._thread.start()
        logger.info("FaceRecognizer started — db: %s", self._db_path)

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("FaceRecognizer stopped.")

    def enqueue(self, face_crop_bgr: np.ndarray, track_id: int,
                frame_num: int, camera_id: str = ""):
        try:
            self._queue.put_nowait((face_crop_bgr.copy(), track_id, frame_num, camera_id))
        except queue.Full:
            pass

    def get_result(self, track_id: int) -> Optional[Tuple[str, float]]:
        with self._lock:
            return self._locked.get(track_id)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        if self._app is None:
            logger.error("FaceRecognizer: failed to load InsightFace — worker inactive.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            face_crop, track_id, frame_num, camera_id = item
            try:
                self._process(face_crop, track_id)
            except Exception as e:
                logger.warning("FaceRecognizer error track=%d: %s", track_id, e)
            self._queue.task_done()

    def _load_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                root=self._model_root,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(160, 160))
            logger.info("InsightFace buffalo_l loaded.")
            return app
        except Exception as e:
            logger.error("Failed to load InsightFace: %s", e)
            return None

    def _load_db(self) -> Dict[str, List[np.ndarray]]:
        path = Path(self._db_path)
        if not path.exists():
            logger.warning("Face DB not found: %s — recognition disabled.", path)
            return {}
        try:
            raw = json.loads(path.read_text())
            db = {}
            for name, embeddings in raw.items():
                db[name] = [np.array(e, dtype=np.float32) for e in embeddings]
            logger.info("Face DB loaded: %d person(s) from %s", len(db), path)
            return db
        except Exception as e:
            logger.error("Failed to load face DB: %s", e)
            return {}

    def _process(self, face_crop_bgr: np.ndarray, track_id: int):
        if not self._db:
            return

        faces = self._app.get(face_crop_bgr)
        if not faces:
            return

        emb = faces[0].normed_embedding  # L2-normalized 512-dim float32

        name, sim = self._match(emb)
        if name is None:
            name = "Desconocido"

        with self._lock:
            if track_id in self._locked:
                return
            votes = self._votes.setdefault(track_id, [])
            votes.append(name)
            if len(votes) >= FACE_VOTES_REQUIRED:
                winner = Counter(votes).most_common(1)[0][0]
                self._locked[track_id] = (winner, sim)
                logger.debug("Face locked track=%d → %s (sim=%.3f)", track_id, winner, sim)

    def _match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        best_name: Optional[str] = None
        best_sim: float = -1.0
        for name, db_embeddings in self._db.items():
            for db_emb in db_embeddings:
                sim = float(np.dot(embedding, db_emb))  # both L2-normed → cosine sim
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
        if best_sim >= FACE_SIMILARITY_THRESHOLD:
            return best_name, best_sim
        return None, best_sim
