"""
reid_manager.py — NX Computing AI | Local Cross-Camera Re-ID Manager

Persistent identity database that links local (pad_index, track_id) pairs to stable
global_ids across cameras and time.  Backed by a JSON file so identities survive
container restarts and are retained for up to REID_TTL_S (default 1 hour).

Matching is a vectorised dot-product over L2-normalised 512-dim embeddings — O(N)
and orders of magnitude cheaper than the OSNet inference that produces the vectors.

Returned event types:
  EVENT_NEW_PERSON    — no match found; new global_id created
  EVENT_PERSON_RETURN — same global_id, but last seen > PRESENCE_WINDOW_S ago
  EVENT_CHANNEL_CHANGE — same global_id, last seen within PRESENCE_WINDOW_S (moved cameras)
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuneable constants ──────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.45   # cosine sim (= dot product on L2-normalised vecs)
PRESENCE_WINDOW_S:    float = 300.0  # 5 min — within this, camera switch = channel_change
REID_TTL_S:           float = 3600.0 # 1 hour — global_id expires if unseen for this long
SAVE_INTERVAL_S:      float = 30.0   # persist to disk at most every N seconds

# ── Event type constants ─────────────────────────────────────────────────────────
EVENT_NEW_PERSON     = "new_person"
EVENT_PERSON_RETURN  = "person_return"
EVENT_CHANNEL_CHANGE = "channel_change"


@dataclass
class _Entry:
    global_id:     str
    embedding:     np.ndarray  # 512-dim, L2-normalised float32
    first_seen_ts: float       # wall clock (time.time()) when first created
    last_seen_ts:  float       # updated on every match
    camera_id:     str         # last camera_id where seen
    visit_count:   int = 1     # increments on each entry/return, NOT on channel_change


class ReIdManager:
    """
    Thread-safe local re-ID database.

    Usage:
        mgr = ReIdManager("/opt/nx/reid_db.json")
        global_id, event_type, prev_camera = mgr.match_or_create(embedding, camera_id)
        mgr.flush()   # call on shutdown
    """

    def __init__(self, db_path: str = "/opt/nx/reid_db.json"):
        self._path = Path(db_path)
        self._db: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._last_save_ts: float = 0.0
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────────

    def match_or_create(
        self,
        embedding: np.ndarray,
        camera_id: str,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Match embedding against the DB and return (global_id, event_type, prev_camera_id).

        event_type is one of EVENT_NEW_PERSON, EVENT_PERSON_RETURN, EVENT_CHANNEL_CHANGE.
        prev_camera_id is None for new persons.
        Thread-safe.
        """
        with self._lock:
            self._expire_stale()
            now = time.time()

            best_gid, best_sim = self._find_best_match(embedding)

            if best_gid is None:
                gid = uuid.uuid4().hex[:12]
                self._db[gid] = _Entry(
                    global_id=gid,
                    embedding=embedding.copy(),
                    first_seen_ts=now,
                    last_seen_ts=now,
                    camera_id=camera_id,
                )
                logger.debug("ReID: new person global_id=%s sim_floor=%.3f", gid, best_sim)
                self._maybe_save()
                return gid, EVENT_NEW_PERSON, None

            entry = self._db[best_gid]
            time_absent = now - entry.last_seen_ts
            prev_camera  = entry.camera_id

            # EMA update: weight old embedding 0.7 to keep a stable reference even if
            # the latest crop comes from a bad angle or partial occlusion.
            _alpha = 0.7
            blended = _alpha * entry.embedding + (1.0 - _alpha) * embedding
            _norm = np.linalg.norm(blended)
            entry.embedding    = blended / _norm if _norm > 1e-6 else embedding.copy()
            entry.last_seen_ts = now
            entry.camera_id    = camera_id

            if time_absent <= PRESENCE_WINDOW_S:
                event = EVENT_CHANNEL_CHANGE
            else:
                event = EVENT_PERSON_RETURN
                entry.visit_count += 1

            logger.debug(
                "ReID: %s global_id=%s sim=%.3f absent=%.0fs cam=%s→%s",
                event, best_gid, best_sim, time_absent, prev_camera, camera_id,
            )
            self._maybe_save()
            return best_gid, event, prev_camera

    def flush(self) -> None:
        """Force-save the DB to disk. Call on pipeline shutdown."""
        with self._lock:
            self._save()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._db:
            return None, -1.0
        ids    = list(self._db.keys())
        embeds = np.stack([self._db[gid].embedding for gid in ids])  # (N, 512)
        sims   = embeds @ embedding                                    # (N,)
        idx    = int(np.argmax(sims))
        sim    = float(sims[idx])
        return (ids[idx], sim) if sim >= SIMILARITY_THRESHOLD else (None, sim)

    def _expire_stale(self) -> None:
        now = time.time()
        stale = [gid for gid, e in self._db.items()
                 if now - e.last_seen_ts > REID_TTL_S]
        for gid in stale:
            del self._db[gid]
        if stale:
            logger.debug("ReID: expired %d stale entries", len(stale))

    def _maybe_save(self) -> None:
        if time.time() - self._last_save_ts >= SAVE_INTERVAL_S:
            self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                gid: {
                    "embedding":     e.embedding.tolist(),
                    "first_seen_ts": e.first_seen_ts,
                    "last_seen_ts":  e.last_seen_ts,
                    "camera_id":     e.camera_id,
                    "visit_count":   e.visit_count,
                }
                for gid, e in self._db.items()
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")))
            tmp.rename(self._path)
            self._last_save_ts = time.time()
            logger.debug("ReID: saved %d entries → %s", len(data), self._path)
        except Exception as exc:
            logger.warning("ReID: save failed: %s", exc)

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("ReID: no DB at %s — starting fresh", self._path)
            return
        try:
            now     = time.time()
            raw     = json.loads(self._path.read_text())
            loaded  = 0
            expired = 0
            for gid, v in raw.items():
                if now - v["last_seen_ts"] > REID_TTL_S:
                    expired += 1
                    continue
                self._db[gid] = _Entry(
                    global_id     = gid,
                    embedding     = np.array(v["embedding"], dtype=np.float32),
                    first_seen_ts = v["first_seen_ts"],
                    last_seen_ts  = v["last_seen_ts"],
                    camera_id     = v.get("camera_id", ""),
                    visit_count   = v.get("visit_count", 1),
                )
                loaded += 1
            logger.info("ReID: loaded %d entries (%d expired) from %s",
                        loaded, expired, self._path)
        except Exception as exc:
            logger.warning("ReID: load failed from %s: %s", self._path, exc)
