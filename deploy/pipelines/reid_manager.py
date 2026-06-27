"""
reid_manager.py — NX Computing AI | Local Cross-Camera Re-ID Manager

Persistent identity database that links local (pad_index, track_id) pairs to stable
global_ids across cameras and time.  Backed by a JSON file so identities survive
container restarts and are retained for up to REID_TTL_S (default 1 hour).

Each global_id stores a gallery of up to GALLERY_MAX_SIZE L2-normalised 512-dim
embeddings covering distinct poses/angles.  Matching uses max-similarity over the
gallery — if any gallery angle matches, the identity is recognised even if the
current angle differs from the others.  New embeddings are added to the gallery
only when max similarity vs. existing members falls in (GALLERY_DIVERSITY_THRESHOLD_MIN,
GALLERY_DIVERSITY_THRESHOLD_MAX) — novel enough to add value, similar enough to
belong to the same identity.

Matching is vectorised: gallery_matrix @ query → shape (N×K,), then max per entry.
O(N×K) but K≤5, negligible vs. OSNet inference.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuneable constants ──────────────────────────────────────────────────────────
# Cosine similarity threshold for cross-camera matching (dot product on L2-normalised vecs).
# Tuning guide for OSNet-x1.0 on DVR sub-stream:
#   (values below were calibrated for x0.25 — x1.0 may need re-calibration)
#   0.65 → very strict, misses real matches
#   0.60 → too strict for x1.0 on sub-stream (misses same-person returns)
#   0.50 → current: good balance for x1.0; reduce to 0.45 if still missing matches
#   0.40 → too low: false positives (different people matching same global_id)
SIMILARITY_THRESHOLD:      float = 0.68
PRESENCE_WINDOW_S:         float = 300.0  # 5 min — within this, camera switch = channel_change
REID_TTL_S:                float = 3600.0 # 1 hour — global_id expires if unseen for this long
SAVE_INTERVAL_S:           float = 30.0   # persist to disk at most every N seconds
GALLERY_MAX_SIZE:          int   = 10     # max embeddings per global_id
# Gallery addition window: embedding is added only if max_sim vs. existing falls in
# (GALLERY_DIVERSITY_THRESHOLD_MIN, GALLERY_DIVERSITY_THRESHOLD_MAX).
# Below MIN → too dissimilar (borderline/noisy match, protects gallery from contamination).
# Above MAX → duplicate angle, skip.
GALLERY_DIVERSITY_THRESHOLD_MAX: float = 0.85
GALLERY_DIVERSITY_THRESHOLD_MIN: float = 0.71

# ── Event type constants ─────────────────────────────────────────────────────────
EVENT_NEW_PERSON     = "new_person"
EVENT_PERSON_RETURN  = "person_return"
EVENT_CHANNEL_CHANGE = "channel_change"


@dataclass
class _Entry:
    """Representa una identidad conocida en la base de datos de re-ID.

    gallery: lista de hasta GALLERY_MAX_SIZE embeddings L2-normalizados 512-dim.
    visit_count solo incrementa en entry/return — no en channel_change (cambio de cámara
    dentro de la ventana de presencia no cuenta como una nueva visita del cliente).
    """
    global_id:     str
    gallery:       List[np.ndarray]  # hasta GALLERY_MAX_SIZE vectores L2-normalizados
    first_seen_ts: float             # wall clock (time.time()) cuando se creó por primera vez
    last_seen_ts:  float             # se actualiza en cada match (entry, return, channel_change)
    camera_id:     str               # última cámara donde fue visto
    visit_count:   int = 1           # incrementa solo en entry/return, no en channel_change


def _gallery_add(gallery: List[np.ndarray], embedding: np.ndarray, max_size: int = GALLERY_MAX_SIZE) -> bool:
    """
    Add embedding to gallery if it represents a novel angle.
    Returns True if the embedding was added.

    Addition rules:
    - Gallery has fewer than max_size entries → always add (first embedding always stored).
    - Gallery is full → add only if max similarity vs. all existing falls in
      (GALLERY_DIVERSITY_THRESHOLD_MIN, GALLERY_DIVERSITY_THRESHOLD_MAX):
        - >= MAX (0.85): duplicate angle — skip.
        - <= MIN (0.71): too dissimilar — likely noisy/borderline match, skip to
          protect gallery integrity against false-positive contamination.
    """
    if gallery:
        gallery_mat = np.stack(gallery)          # (K, 512)
        sims = gallery_mat @ embedding           # (K,)
        max_sim = float(np.max(sims))
        if max_sim >= GALLERY_DIVERSITY_THRESHOLD_MAX or max_sim < GALLERY_DIVERSITY_THRESHOLD_MIN:
            return False  # outside valid diversity window — skip
    if len(gallery) < max_size:
        gallery.append(embedding.copy())
        return True
    # Gallery full and the new vector is sufficiently diverse — replace the member
    # that is most similar to the others (least informative) to keep max diversity.
    gallery_mat = np.stack(gallery)              # (K, 512)
    # Each member's max similarity to the rest
    self_sims = np.array([
        float(np.max(np.delete(gallery_mat, i, axis=0) @ gallery[i]))
        for i in range(len(gallery))
    ])
    replace_idx = int(np.argmax(self_sims))
    gallery[replace_idx] = embedding.copy()
    return True


def _gallery_best_sim(gallery: List[np.ndarray], embedding: np.ndarray) -> float:
    """Return the maximum cosine similarity between embedding and any gallery member."""
    if not gallery:
        return -1.0
    gallery_mat = np.stack(gallery)  # (K, 512)
    return float(np.max(gallery_mat @ embedding))


class ReIdManager:
    """
    Thread-safe local re-ID database.

    Usage:
        mgr = ReIdManager("/opt/nx/reid_db.json")
        global_id, event_type, prev_camera = mgr.match_or_create(embedding, camera_id)
        mgr.flush()   # call on shutdown
    """

    def __init__(self, db_path: str = "/opt/nx/reid_db.json", gallery_max_size: int = GALLERY_MAX_SIZE):
        """Carga la DB desde disco y la deja lista para usar. El lock protege todos los accesos al dict _db.

        gallery_max_size es configurable desde config.yaml (reid_gallery_size) para ajustarlo
        según el número de cámaras: más cámaras = más ángulos distintos = galería más grande.
        _last_save_ts controla la frecuencia de escritura a disco (máximo cada SAVE_INTERVAL_S).
        """
        self._path = Path(db_path)
        self._gallery_max_size = gallery_max_size
        self._db: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._last_save_ts: float = 0.0
        self._load()  # carga al iniciar, descartando entradas con TTL vencido

    # ── Public API ──────────────────────────────────────────────────────────────

    def match_or_create(
        self,
        embedding: np.ndarray,
        camera_id: str,
        threshold: Optional[float] = None,
        create: bool = True,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Match embedding against the DB and return (global_id, event_type, prev_camera_id).

        threshold overrides SIMILARITY_THRESHOLD for this call — probes.py computes the
        right value based on the detection's aspect ratio before calling this method.
        If create=False and no match is found, returns (None, None, None) instead of
        seeding a new identity; used for partial-body detections that should not create
        new identities but can still match existing ones.
        event_type is one of EVENT_NEW_PERSON, EVENT_PERSON_RETURN, EVENT_CHANNEL_CHANGE.
        prev_camera_id is None for new persons.
        Thread-safe.
        """
        with self._lock:
            self._expire_stale()
            now = time.time()

            best_gid, best_sim = self._find_best_match(embedding, threshold=threshold)

            if best_gid is None:
                if not create:
                    return None, None, None
                gid = uuid.uuid4().hex[:12]
                self._db[gid] = _Entry(
                    global_id=gid,
                    gallery=[embedding.copy()],
                    first_seen_ts=now,
                    last_seen_ts=now,
                    camera_id=camera_id,
                )
                logger.info("ReID: new_person gid=%s best_sim=%.3f (no match below threshold=%.2f)",
                            gid, best_sim, threshold if threshold is not None else SIMILARITY_THRESHOLD)
                self._maybe_save()
                return gid, EVENT_NEW_PERSON, None
            # TODO revisar si no poner esto en un else, porque acabamos de poner el tiempo que lo encontramos, no tiene sentido tiempo absent de una persona nueva
            entry = self._db[best_gid]
            time_absent = now - entry.last_seen_ts
            prev_camera  = entry.camera_id

            added = _gallery_add(entry.gallery, embedding, self._gallery_max_size)
            entry.last_seen_ts = now
            entry.camera_id    = camera_id

            if time_absent <= PRESENCE_WINDOW_S:
                event = EVENT_CHANNEL_CHANGE
            else:
                event = EVENT_PERSON_RETURN
                entry.visit_count += 1

            logger.info(
                "ReID: %s gid=%s sim=%.3f absent=%.0fs cam=%s→%s gallery=%d%s",
                event, best_gid, best_sim, time_absent, prev_camera, camera_id,
                len(entry.gallery), " +angle" if added else "",
            )
            self._maybe_save()
            return best_gid, event, prev_camera

    def update_embedding(self, global_id: str, embedding: np.ndarray) -> None:
        """Add embedding to the gallery of a known global_id without matching.
        Called periodically for active tracks to keep the gallery fresh.
        Only adds when the embedding represents a novel angle (diversity check).
        No-op if global_id is not in the DB (expired or never created).
        """
        with self._lock:
            entry = self._db.get(global_id)
            if entry is None:
                return
            added = _gallery_add(entry.gallery, embedding, self._gallery_max_size)
            entry.last_seen_ts = time.time()
            self._maybe_save()
            if added:
                logger.debug(
                    "ReID: gallery updated global_id=%s size=%d",
                    global_id, len(entry.gallery),
                )

    def flush(self) -> None:
        """Force-save the DB to disk. Call on pipeline shutdown."""
        with self._lock:
            self._save()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _find_best_match(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """Busca la identidad con mayor similitud coseno en la DB usando una sola llamada BLAS.

        Concatena todas las galerías en una sola matriz para evitar loops Python con numpy.
        Devuelve (global_id, sim) si sim >= effective_threshold, o (None, best_sim) si no hay match.
        threshold overrides SIMILARITY_THRESHOLD when provided (e.g. for partial-body views).
        Debe llamarse dentro del lock (_lock).
        """
        if not self._db:
            return None, -1.0
        effective_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
        ids       = list(self._db.keys())
        galleries = [self._db[gid].gallery for gid in ids]
        sizes     = [len(g) for g in galleries]

        # Una sola multiplicación matricial sobre todos los vectores de galería concatenados.
        all_vecs = np.concatenate([np.stack(g) for g in galleries])  # (sum_K, 512)
        all_sims = all_vecs @ embedding                               # (sum_K,)

        # Reducir: max similarity por entrada usando offsets acumulados.
        best_sim = -1.0
        best_idx = 0
        offset   = 0
        for i, size in enumerate(sizes):
            s = float(np.max(all_sims[offset:offset + size]))
            if s > best_sim:
                best_sim = s
                best_idx = i
            offset += size

        return (ids[best_idx], best_sim) if best_sim >= effective_threshold else (None, best_sim)

    def _expire_stale(self) -> None:
        """Elimina entradas que no han sido vistas en REID_TTL_S segundos. Llamar dentro del lock."""
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
        """Persiste la DB en disco de forma atómica (escribe en .tmp y luego rename).

        El rename atómico garantiza que el archivo nunca quede en estado corrupto
        si el proceso se mata a mitad de escritura — el tmp se descarta automáticamente.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                gid: {
                    "gallery":       [v.tolist() for v in e.gallery],  # np.ndarray → lista para JSON
                    "first_seen_ts": e.first_seen_ts,
                    "last_seen_ts":  e.last_seen_ts,
                    "camera_id":     e.camera_id,
                    "visit_count":   e.visit_count,
                }
                for gid, e in self._db.items()
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")))  # sin espacios = menos bytes
            tmp.rename(self._path)  # rename atómico: archivo siempre consistente en disco
            self._last_save_ts = time.time()
            logger.debug("ReID: saved %d entries → %s", len(data), self._path)
        except Exception as exc:
            logger.warning("ReID: save failed: %s", exc)

    def _load(self) -> None:
        """Carga la DB desde disco al iniciar, descartando entradas con TTL vencido.

        Migración automática de esquema: el formato antiguo guardaba un solo "embedding" (lista).
        El formato nuevo usa "gallery" (lista de listas). Ambos se leen correctamente.
        Entradas sin "gallery" ni "embedding" se descartan silenciosamente (datos corruptos).
        """
        if not self._path.exists():
            logger.info("ReID: no DB at %s — starting fresh", self._path)
            return
        try:
            now     = time.time()
            raw     = json.loads(self._path.read_text())
            loaded  = 0
            expired = 0
            skipped = 0
            for gid, v in raw.items():
                if now - v["last_seen_ts"] > REID_TTL_S:
                    expired += 1
                    continue  # entrada expirada — no cargar

                # Migración de esquema: "embedding" (antiguo, un solo vec) → "gallery" (nuevo, lista)
                if "gallery" in v:
                    gallery = [np.array(e, dtype=np.float32) for e in v["gallery"]]
                elif "embedding" in v:
                    gallery = [np.array(v["embedding"], dtype=np.float32)]  # envolver en lista
                else:
                    skipped += 1
                    continue  # formato desconocido — descartar

                if not gallery:
                    skipped += 1
                    continue

                self._db[gid] = _Entry(
                    global_id     = gid,
                    gallery       = gallery,
                    first_seen_ts = v["first_seen_ts"],
                    last_seen_ts  = v["last_seen_ts"],
                    camera_id     = v.get("camera_id", ""),
                    visit_count   = v.get("visit_count", 1),
                )
                loaded += 1
            logger.info(
                "ReID: loaded %d entries (%d expired, %d skipped) from %s",
                loaded, expired, skipped, self._path,
            )
        except Exception as exc:
            logger.warning("ReID: load failed from %s: %s", self._path, exc)
