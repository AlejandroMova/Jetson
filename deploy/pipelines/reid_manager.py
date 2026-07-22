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

match_or_create()'s embedding param accepts either a single (512,) vector or a list of
up to a few candidate vectors from the same track's retry window (see
probes.py's EMBEDDING_BUFFER_MAX) — the match uses the best-of-N against the gallery,
using whichever candidate actually won to update the gallery (never an average, to
avoid blurring distinct angles together). Added 2026-07-16 (calibración ronda 3):
before this, a single bad-angle/blurred frame could doom a track to a spurious
new_person even though the tracker never lost it.

Acceptance has two tiers: SIMILARITY_THRESHOLD (0.85) applies everywhere; a lower
SIMILARITY_THRESHOLD_QUICK_REMATCH (0.75) applies only when the best candidate was
last seen on the SAME camera less than QUICK_REMATCH_WINDOW_S (45s) ago — this targets
the dominant real-world failure mode confirmed via visual calibration against real
crops (tracker loses and re-acquires the same physical person within seconds on the
same camera), without loosening the bar for cross-camera or longer-gap matches, where
a lower global threshold was already proven (2026-07-08 calibration) to merge
different people.

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
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Tuneable constants ──────────────────────────────────────────────────────────
# Cosine similarity threshold for cross-camera matching (dot product on L2-normalised vecs).
# Calibrated 2026-07-08 against real crops from client DEMOONE (osnet_reid.csv +
# /api/admin/crops): of 9 verified match pairs in the 0.708-0.713 range, 6 were
# different people wrongly merged under the same global_id. The inherited 0.68
# (calibrated for OSNet x0.25, never re-verified for x1.0) was far too permissive.
# Raised to 0.85 pending a second, larger calibration round.
SIMILARITY_THRESHOLD:      float = 0.76
# Umbral acotado para re-match rápido — agregado 2026-07-16 (calibración ronda 3) junto
# con el buffer multi-frame en probes.py. Bajar SIMILARITY_THRESHOLD globalmente ya se
# probó y se descartó (calibración 2026-07-08: fusiona personas distintas). Pero el gap
# de tiempo sí separó bien en la data real: <60s en la misma cámara → mayoría misma
# persona; >=60s → casi siempre personas distintas. Estos dos números acotan el umbral
# más bajo exactamente a ese caso — ver match_or_create() para la lógica de aceptación.
# Pendiente de validar con una ronda de calibración post-despliegue (osnet_reid.csv +
# manifest.csv) — son puntos de partida razonados, no un óptimo medido.
SIMILARITY_THRESHOLD_QUICK_REMATCH: float = 0.75
QUICK_REMATCH_WINDOW_S:             float = 45.0
PRESENCE_WINDOW_S:         float = 300.0  # 5 min — within this, camera switch = channel_change
REID_TTL_S:                float = 3600.0 # 1 hour — global_id expires if unseen for this long
SAVE_INTERVAL_S:           float = 30.0   # persist to disk at most every N seconds
GALLERY_MAX_SIZE:          int   = 10     # max embeddings per global_id
# Gallery addition window: embedding is added only if max_sim vs. existing falls in
# (GALLERY_DIVERSITY_THRESHOLD_MIN, GALLERY_DIVERSITY_THRESHOLD_MAX).
# Below MIN → too dissimilar (borderline/noisy match, protects gallery from contamination).
# Above MAX → duplicate angle, skip.
# MAX raised alongside SIMILARITY_THRESHOLD (2026-07-08): _gallery_add() only runs after
# match_or_create() already found a match (sim >= SIMILARITY_THRESHOLD) — if MAX had stayed
# at 0.85, every successful match would score >= MAX and get rejected as a "duplicate
# angle", freezing every gallery at 1 embedding forever. Left a 0.85-0.95 window so
# genuine new angles of the same person still get added; only near-identical frames
# (>=0.95) are skipped as redundant. MIN is now rarely the binding constraint at this
# range but stays as a safety net for the periodic refresh path.
GALLERY_DIVERSITY_THRESHOLD_MAX: float = 0.95
GALLERY_DIVERSITY_THRESHOLD_MIN: float = 0.71

# Always-on CSV analysis log (clients/<cliente>/logs/osnet_reid.csv) — same rotation
# policy as face_recognition.csv (see probes.py's _face_csv_logger).
CSV_LOG_MAX_BYTES:   int = 20 * 1024 * 1024  # 20 MB per file
CSV_LOG_BACKUP_COUNT: int = 5                # ~100 MB max on disk, oldest rotated out

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
        global_id, event_type, prev_camera, expired_ids = mgr.match_or_create(embedding, camera_id)
        mgr.flush()   # call on shutdown
    """

    def __init__(self, db_path: str = "/opt/nx/reid_db.json", gallery_max_size: int = GALLERY_MAX_SIZE,
                 csv_log_dir: Optional[str] = None):
        """Carga la DB desde disco y la deja lista para usar. El lock protege todos los accesos al dict _db.

        gallery_max_size es configurable desde config.yaml (reid_gallery_size) para ajustarlo
        según el número de cámaras: más cámaras = más ángulos distintos = galería más grande.
        _last_save_ts controla la frecuencia de escritura a disco (máximo cada SAVE_INTERVAL_S).
        csv_log_dir, si se pasa, activa un log CSV persistente y siempre-activo (no gateado
        por modo stream, igual que face_recognition.csv) con una fila por cada creación/match/
        refresh de galería — pensado para analizar después similitud y comportamiento de la
        galería, no para debugging en vivo.
        """
        self._path = Path(db_path)
        self._gallery_max_size = gallery_max_size
        self._db: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._last_save_ts: float = 0.0
        self._csv_logger = self._init_csv_logger(csv_log_dir) if csv_log_dir else None
        self._load()  # carga al iniciar, descartando entradas con TTL vencido

    @staticmethod
    def _init_csv_logger(csv_log_dir: str) -> logging.Logger:
        """Crea (idempotente) el logger CSV en <csv_log_dir>/osnet_reid.csv.

        Columnas: camera_id,track_id,global_id,event,similarity,gallery_size,added_angle,
        prev_camera,absent_s,quick_rematch (agregada 2026-07-16 — "yes" si el match pasó
        por el umbral acotado SIMILARITY_THRESHOLD_QUICK_REMATCH en vez del global; vacía
        para new_person/gallery_refresh, donde no aplica). Separado de stdout/docker logs
        (propagate=False) igual que el CSV de face recognition.
        """
        log_dir = Path(csv_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_logger = logging.getLogger("nx.osnet_csv")
        csv_logger.setLevel(logging.INFO)
        csv_logger.propagate = False
        if not csv_logger.handlers:  # idempotente si ReIdManager se reinstancia
            handler = RotatingFileHandler(
                log_dir / "osnet_reid.csv",
                maxBytes=CSV_LOG_MAX_BYTES, backupCount=CSV_LOG_BACKUP_COUNT,
            )
            handler.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))
            csv_logger.addHandler(handler)
        return csv_logger

    # ── Public API ──────────────────────────────────────────────────────────────

    def match_or_create(
        self,
        embedding: Union[np.ndarray, List[np.ndarray]],
        camera_id: str,
        track_id: Optional[int] = None,
        create: bool = True,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """
        Match embedding(s) against the DB and return
        (global_id, event_type, prev_camera_id, expired_ids).

        embedding accepts a single (512,) vector or a list of a few candidate vectors
        from the same track's retry window (see EMBEDDING_BUFFER_MAX in probes.py) —
        matching uses the best-of-N against the gallery; the winning candidate (never
        an average) is what gets used for _gallery_add()/seeding a new identity.

        Acceptance is two-tier: best_sim >= SIMILARITY_THRESHOLD (0.85) always accepts.
        Below that, best_sim >= SIMILARITY_THRESHOLD_QUICK_REMATCH (0.75) also accepts
        IF the best candidate's global_id was last seen on the SAME camera_id less than
        QUICK_REMATCH_WINDOW_S (45s) ago — targets the dominant real failure mode
        (tracker loses and re-acquires the same person within seconds on the same
        camera) without loosening the bar for cross-camera/longer-gap matches (proven
        unsafe by the 2026-07-08 calibration).

        If create=False and no match is found, returns (None, None, None, expired_ids)
        instead of seeding a new identity — used by probes.py to retry an ambiguous
        view on a later frame rather than committing to a new identity from a single
        bad-angle first look (see FULL_BODY_MIN_RATIO in probes.py). When create=True
        (the default), always resolves a global_id — either an existing match or a
        freshly created identity.
        event_type is one of EVENT_NEW_PERSON, EVENT_PERSON_RETURN, EVENT_CHANNEL_CHANGE.
        prev_camera_id is None for new persons.
        expired_ids lists any global_ids just dropped by _expire_stale() this call —
        the caller (probes.py) uses it to purge FaceRecognizer's own vote/lock state
        for the same ids, since nothing else would ever tell it to forget them.
        track_id is log-only (CSV analysis log) — does not affect matching.
        Thread-safe.
        """
        embeddings = [embedding] if isinstance(embedding, np.ndarray) else list(embedding)
        with self._lock:
            expired_ids = self._expire_stale()
            now = time.time()

            best_gid, best_sim, best_embedding = self._find_best_match(embeddings)

            accepted_gid: Optional[str] = None
            quick_rematch = False
            if best_gid is not None:
                entry = self._db[best_gid]
                if best_sim >= SIMILARITY_THRESHOLD:
                    accepted_gid = best_gid
                elif (best_sim >= SIMILARITY_THRESHOLD_QUICK_REMATCH
                      and entry.camera_id == camera_id
                      and (now - entry.last_seen_ts) <= QUICK_REMATCH_WINDOW_S):
                    accepted_gid = best_gid
                    quick_rematch = True

            if accepted_gid is None:
                if not create:
                    return None, None, None, expired_ids
                gid = uuid.uuid4().hex[:12]
                # embeddings[-1]: frame más reciente del buffer — normalmente el de
                # mejor ratio, ya que fue el que disparó should_create. best_embedding
                # solo tiene sentido cuando sí hubo match contra algo existente.
                self._db[gid] = _Entry(
                    global_id=gid,
                    gallery=[embeddings[-1].copy()],
                    first_seen_ts=now,
                    last_seen_ts=now,
                    camera_id=camera_id,
                )
                logger.info("ReID: new_person gid=%s best_sim=%.3f (no match below threshold=%.2f)",
                            gid, best_sim, SIMILARITY_THRESHOLD)
                if self._csv_logger:
                    self._csv_logger.info(
                        "%s,%s,%s,new_person,%.4f,1,yes,,,",
                        camera_id, track_id if track_id is not None else "", gid, best_sim,
                    )
                self._maybe_save()
                return gid, EVENT_NEW_PERSON, None, expired_ids
            # TODO revisar si no poner esto en un else, porque acabamos de poner el tiempo que lo encontramos, no tiene sentido tiempo absent de una persona nueva
            entry = self._db[accepted_gid]
            time_absent = now - entry.last_seen_ts
            prev_camera  = entry.camera_id

            added = _gallery_add(entry.gallery, best_embedding, self._gallery_max_size)
            entry.last_seen_ts = now
            entry.camera_id    = camera_id

            if time_absent <= PRESENCE_WINDOW_S:
                event = EVENT_CHANNEL_CHANGE
            else:
                event = EVENT_PERSON_RETURN
                entry.visit_count += 1

            logger.info(
                "ReID: %s gid=%s sim=%.3f absent=%.0fs cam=%s→%s gallery=%d%s%s",
                event, accepted_gid, best_sim, time_absent, prev_camera, camera_id,
                len(entry.gallery), " +angle" if added else "",
                " [quick-rematch]" if quick_rematch else "",
            )
            if self._csv_logger:
                self._csv_logger.info(
                    "%s,%s,%s,%s,%.4f,%d,%s,%s,%.0f,%s",
                    camera_id, track_id if track_id is not None else "", accepted_gid, event,
                    best_sim, len(entry.gallery), "yes" if added else "no", prev_camera, time_absent,
                    "yes" if quick_rematch else "no",
                )
            self._maybe_save()
            return accepted_gid, event, prev_camera, expired_ids

    def update_embedding(self, global_id: str, embedding: np.ndarray, track_id: Optional[int] = None) -> None:
        """Add embedding to the gallery of a known global_id without matching.
        Called periodically for active tracks to keep the gallery fresh.
        Only adds when the embedding represents a novel angle (diversity check).
        No-op if global_id is not in the DB (expired or never created).
        track_id is log-only (CSV analysis log) — does not affect the update.
        """
        with self._lock:
            entry = self._db.get(global_id)
            if entry is None:
                return
            added = _gallery_add(entry.gallery, embedding, self._gallery_max_size)
            entry.last_seen_ts = time.time()
            if added:
                logger.debug(
                    "ReID: gallery updated global_id=%s size=%d",
                    global_id, len(entry.gallery),
                )
            if self._csv_logger:
                self._csv_logger.info(
                    "%s,%s,%s,gallery_refresh,,%d,%s,,,",
                    entry.camera_id, track_id if track_id is not None else "", global_id,
                    len(entry.gallery), "yes" if added else "no",
                )
            self._maybe_save()

    def flush(self) -> None:
        """Force-save the DB to disk. Call on pipeline shutdown."""
        with self._lock:
            self._save()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _find_best_match(
        self, embeddings: List[np.ndarray]
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """Busca la identidad con mayor similitud coseno en la DB usando una sola llamada BLAS,
        contra el mejor de embeddings (una o varias vistas candidatas del mismo track).

        Concatena todas las galerías en una sola matriz para evitar loops Python con numpy.
        NO aplica ningún threshold — devuelve siempre el mejor candidato encontrado junto con
        cuál de los embeddings de entrada fue el ganador (para que match_or_create() lo use
        en _gallery_add()/como semilla de identidad nueva, sin promediar). La decisión de
        aceptar vive en match_or_create(), que necesita camera_id/last_seen_ts —contexto que
        esta función no tiene— para el umbral acotado de re-match rápido.
        Devuelve (None, -1.0, None) si la DB está vacía.
        Debe llamarse dentro del lock (_lock).
        """
        if not self._db:
            return None, -1.0, None
        ids       = list(self._db.keys())
        galleries = [self._db[gid].gallery for gid in ids]
        sizes     = [len(g) for g in galleries]

        # Una sola multiplicación matricial: todas las galerías × todos los embeddings candidatos.
        all_vecs = np.concatenate([np.stack(g) for g in galleries])  # (sum_K, 512)
        query    = np.stack(embeddings)                               # (M, 512)
        all_sims = all_vecs @ query.T                                 # (sum_K, M)

        # Reducir: max similarity por entrada (sobre K y M a la vez) usando offsets acumulados.
        best_sim, best_idx, best_q, offset = -1.0, 0, 0, 0
        for i, size in enumerate(sizes):
            block = all_sims[offset:offset + size]          # (size, M)
            flat  = int(np.argmax(block))
            s     = float(block.flat[flat])
            if s > best_sim:
                best_sim, best_idx, best_q = s, i, flat % block.shape[1]
            offset += size

        return ids[best_idx], best_sim, embeddings[best_q]

    def _expire_stale(self) -> List[str]:
        """Elimina entradas que no han sido vistas en REID_TTL_S segundos. Llamar dentro del lock.

        Retorna la lista de global_ids recién expirados, para que match_or_create()
        se la pase al llamador (probes.py) — es la única forma de avisarle a
        FaceRecognizer que también olvide su estado de votos/candado para esos ids,
        ya que ReIdManager y FaceRecognizer son diccionarios independientes.
        """
        now = time.time()
        stale = [gid for gid, e in self._db.items()
                 if now - e.last_seen_ts > REID_TTL_S]
        for gid in stale:
            del self._db[gid]
        if stale:
            logger.debug("ReID: expired %d stale entries", len(stale))
        return stale

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
