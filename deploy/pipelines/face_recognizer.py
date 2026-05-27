"""
face_recognizer.py — NX Computing AI | ArcFace Identity Matching
Async worker: receives face crops from PeopleNet class_id=2 (face) detections, extracts 512-dim
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
        """Configura el worker. InsightFace se carga en start() para evitar conflictos CUDA con TRT.

        _locked almacena el resultado final por track_id una vez que acumula FACE_VOTES_REQUIRED votos.
        _votes acumula las predicciones frame a frame antes de bloquear la identidad.
        La DB se carga en __init__ porque no usa GPU — es solo lectura de JSON + conversión a ndarray.
        """
        self._db_path = db_path
        self._model_root = model_root
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._locked: Dict[int, Tuple[str, float]] = {}   # track_id → (name, conf) bloqueado
        self._votes: Dict[int, List[str]] = {}             # track_id → lista de predicciones acumuladas
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._app = None   # se carga en start() después de que TRT inicialice CUDA
        self._db = self._load_db()

    def start(self):
        """Carga InsightFace buffalo_l y arranca el hilo worker. Llamar después de pipeline.set_state(PLAYING)."""
        self._app = self._load_model()  # debe correr después de set_state(PLAYING) — misma razón que AppearanceWorker
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="face-recognizer"
        )
        self._thread.start()
        logger.info("FaceRecognizer started — db: %s", self._db_path)

    def stop(self):
        """Señaliza al worker que pare y espera a que el hilo termine (máximo 5 s)."""
        self._running = False
        self._queue.put(None)  # sentinel para desbloquear el get() en _worker_loop
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("FaceRecognizer stopped.")

    def enqueue(self, face_crop_bgr: np.ndarray, track_id: int,
                frame_num: int, camera_id: str = ""):
        """Encola un crop de rostro para reconocimiento. No bloqueante — descarta si la cola está llena."""
        try:
            self._queue.put_nowait((face_crop_bgr.copy(), track_id, frame_num, camera_id))
        except queue.Full:
            pass  # descarte silencioso — el probe reintentará en el próximo frame

    def get_result(self, track_id: int) -> Optional[Tuple[str, float]]:
        with self._lock:
            return self._locked.get(track_id)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        """Loop principal del hilo worker: consume la cola y acumula votos de identidad.

        Usa get(timeout=1.0) para no bloquearse indefinidamente y poder detectar _running=False.
        Cada item procesado llama a _process(), que internamente aplica el sistema de votos.
        """
        if self._app is None:
            logger.error("FaceRecognizer: failed to load InsightFace — worker inactive.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break  # sentinel enviado por stop()
            face_crop, track_id, frame_num, camera_id = item
            try:
                self._process(face_crop, track_id)
            except Exception as e:
                logger.warning("FaceRecognizer error track=%d: %s", track_id, e)
            self._queue.task_done()

    def _load_model(self):
        """Carga InsightFace buffalo_l en CPU. Retorna la app InsightFace o None si falla.

        ctx_id=-1 forza CPU; det_size=(160, 160) es suficiente para crops de cara del SGIE.
        CPUExecutionProvider evita conflictos con TensorRT que ya usa la GPU para los SGIEs.
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                root=self._model_root,
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(160, 160))  # ctx_id=-1 = CPU
            logger.info("InsightFace buffalo_l loaded.")
            return app
        except Exception as e:
            logger.error("Failed to load InsightFace: %s", e)
            return None

    def _load_db(self) -> Dict[str, List[np.ndarray]]:
        """Carga known_faces.json y convierte los embeddings de listas a np.ndarray float32.

        Retorna dict vacío si el archivo no existe o tiene errores de formato.
        En ese caso, el reconocimiento queda deshabilitado silenciosamente hasta que
        el operador registre caras con register_face.py y reinicie el pipeline.
        """
        path = Path(self._db_path)
        if not path.exists():
            logger.warning("Face DB not found: %s — recognition disabled.", path)
            return {}
        try:
            raw = json.loads(path.read_text())
            db = {}
            for name, embeddings in raw.items():
                # Convertir cada embedding de lista JSON a ndarray float32 para dot product
                db[name] = [np.array(e, dtype=np.float32) for e in embeddings]
            logger.info("Face DB loaded: %d person(s) from %s", len(db), path)
            return db
        except Exception as e:
            logger.error("Failed to load face DB: %s", e)
            return {}

    def _process(self, face_crop_bgr: np.ndarray, track_id: int):
        """Infiere identidad de un crop de cara y acumula votos hasta bloquear el resultado.

        Sistema de votación: requiere FACE_VOTES_REQUIRED coincidencias antes de bloquear
        la identidad por track_id, reduciendo falsos positivos en frames difíciles.
        Una vez bloqueada la identidad (_locked), no se vuelve a procesar ese track.
        """
        if not self._db:
            return  # DB vacía — reconocimiento deshabilitado para este cliente

        # Detectar y alinear la cara en el crop usando InsightFace
        faces = self._app.get(face_crop_bgr)
        if not faces:
            return  # sin cara detectada en este crop (blur, oclusión, etc.)

        # Usar el embedding de la primera cara (mayor bbox = más prominente)
        emb = faces[0].normed_embedding  # vector 512-dim ya L2-normalizado por InsightFace

        # Comparar con la DB y obtener el mejor match
        name, sim = self._match(emb)
        if name is None:
            name = "Desconocido"  # por debajo del threshold → cara desconocida

        with self._lock:
            if track_id in self._locked:
                return  # identidad ya bloqueada — no procesar más frames de este track

            # Agregar voto y verificar si alcanzamos el umbral de confianza
            votes = self._votes.setdefault(track_id, [])
            votes.append(name)
            if len(votes) >= FACE_VOTES_REQUIRED:
                # Elegir la identidad más votada (mayoritaria) como resultado final
                winner = Counter(votes).most_common(1)[0][0]
                self._locked[track_id] = (winner, sim)
                logger.debug("Face locked track=%d → %s (sim=%.3f)", track_id, winner, sim)

    def _match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Busca la persona con mayor similitud coseno en la DB.

        Compara el embedding contra todos los embeddings de cada persona.
        Retorna (nombre, similitud) si supera FACE_SIMILARITY_THRESHOLD, o (None, sim) si no.
        Ambos vectores están L2-normalizados, por lo que dot product = similitud coseno.
        """
        best_name: Optional[str] = None
        best_sim: float = -1.0

        # Iterar sobre cada persona y todos sus embeddings (puede haber varios ángulos)
        for name, db_embeddings in self._db.items():
            for db_emb in db_embeddings:
                sim = float(np.dot(embedding, db_emb))  # cosine sim: ambos L2-normalizados
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        # Retornar solo si supera el threshold; de lo contrario indicar desconocido
        if best_sim >= FACE_SIMILARITY_THRESHOLD:
            return best_name, best_sim
        return None, best_sim
