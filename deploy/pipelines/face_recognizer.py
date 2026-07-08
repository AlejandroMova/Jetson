"""
face_recognizer.py — NX Computing AI | ArcFace Identity Matching
Async worker: receives face crops from PeopleNet class_id=2 (face) detections, extracts 512-dim
ArcFace embeddings via InsightFace buffalo_l, and matches against known_faces.json.

Architecture: same non-blocking queue pattern as NxApiClient / AppearanceWorker.
  probe → enqueue(face_crop, global_id, frame_num, camera_id)   ← O(1)
  worker thread → InsightFace align+embed → cosine similarity → vote cache
  probe (next frame) → get_result(global_id) → (uuid_str, confidence, yaw) | None

Identity is keyed by the cross-camera global_id from ReIdManager, not by the
local per-camera track_id — a track_id resets every time the person changes
camera, which used to mean re-running the full 3-vote identification from
scratch in every camera. Keying by global_id lets an already-locked identity
ride along across cameras via ReID's own appearance-gallery continuity
(probes.py only starts feeding crops once a track's global_id is resolved —
see _FaceRecognitionHandler.process_face). Votes never stop accumulating
even after a lock: uniforms can make different employees look alike to
OSNet's appearance embedding, so if a later face crop strongly disagrees with
the current lock, the tag gets corrected (see _process) instead of trusting
a one-time match forever.

JSON format (new, UUID-keyed):
  {
    "<employee-uuid>": {
      "name": "Juan Perez",
      "embeddings": [[...512 floats...], [...512 floats...]]
    }
  }

Legacy format (name-keyed, backwards compatible):
  { "Juan Perez": [[...512 floats...], [...512 floats...]] }

On first sync_from_backend() the file is rewritten in the new format.
The backend assigns the UUID (from employees.id) which the Jetson must respect
and echo back in all face recognition events so the backend can join on UUID FK.
"""
import json
import logging
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FACE_SIMILARITY_THRESHOLD: float = 0.50  # cosine sim minimum for a positive match
FACE_VOTES_REQUIRED: int         = 3     # votes in the sliding window before (re)locking identity

# Muestra de mal ángulo: yaw absoluto (grados) por encima del cual una cara no
# cuenta como voto — la persona está de perfil/casi de espaldas, el embedding
# sale como ruido sin importar qué tan bueno sea el modelo. face.pose viene
# gratis del modelo landmark_3d_68 que buffalo_l ya carga, sin costo extra.
FACE_MAX_YAW_DEGREES: float = 35.0

# ponytail: diagnóstico temporal — guarda hasta N crops crudos (antes de CLAHE)
# a disco para inspección visual manual. Quitar cuando ya no se necesite ver
# a ojo qué le está llegando al reconocedor.
FACE_CROP_SAMPLE_MAX: int = 30


class FaceRecognizer:
    """
    Runs InsightFace ArcFace embedding in a background thread.
    Matches crops against a per-client known_faces.json database.

    Identity keys are employee UUIDs (strings) assigned by the backend.
    Use get_display_name(uuid) to obtain the human-readable name for OSD.
    """

    def __init__(self, db_path: str, model_root: str = "/nx_tech/models/insightface",
                 api_base_url: str = "", api_key: str = "", queue_size: int = 64):
        """Configura el worker. InsightFace se carga en start() para evitar conflictos CUDA con TRT.

        _locked almacena el resultado actual por global_id (uuid, conf, yaw). _votes es una ventana
        deslizante (deque, maxlen=FACE_VOTES_REQUIRED) de las últimas predicciones por global_id —
        se sigue alimentando incluso después de bloquear, así que un cambio sostenido de mayoría
        corrige el candado en vez de quedar congelado en el primer match.
        La DB se carga en __init__ porque no usa GPU — es solo lectura de JSON + conversión a ndarray.

        Args:
            db_path:      Ruta a known_faces.json del cliente.
            model_root:   Directorio donde InsightFace descarga/busca buffalo_l.
            api_base_url: URL HTTP del backend (para sync_from_backend).
            api_key:      API key del Jetson (para sync_from_backend).
            queue_size:   Máximo de crops pendientes en la cola.
        """
        self._db_path = db_path
        self._model_root = model_root
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._locked: Dict[str, Tuple[str, float, float]] = {}  # global_id → (uuid/name, conf, yaw)
        self._votes: Dict[str, Deque[str]] = {}            # global_id → ventana de predicciones
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._app = None   # InsightFace — se carga en start() después de que TRT inicialice CUDA
        self._db: Dict[str, List[np.ndarray]] = {}
        self._uuid_to_name: Dict[str, str] = {}  # uuid → nombre legible para OSD
        # ponytail: diagnóstico temporal (ver FACE_CROP_SAMPLE_MAX) — carpeta y contador
        # para guardar una muestra acotada de crops crudos.
        self._crop_sample_dir = Path(self._db_path).parent / "logs" / "face_crops_sample"
        self._crop_sample_count = 0
        self._load_db()

    def start(self):
        """Carga InsightFace buffalo_l, realiza sync inicial con backend y arranca el hilo worker.

        Llamar después de pipeline.set_state(PLAYING) para evitar conflictos GPU con TRT.
        Si api_base_url está configurado, realiza un sync_from_backend() antes de procesar
        para asegurar que el Jetson arranca con el roster más reciente.
        """
        self._app = self._load_model()
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="face-recognizer"
        )
        self._thread.start()
        logger.info("FaceRecognizer started — db: %s", self._db_path)

        # Sync inicial: el Jetson puede haber estado offline y el roster cambió.
        if self._api_base_url and self._api_key:
            threading.Thread(
                target=self.sync_from_backend, daemon=True, name="face-sync-startup"
            ).start()

    def stop(self):
        """Señaliza al worker que pare y espera a que el hilo termine (máximo 5 s)."""
        self._running = False
        self._queue.put(None)  # sentinel para desbloquear el get() en _worker_loop
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("FaceRecognizer stopped.")

    def enqueue(self, face_crop_bgr: np.ndarray, identity_key: str,
                frame_num: int, camera_id: str = ""):
        """Encola un crop de rostro para reconocimiento, indexado por global_id (identity_key).
        No bloqueante — descarta si la cola está llena."""
        try:
            self._queue.put_nowait((face_crop_bgr.copy(), identity_key, frame_num, camera_id))
        except queue.Full:
            pass  # descarte silencioso — el probe reintentará en el próximo frame

    def get_result(self, identity_key: str) -> Optional[Tuple[str, float, float]]:
        """Retorna (uuid_or_name, confidence, yaw) si la identidad ya tiene un candado
        vigente para este global_id, o None si aún no hay suficientes votos."""
        with self._lock:
            return self._locked.get(identity_key)

    def forget(self, global_id: str) -> None:
        """Limpia el estado de votos/candado de un global_id que ReIdManager ya
        expiró (ver reid_manager.py::_expire_stale). Sin esto, _locked/_votes
        crecerían para siempre — nada más los limpia por global_id."""
        with self._lock:
            self._locked.pop(global_id, None)
            self._votes.pop(global_id, None)

    def get_display_name(self, uuid_str: str) -> str:
        """Retorna el nombre legible del empleado para mostrar en OSD y stream logs.

        Si uuid_str es un UUID conocido, devuelve el nombre registrado.
        En caso de formato legacy (nombre como clave), devuelve el mismo string.
        Fallback: devuelve el uuid_str tal cual.
        """
        return self._uuid_to_name.get(uuid_str, uuid_str)

    def reload(self, raw_db: dict) -> None:
        """Reemplaza la DB en memoria con un nuevo dict cargado del backend.

        Resetea _locked y _votes para evitar que votos stale de empleados revocados
        contaminen nuevas identificaciones.

        Args:
            raw_db: Dict en el formato nuevo o legacy — se parsea con _parse_raw_db().
        """
        new_db, new_uuid_to_name = _parse_raw_db(raw_db)
        with self._lock:
            self._db = new_db
            self._uuid_to_name = new_uuid_to_name
            self._locked.clear()
            self._votes.clear()
        logger.info("FaceRecognizer recargada: %d persona(s).", len(new_db))

    def sync_from_backend(self, action: str = "sync", employee_id: str = "") -> None:
        """Sincroniza known_faces.json desde GET /api/employees/embeddings.

        Bloqueante — debe llamarse desde un hilo de fondo, no desde el probe.
        En caso de error HTTP, logea y retorna sin modificar la DB en memoria.

        Args:
            action:      "sync" (activación) o "revoke" — ambos provocan un pull completo.
            employee_id: UUID del empleado afectado (solo para logging).
        """
        if not self._api_base_url or not self._api_key:
            logger.debug("sync_from_backend: API_BASE_URL o API_KEY no configurados — omitiendo.")
            return

        url = f"{self._api_base_url}/api/employees/embeddings"
        try:
            import requests
            resp = requests.get(
                url,
                headers={"X-API-Key": self._api_key},
                timeout=30,
            )
            resp.raise_for_status()
            items = resp.json()  # list of {employee_id, name, embedding, model_version}
        except Exception as exc:
            logger.warning("sync_from_backend: GET %s falló: %s", url, exc)
            return

        # Construir el nuevo formato JSON y guardarlo en disco.
        new_raw: dict = {}
        for item in items:
            uid = str(item.get("employee_id", ""))
            name = item.get("name", uid)
            embs = item.get("embeddings", [])
            if uid and embs:
                new_raw[uid] = {"name": name, "embeddings": embs}

        try:
            Path(self._db_path).write_text(json.dumps(new_raw, indent=2))
        except Exception as exc:
            logger.warning("sync_from_backend: no se pudo escribir %s: %s", self._db_path, exc)

        self.reload(new_raw)
        logger.info(
            "sync_from_backend: %d empleado(s) sincronizados desde backend [action=%s%s].",
            len(new_raw), action,
            f" employee_id={employee_id}" if employee_id else "",
        )

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self):
        """Main worker loop: consumes the queue and accumulates identity votes.
        Uses get(timeout=1.0) to remain responsive to _running=False without blocking indefinitely.
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
                break  # sentinel sent by stop()
            # NOTE: _camera_id reserved for future per-camera routing — not yet wired to _process.
            face_crop, identity_key, frame_num, _camera_id = item
            try:
                self._process(face_crop, identity_key)
            except Exception as e:
                logger.warning("FaceRecognizer error global_id=%s: %s", identity_key, e)
            self._queue.task_done()

    def _load_model(self):
        """Carga InsightFace buffalo_l en CPU. Retorna la app InsightFace o None si falla.

        ctx_id=-1 forza CPU; det_size=(160, 160) es suficiente para crops de cara del SGIE.
        CPUExecutionProvider evita conflictos con TensorRT que ya usa la GPU para los SGIEs.

        NO usar CUDAExecutionProvider aquí: el wheel onnxruntime-gpu instalado en
        Dockerfile.jetson (nschloe/onnxruntime-aarch64-ubuntu22) es el mismo que causó
        "kernel Cask errors" (choque de contexto CUDA con TensorRT) durante la migración
        de OSNet — ver Future.md "CHANGE TO OSNET1" y ErrorHistory.md. Por eso OSNet se
        movió a un SGIE nativo de DeepStream en vez de correr sobre onnxruntime-gpu.
        Se probó habilitar CUDA aquí (2026-07-07) y se revirtió sin desplegar por el mismo
        riesgo documentado — el camino real a GPU es exportar buffalo_l a TensorRT y
        correrlo como SGIE nativo, no onnxruntime-gpu con este wheel.
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

    def _load_db(self) -> None:
        """Carga known_faces.json desde disco. Soporta formato legacy (nombre-clave) y nuevo (UUID-clave).

        Soporta ambos formatos:
          Legacy: {"Juan Perez": [[...], [...]]}
          Nuevo:  {"<uuid>": {"name": "Juan Perez", "embeddings": [[...], [...]]}}

        Retorna silenciosamente vacío si el archivo no existe o tiene errores.
        """
        path = Path(self._db_path)
        if not path.exists():
            logger.warning("Face DB not found: %s — recognition disabled.", path)
            self._db = {}
            self._uuid_to_name = {}
            return
        try:
            raw = json.loads(path.read_text())
            self._db, self._uuid_to_name = _parse_raw_db(raw)
            logger.info("Face DB loaded: %d person(s) from %s", len(self._db), path)
        except Exception as e:
            logger.error("Failed to load face DB: %s", e)
            self._db = {}
            self._uuid_to_name = {}

    def _process(self, face_crop_bgr: np.ndarray, global_id: str):
        """Infiere identidad de un crop de cara y acumula votos en una ventana deslizante.

        Sistema de votación: requiere FACE_VOTES_REQUIRED coincidencias (de las últimas
        FACE_VOTES_REQUIRED muestras) para (re)bloquear la identidad por global_id,
        reduciendo falsos positivos en frames difíciles. A diferencia del esquema
        anterior por track_id, aquí NUNCA se deja de votar aunque ya haya un candado —
        si ReID (OSNet) le pasó este global_id a una persona distinta por error
        (ej. uniformes parecidos), un cambio sostenido de mayoría en la ventana corrige
        el candado en vez de quedar pegado al primer match para siempre.

        Muestras de mal ángulo (yaw > FACE_MAX_YAW_DEGREES) no cuentan como voto —
        se descartan antes de tocar la ventana, igual que "sin cara detectada".
        """
        if not self._db:
            return  # DB vacía — reconocimiento deshabilitado para este cliente

        self._maybe_save_crop_sample(face_crop_bgr, global_id)

        # CLAHE sobre el canal L (LAB) — mejora contraste sin distorsionar color,
        # ayuda al detector interno de InsightFace en escenas de poca luz. Barato
        # (una operación por crop, no por frame) y no afecta la medición de yaw
        # (geometría de landmarks, no intensidad de píxel).
        face_crop_bgr = _apply_clahe(face_crop_bgr)

        # Detectar y alinear la cara en el crop usando InsightFace
        faces = self._app.get(face_crop_bgr)
        if not faces:
            return  # sin cara detectada en este crop (blur, oclusión, etc.)

        face = faces[0]  # mayor bbox = más prominente

        # face.pose = [pitch, yaw, roll], viene gratis del modelo landmark_3d_68
        # que buffalo_l ya carga — sin inferencia extra. Cara muy de perfil no
        # cuenta como voto: el embedding sale como ruido sin importar el modelo.
        yaw = float(face.pose[1]) if face.pose is not None else 0.0
        if abs(yaw) > FACE_MAX_YAW_DEGREES:
            logger.debug("Face yaw rechazado global_id=%s yaw=%.1f°", global_id, yaw)
            return

        emb = face.normed_embedding  # vector 512-dim ya L2-normalizado por InsightFace

        # Comparar con la DB y obtener el mejor match
        identity_key, sim = self._match(emb)
        if identity_key is None:
            identity_key = "Unknown"  # below threshold — unrecognised face

        with self._lock:
            # Ventana deslizante de las últimas FACE_VOTES_REQUIRED predicciones —
            # se sigue alimentando aunque ya haya un candado vigente.
            votes = self._votes.setdefault(global_id, deque(maxlen=FACE_VOTES_REQUIRED))
            votes.append(identity_key)
            if len(votes) >= FACE_VOTES_REQUIRED:
                # Elegir la identidad más votada (mayoritaria) en la ventana actual.
                winner = Counter(votes).most_common(1)[0][0]
                current = self._locked.get(global_id)
                if current is None:
                    self._locked[global_id] = (winner, sim, yaw)
                    logger.debug("Face locked global_id=%s → %s (sim=%.3f, yaw=%.1f°)",
                                 global_id, self.get_display_name(winner), sim, yaw)
                elif current[0] != winner:
                    # La mayoría de la ventana cambió respecto al candado vigente —
                    # corregir el tag (salvaguarda de uniformes parecidos).
                    self._locked[global_id] = (winner, sim, yaw)
                    logger.info(
                        "Face re-tagged global_id=%s: %s → %s (sim=%.3f, yaw=%.1f°)",
                        global_id, self.get_display_name(current[0]),
                        self.get_display_name(winner), sim, yaw,
                    )
                else:
                    # Misma identidad reconfirmada — solo refresca la confianza.
                    self._locked[global_id] = (winner, sim, yaw)

    def _maybe_save_crop_sample(self, face_crop_bgr: np.ndarray, global_id: str) -> None:
        """ponytail: diagnóstico temporal — guarda hasta FACE_CROP_SAMPLE_MAX crops
        crudos (antes de CLAHE) a disco para inspección visual manual. Quitar esta
        llamada (y el método) cuando ya no se necesite ver a ojo qué le está
        llegando al reconocedor."""
        if self._crop_sample_count >= FACE_CROP_SAMPLE_MAX:
            return
        try:
            self._crop_sample_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self._crop_sample_dir / f"{ts}_{global_id}_{self._crop_sample_count:02d}.jpg"
            cv2.imwrite(str(path), face_crop_bgr)
            self._crop_sample_count += 1
        except Exception as e:
            logger.warning("No se pudo guardar crop de muestra: %s", e)

    def _match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Busca la persona con mayor similitud coseno en la DB.

        Compara el embedding contra todos los embeddings de cada persona.
        Retorna (uuid_or_name, similitud) si supera FACE_SIMILARITY_THRESHOLD, o (None, sim) si no.
        Ambos vectores están L2-normalizados, por lo que dot product = similitud coseno.
        """
        best_key: Optional[str] = None
        best_sim: float = -1.0

        # Iterar sobre cada persona y todos sus embeddings (puede haber varios ángulos)
        for key, db_embeddings in self._db.items():
            for db_emb in db_embeddings:
                sim = float(np.dot(embedding, db_emb))  # cosine sim: ambos L2-normalizados
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

        # Retornar solo si supera el threshold; de lo contrario indicar desconocido
        if best_sim >= FACE_SIMILARITY_THRESHOLD:
            return best_key, best_sim
        return None, best_sim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(crop_bgr: np.ndarray) -> np.ndarray:
    """Aplica CLAHE al canal L (LAB) de un crop BGR — mejora contraste local sin
    distorsionar el balance de color. clipLimit=2.0/tileGridSize=8x8 son los
    valores por defecto recomendados de OpenCV, sin calibrar contra footage real."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _parse_raw_db(raw: dict) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, str]]:
    """Parsea known_faces.json en estructura interna y mapa UUID → nombre.

    Detecta el formato por el tipo del primer valor:
      - list  → formato legacy (nombre-clave): {"Juan Perez": [[...], ...]}
      - dict  → formato nuevo (UUID-clave):   {"<uuid>": {"name": "...", "embeddings": [[...]]}}

    Returns:
        db:           Dict clave → lista de np.ndarray float32 para dot product.
        uuid_to_name: Dict clave → nombre legible (en legacy, nombre == clave).
    """
    db: Dict[str, List[np.ndarray]] = {}
    uuid_to_name: Dict[str, str] = {}

    for key, value in raw.items():
        if isinstance(value, list):
            # Formato legacy: la clave ES el nombre
            db[key] = [np.array(e, dtype=np.float32) for e in value]
            uuid_to_name[key] = key  # nombre == clave en el formato antiguo
        elif isinstance(value, dict):
            # Formato nuevo: clave es UUID, value tiene "name" y "embeddings"
            name = value.get("name", key)
            embeddings = value.get("embeddings", [])
            db[key] = [np.array(e, dtype=np.float32) for e in embeddings]
            uuid_to_name[key] = name

    return db, uuid_to_name
