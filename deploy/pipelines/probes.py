"""
probes.py — NX Computing AI | Edge Inference Core
Probe de GStreamer para extracción de metadatos DeepStream y envío a API REST.

Arquitectura:
  PGIE (PeopleNet, gie-id=1) detecta person/bag/face en el frame completo.
  Handlers opcionales (uno por capability activa) procesan cada persona detectada.
  PoseWorker y FaceRecognizer corren en hilos de fondo (patrón async NxApiClient).

Para agregar un nuevo modelo:
  1. Crear una clase que implemente process(obj_meta, frame_num, frame_np).
  2. Añadirla a _HANDLER_REGISTRY bajo su capability name.
  3. Si necesita un worker async, agregarlo en init_workers() y wirearlo en init_handlers().
  4. Añadir la entrada en SGIE_CONFIGS en app.py (o None si es Python worker).
"""
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import pyds
import queue
import logging
import threading
import time
import uuid
import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import requests

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURACIÓN
# Ajusta estos valores según tu entorno. Idealmente muévalos a un .env o config.
# ==============================================================================
JETSON_ID: str    = os.environ.get("JETSON_ID",  os.uname().nodename)
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY: str      = os.environ.get("API_KEY",    "your-api-key")

# Mapa pad_index → número de canal real del DVR.
# Se inicializa desde app.py llamando a init_channel_map(cfg.channels).
# Ejemplo: channels=[1,3,5] → {0: 1, 1: 3, 2: 5}
_channel_map: Dict[int, int] = {}

def init_channel_map(channels: list):
    """Llamar desde app.py después de load_config(), antes de arrancar el pipeline."""
    global _channel_map
    _channel_map = {idx: ch for idx, ch in enumerate(channels)}
    logger.info("Channel map: %s", _channel_map)

def _camera_id_for(pad_index: int) -> str:
    """Devuelve un camera_id con el canal real, ej. 'jetson-nx-001-ch03'."""
    ch = _channel_map.get(pad_index, pad_index)
    return f"{JETSON_ID}-ch{ch:02d}"

# ── GIE unique-ids ────────────────────────────────────────────────────────────
PGIE_UNIQUE_ID: int      = 1   # PeopleNet
SGIE_AGE_GENDER_ID: int  = 2   # ResNet-18 Pedestrian Attr
SGIE_FACE_DETECT_ID: int = 3   # FaceDetectIR SGIE

# ── PGIE class ids ────────────────────────────────────────────────────────────
PGIE_CLASS_PERSON: int = 0
PGIE_CLASS_BAG:    int = 1
PGIE_CLASS_FACE:   int = 2

# ── Confidence thresholds ─────────────────────────────────────────────────────
OSD_CONFIDENCE_THRESHOLD: float      = 0.40
MIN_CLASSIFICATION_PROB: float       = 0.3
FACE_DET_CONFIDENCE_THRESHOLD: float = 0.40

# ── Age/gender voting ─────────────────────────────────────────────────────────
VOTE_SAMPLE_INTERVAL: int = 5
VOTES_REQUIRED: int       = 10
VOTE_MIN_WIDTH: int       = 64
VOTE_MIN_HEIGHT: int      = 160

# ── Fall detection ────────────────────────────────────────────────────────────
POSE_SAMPLE_INTERVAL: int  = 10   # enqueue crop every N frames per person
POSE_MIN_PERSON_WIDTH: int = 40   # skip persons narrower than this (too far)

# ── Face recognition ──────────────────────────────────────────────────────────
FACE_SAMPLE_INTERVAL: int  = 30   # frames between recognition attempts per track

# ── Analytics ─────────────────────────────────────────────────────────────────
ANALYTICS_SEND_INTERVAL_SECS: float = 60.0

# ── Crop capture ──────────────────────────────────────────────────────────────
CROPS_DIR: str            = "crops"
CROP_SAMPLE_INTERVAL: int = 15
CROP_MAX_PER_PERSON: int  = 5
CROP_MIN_WIDTH: int       = 40
CROP_MIN_HEIGHT: int      = 80


# ==============================================================================
# 1. BUS CALL — manejo de mensajes del pipeline
# ==============================================================================
def bus_call(_bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("Fin del flujo de video (EOS).")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("GStreamer WARNING: %s — %s", err, debug)
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("GStreamer ERROR: %s — %s", err, debug)
        loop.quit()
    return True


# ==============================================================================
# 2. CLIENTE DE API REST NO BLOQUEANTE
#    Hilo dedicado + cola FIFO. El probe solo encola; nunca bloquea el pipeline.
# ==============================================================================
class NxApiClient:
    """
    Envía peticiones HTTP al backend en un hilo de fondo independiente.
    El probe de GStreamer solo hace `enqueue()` (O(1), sin I/O), garantizando
    que las llamadas de red nunca impacten los FPS del pipeline.
    """

    def __init__(self, base_url: str, api_key: str, max_queue_size: int = 512):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        })
        self._worker_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="nx-api-worker",
        )
        self._worker_thread.start()
        logger.info("NxApiClient iniciado → %s", self._base_url)

    def stop(self):
        self._running = False
        self._queue.put(None)  # Señal de parada
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._session.close()
        logger.info("NxApiClient detenido.")

    # ------------------------------------------------------------------
    # Encolar peticiones (llamado desde el probe, sin bloquear)
    # ------------------------------------------------------------------
    def enqueue(self, method: str, endpoint: str, payload: Optional[dict] = None):
        try:
            self._queue.put_nowait((method, endpoint, payload))
        except queue.Full:
            logger.warning("Cola API llena — descartando: %s %s", method, endpoint)

    # ------------------------------------------------------------------
    # Worker interno
    # ------------------------------------------------------------------
    def _worker_loop(self):
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            method, endpoint, payload = item
            self._send(method, endpoint, payload)
            self._queue.task_done()

    def _send(self, method: str, endpoint: str, payload: Optional[dict]):
        url = f"{self._base_url}{endpoint}"
        try:
            resp = self._session.request(
                method=method,
                url=url,
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
            logger.debug("%s %s → %d", method, endpoint, resp.status_code)
        except requests.exceptions.Timeout:
            logger.warning("Timeout: %s %s", method, url)
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP %d: %s %s → %s",
                         e.response.status_code, method, url, e.response.text[:300])
        except requests.exceptions.ConnectionError:
            logger.debug("Sin conexión: %s", url)

    # ------------------------------------------------------------------
    # Métodos de negocio (wrappers de alto nivel)
    # ------------------------------------------------------------------
    def post_detection_event(self, camera_id: str, frame_num: int, detections: List[dict],
                             event_type: str = "person_detection"):
        """
        Envía un evento de detección al backend.
        event_type: "person_detection" (primera aparición) o
                    "person_classified" (clasificación demográfica confirmada).
        """
        payload = {
            "event_id": str(uuid.uuid4()),
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "severity": "info",
            "status": "open",
            "detections": detections,
        }
        self.enqueue("POST", "/api/events", payload)

    def post_analytics_snapshot(self, camera_id: str, stats: dict):
        """Envía un snapshot de analytics agregadas (edad/género, conteo) por cámara."""
        payload = {
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **stats,
        }
        self.enqueue("POST", "/api/analytics/ingest", payload)

    def ack_event(self, event_id: str):
        self.enqueue("POST", f"/api/events/{event_id}/ack")

    def resolve_event(self, event_id: str):
        self.enqueue("POST", f"/api/events/{event_id}/resolve")

    def add_event_note(self, event_id: str, text: str):
        self.enqueue("POST", f"/api/events/{event_id}/notes", {"text": text})

    def post_crop(self, camera_id: str, track_id: int, frame_num: int, crop_b64: str, bbox: dict):
        """Envía un recorte de persona a la API para construcción de dataset."""
        payload = {
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "track_id": track_id,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_b64": crop_b64,
            "bbox": bbox,
        }
        self.enqueue("POST", "/api/crops", payload)

    def post_reference_frame(self, camera_id: str, frame_num: int, frame_b64: str, width: int, height: int):
        """
        Envía un frame de referencia sin personas al backend.
        El dashboard lo usa como fondo para superponer el heatmap de posiciones.
        Se envía una sola vez por sesión por cámara (primer frame limpio disponible).
        """
        payload = {
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_b64": frame_b64,
            "width": width,
            "height": height,
        }
        self.enqueue("POST", "/api/cameras/reference-frame", payload)


# Instancia global — se inicializa en main() antes de arrancar el pipeline
api_client = NxApiClient(base_url=API_BASE_URL, api_key=API_KEY)


# ==============================================================================
# 3. HELPERS DE METADATOS DEEPSTREAM
# ==============================================================================
def _iter_pyds_list(pyds_list, cast_fn):
    """
    Generador seguro sobre listas enlazadas de pyds.
    Maneja StopIteration internamente para evitar repetición de try/except.
    """
    node = pyds_list
    while node is not None:
        try:
            yield cast_fn(node.data)
        except StopIteration:
            return
        try:
            node = node.next
        except StopIteration:
            return


# Mapa de las 6 clases del modelo Pedestrian Attr → (género legible, grupo de edad legible)
_AGE_GENDER_LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "female_adult":  ("Mujer",  "Adulta"),
    "female_senior": ("Mujer",  "Mayor"),
    "female_young":  ("Mujer",  "Joven"),
    "male_adult":    ("Hombre", "Adulto"),
    "male_senior":   ("Hombre", "Mayor"),
    "male_young":    ("Hombre", "Joven"),
}

# Mismo mapa pero para el payload de la API (valores en inglés, snake_case)
_AGE_GENDER_API_MAP: Dict[str, Tuple[str, str]] = {
    "female_adult":  ("female", "adult"),
    "female_senior": ("female", "senior"),
    "female_young":  ("female", "young"),
    "male_adult":    ("male",   "adult"),
    "male_senior":   ("male",   "senior"),
    "male_young":    ("male",   "young"),
}


def _parse_age_gender(classifier_meta) -> Tuple[str, str, str, float]:
    """
    Extrae el label y la probabilidad del modelo SGIE (ResNet-18 Pedestrian Attr).

    Retorna: (raw_label, gender_display, age_display, prob)
      - raw_label      → clave en analytics ("male_adult", etc.)
      - gender_display → texto OSD ("Hombre", "Mujer")
      - age_display    → texto OSD ("Adulto", "Joven", etc.)
      - prob           → confianza 0.0–1.0 (result_prob de NvDsLabelInfo)

    Retorna ("", "", "", 0.0) si el modelo aún no infirió.
    """
    for label_info in _iter_pyds_list(
        classifier_meta.label_info_list, pyds.NvDsLabelInfo.cast
    ):
        raw = label_info.result_label.strip().rstrip("\x00").strip()
        if not raw:
            continue
        if raw in _AGE_GENDER_LABEL_MAP:
            gender_disp, age_disp = _AGE_GENDER_LABEL_MAP[raw]
            prob = min(float(label_info.result_prob), 1.0)
            return raw, gender_disp, age_disp, prob
        logger.debug("SGIE label desconocido: %r (gie-id=%d)", raw,
                     classifier_meta.unique_component_id)
    return "", "", "", 0.0


def _build_detection_dict(obj_meta) -> dict:
    """Serializa un NvDsObjectMeta a un dict listo para JSON."""
    r = obj_meta.rect_params
    return {
        "track_id": int(obj_meta.object_id),
        "class_id": int(obj_meta.class_id),
        "confidence": round(float(obj_meta.confidence), 3),
        "bbox": {
            "left":   int(r.left),
            "top":    int(r.top),
            "width":  int(r.width),
            "height": int(r.height),
        },
    }


def _set_osd_text(
    obj_meta,
    text: str,
    border_color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 1.0),
):
    """
    Aplica estilo y texto al OSD de un objeto.
    border_color: (R, G, B, A) — azul mientras analiza, verde cuando clasificado.
    """
    obj_meta.text_params.display_text = text

    fp = obj_meta.text_params.font_params
    fp.font_name = "Sans"
    fp.font_size = 12
    fp.font_color.set(1.0, 1.0, 1.0, 1.0)   # Blanco opaco

    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)  # Negro 60%

    obj_meta.rect_params.border_color.set(*border_color)
    obj_meta.rect_params.border_width = 2


# ==============================================================================
# 4. HANDLERS DE PIPELINE
#    Cada capability activa instancia un handler que procesa cada objeto
#    detectado por el PGIE. Los handlers son registrados en init_handlers()
#    y despachados desde osd_sink_pad_buffer_probe().
#
#    Contrato de process():
#      Inputs : obj_meta (NvDsObjectMeta), frame_num (int)
#      Returns: HandlerResult o None (sin acción)
# ==============================================================================

class _HandlerResult:
    """Resultado que un handler devuelve al probe para OSD y API."""
    __slots__ = ("osd_text", "border_color", "event_type", "det_extra", "analytics_update")

    def __init__(
        self,
        osd_text: str = "",
        border_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        event_type: str = "",          # si no vacío → emite evento API adicional
        det_extra: Optional[dict] = None,
        analytics_update: Optional[dict] = None,
    ):
        self.osd_text = osd_text
        self.border_color = border_color
        self.event_type = event_type
        self.det_extra = det_extra or {}
        self.analytics_update = analytics_update or {}


class _AgeGenderHandler:
    """
    Clasificación de género y grupo de edad por votación sobre el SGIE ResNet-18.
    Acumula VOTES_REQUIRED muestras del SGIE antes de fijar el resultado.
    """

    def __init__(self):
        self._cache: Dict[int, Tuple[str, str, str, float]] = {}
        self._votes: Dict[int, List[str]] = {}
        self._vote_last_frame: Dict[int, int] = {}

    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        p_track_id = int(obj_meta.object_id)
        bbox_w = int(obj_meta.rect_params.width)
        bbox_h = int(obj_meta.rect_params.height)
        person_too_small = bbox_w < VOTE_MIN_WIDTH or bbox_h < VOTE_MIN_HEIGHT

        # Read current SGIE output for this object
        raw_label, gender_disp, age_disp, prob = "", "", "", 0.0
        _cls_items = 0
        _seen_ids: list = []
        for cls_meta in _iter_pyds_list(
            obj_meta.classifier_meta_list, pyds.NvDsClassifierMeta.cast
        ):
            _cls_items += 1
            _seen_ids.append(cls_meta.unique_component_id)
            if cls_meta.unique_component_id == SGIE_AGE_GENDER_ID:
                raw_label, gender_disp, age_disp, prob = _parse_age_gender(cls_meta)
                break

        if frame_num % 30 == 0:
            logger.info(
                "DIAG P#%d frame=%d | cls_items=%d ids=%s | label=%r prob=%.3f",
                p_track_id, frame_num, _cls_items, _seen_ids, raw_label, prob,
            )

        is_new_classification = False
        if p_track_id in self._cache:
            raw_label, gender_disp, age_disp, prob = self._cache[p_track_id]
        elif person_too_small:
            return _HandlerResult(
                osd_text=f"P#{p_track_id} | Muy lejos",
                border_color=(0.8, 0.8, 0.0, 1.0),
            )
        else:
            last_frame = self._vote_last_frame.get(p_track_id, -VOTE_SAMPLE_INTERVAL)
            if (raw_label and prob >= MIN_CLASSIFICATION_PROB
                    and (frame_num - last_frame) >= VOTE_SAMPLE_INTERVAL):
                votes = self._votes.setdefault(p_track_id, [])
                votes.append(raw_label)
                self._vote_last_frame[p_track_id] = frame_num

                if len(votes) >= VOTES_REQUIRED:
                    winner = max(set(votes), key=votes.count)
                    vote_prob = votes.count(winner) / len(votes)
                    w_gender, w_age = _AGE_GENDER_LABEL_MAP[winner]
                    self._cache[p_track_id] = (winner, w_gender, w_age, vote_prob)
                    raw_label, gender_disp, age_disp, prob = self._cache[p_track_id]
                    is_new_classification = True
                    logger.debug(
                        "P#%d votación completa: %s (%.0f%% de %d votos)",
                        p_track_id, winner, vote_prob * 100, len(votes),
                    )
                else:
                    n_votes = len(votes)
                    return _HandlerResult(
                        osd_text=f"P#{p_track_id} | Analizando... ({n_votes}/{VOTES_REQUIRED})",
                        border_color=(0.0, 0.8, 1.0, 1.0),
                    )
            else:
                n_votes = len(self._votes.get(p_track_id, []))
                return _HandlerResult(
                    osd_text=f"P#{p_track_id} | Analizando... ({n_votes}/{VOTES_REQUIRED})",
                    border_color=(0.0, 0.8, 1.0, 1.0),
                )

        api_gender, api_age = _AGE_GENDER_API_MAP[raw_label]
        analytics = {}
        det_extra = {}
        event_type = ""
        if is_new_classification:
            event_type = "person_classified"
            det_extra = {
                "demographics": {
                    "gender":     api_gender,
                    "age_group":  api_age,
                    "label":      raw_label,
                    "confidence": round(prob, 3),
                }
            }
            analytics = {
                "age_gender_classes": raw_label,
                "gender_key": "gender_male" if raw_label.startswith("male") else "gender_female",
            }

        return _HandlerResult(
            osd_text=f"P#{p_track_id} | {gender_disp} | {age_disp} {prob:.0%}",
            border_color=(0.0, 1.0, 0.0, 1.0),
            event_type=event_type,
            det_extra=det_extra,
            analytics_update=analytics,
        )


class _EppHandler:
    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        return None


class _FireSmokeHandler:
    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        return None


class _LicensePlateHandler:
    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        return None


class _FallDetectionHandler:
    """
    Pose-based fall detection via async PoseWorker (MoveNet ONNX).
    Enqueues person crops non-blocking; reads results on subsequent frames.
    """

    def __init__(self):
        self._worker = None

    def set_worker(self, worker) -> None:
        self._worker = worker

    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        if self._worker is None:
            return None

        p_track_id = int(obj_meta.object_id)
        r = obj_meta.rect_params
        bbox_w, bbox_h = int(r.width), int(r.height)
        bbox = {"left": max(0, int(r.left)), "top": max(0, int(r.top)),
                "width": bbox_w, "height": bbox_h}

        if (frame_np is not None
                and frame_num % POSE_SAMPLE_INTERVAL == 0
                and bbox_w >= POSE_MIN_PERSON_WIDTH):
            l, t = bbox["left"], bbox["top"]
            crop = frame_np[t:t + bbox_h, l:l + bbox_w]
            if crop.size > 0:
                self._worker.enqueue(crop, p_track_id, frame_num, bbox)

        result = self._worker.get_result(p_track_id)
        if result is None:
            return None

        if result.is_falling:
            is_new_fall = self._worker.pop_new_fall(p_track_id)
            return _HandlerResult(
                osd_text=f"P#{p_track_id} | CAIDA",
                border_color=(1.0, 0.0, 0.0, 1.0),
                event_type="fall_detected" if is_new_fall else "",
                det_extra={
                    "fall_score":  result.fall_score,
                    "avg_kp_conf": round(result.avg_conf, 3),
                } if is_new_fall else {},
            )

        return None


class _FaceRecognitionHandler:
    """
    Face recognition via async FaceRecognizer (InsightFace ArcFace).
    Receives face detections from FaceDetectIR SGIE (gie-id=3), crops face from
    the full frame, enqueues for embedding extraction, and caches results per track.

    This handler is NOT in _HANDLER_REGISTRY — it is dispatched separately via
    _face_handler because it processes SGIE_FACE_DETECT_ID objects, not PGIE objects.
    """

    def __init__(self, worker):
        self._worker = worker
        self._last_sample: Dict[int, int] = {}
        self._cache: Dict[int, Tuple[str, float]] = {}

    def process_face(
        self,
        face_obj_meta,
        frame_num: int,
        frame_np,
        persons_meta: list,
        camera_id: str,
    ) -> None:
        if self._worker is None or frame_np is None:
            return
        if face_obj_meta.confidence < FACE_DET_CONFIDENCE_THRESHOLD:
            return

        parent_track_id = self._find_parent_track(face_obj_meta, persons_meta)
        if parent_track_id is None:
            return

        last = self._last_sample.get(parent_track_id, -FACE_SAMPLE_INTERVAL)
        if frame_num - last < FACE_SAMPLE_INTERVAL:
            return

        r = face_obj_meta.rect_params
        l = max(0, int(r.left))
        t = max(0, int(r.top))
        w = max(1, int(r.width))
        h = max(1, int(r.height))
        face_crop = frame_np[t:t + h, l:l + w]
        if face_crop.size == 0:
            return

        self._last_sample[parent_track_id] = frame_num
        self._worker.enqueue(face_crop, parent_track_id, frame_num, camera_id)

        result = self._worker.get_result(parent_track_id)
        if result:
            self._cache[parent_track_id] = result

    def get_identity(self, track_id: int) -> Optional[Tuple[str, float]]:
        result = self._worker.get_result(track_id) if self._worker else None
        if result:
            self._cache[track_id] = result
        return self._cache.get(track_id)

    @staticmethod
    def _find_parent_track(face_obj_meta, persons_meta: list) -> Optional[int]:
        """Find the person whose bbox contains the center of this face detection."""
        fr = face_obj_meta.rect_params
        face_cx = fr.left + fr.width / 2
        face_cy = fr.top + fr.height / 2
        for p in persons_meta:
            pr = p.rect_params
            if (pr.left <= face_cx <= pr.left + pr.width
                    and pr.top <= face_cy <= pr.top + pr.height):
                return int(p.object_id)
        return None


# ==============================================================================
# 5. WORKER GLOBALS + LIFECYCLE
# ==============================================================================

_pose_worker     = None   # PoseWorker (fall_detection)
_face_recognizer = None   # FaceRecognizer (face_recognition)


def init_workers(
    pipeline_capabilities: List[str],
    model_dir: str,
    face_db_path: str = "",
) -> None:
    global _pose_worker, _face_recognizer

    if "fall_detection" in pipeline_capabilities:
        from pose_worker import PoseWorker
        model_path = str(Path(model_dir) / "movenet" / "movenet_singlepose_lightning_192.onnx")
        _pose_worker = PoseWorker(model_path)
        _pose_worker.start()

    if "face_recognition" in pipeline_capabilities:
        from face_recognizer import FaceRecognizer
        _face_recognizer = FaceRecognizer(db_path=face_db_path, model_root=str(Path(model_dir) / "insightface"))
        _face_recognizer.start()


def stop_workers() -> None:
    if _pose_worker is not None:
        _pose_worker.stop()
    if _face_recognizer is not None:
        _face_recognizer.stop()


# ==============================================================================
# 6. HANDLER REGISTRY + INIT
# ==============================================================================

_active_handlers: List = []
_face_handler: Optional[_FaceRecognitionHandler] = None

_HANDLER_REGISTRY = {
    "age_gender":    _AgeGenderHandler,
    "epp_detection": _EppHandler,
    "fire_smoke":    _FireSmokeHandler,
    "license_plate": _LicensePlateHandler,
    "fall_detection": _FallDetectionHandler,
    # face_recognition is NOT here — handled separately via _face_handler
}


def init_handlers(pipeline_capabilities: List[str]) -> None:
    global _active_handlers, _face_handler
    _active_handlers = []
    _face_handler = None

    for cap in pipeline_capabilities:
        cls = _HANDLER_REGISTRY.get(cap)
        if cls:
            handler = cls()
            _active_handlers.append(handler)
            if isinstance(handler, _FallDetectionHandler) and _pose_worker is not None:
                handler.set_worker(_pose_worker)
                logger.info("FallDetectionHandler → PoseWorker")

        if cap == "face_recognition" and _face_recognizer is not None:
            _face_handler = _FaceRecognitionHandler(_face_recognizer)
            logger.info("FaceRecognitionHandler → FaceRecognizer")

    names = [type(h).__name__ for h in _active_handlers]
    if _face_handler:
        names.append("_FaceRecognitionHandler")
    logger.info("Active handlers: %s", names if names else ["(none — people_counting only)"])


# ==============================================================================
# 7. PROBE PRINCIPAL — OSD + ENVÍO A API
# ==============================================================================

# IDs de personas ya notificadas a la API (primera aparición)
_person_notified: set = set()

# Crops capturados por track_id: cantidad guardada y último frame muestreado
_crop_counts: Dict[int, int] = {}
_crop_last_frame: Dict[int, int] = {}

# Frame de referencia enviado por pad_index — una vez por cámara por sesión
_reference_frame_sent: Dict[int, bool] = {}

# Analytics y timestamp por pad_index — conteos independientes por cámara
_analytics: Dict[int, Dict] = {}
_analytics_last_sent: Dict[int, float] = {}


def _get_analytics(pad_index: int) -> Dict:
    if pad_index not in _analytics:
        _analytics[pad_index] = {
            "person_count": 0, "gender_male": 0,
            "gender_female": 0, "age_gender_classes": {},
        }
    return _analytics[pad_index]


def _get_analytics_last_sent(pad_index: int) -> float:
    if pad_index not in _analytics_last_sent:
        _analytics_last_sent[pad_index] = time.monotonic()
    return _analytics_last_sent[pad_index]


def _save_and_send_crop(
    crop_bgr: np.ndarray,
    camera_id: str,
    track_id: int,
    frame_num: int,
    bbox: dict,
) -> None:
    person_dir = Path(CROPS_DIR) / camera_id / str(track_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    filepath = person_dir / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filepath), crop_bgr)

    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    crop_b64 = base64.b64encode(buf).decode("utf-8")
    api_client.post_crop(camera_id, track_id, frame_num, crop_b64, bbox)


def osd_sink_pad_buffer_probe(_pad, info):
    """
    Probe conectado al sink-pad de nvdsosd.
    Itera sobre objetos del frame: personas del PGIE y caras del FaceDetectIR SGIE.
    Despacha handlers activos y envía eventos / analytics a la API REST.
    """
    global _person_notified, _crop_counts, _crop_last_frame

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame_meta in _iter_pyds_list(
        batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast
    ):
        frame_num = frame_meta.frame_num
        pad_index = frame_meta.pad_index
        camera_id = _camera_id_for(pad_index)
        frame_detections: List[dict] = []

        frame_np = None
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_np = np.array(n_frame, copy=True, order='C')
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            if frame_num % 30 == 0:
                logger.warning("get_nvds_buf_surface falló frame=%d: %s", frame_num, e)

        # ── First pass: collect persons and face detections ───────────────────
        persons_meta: List = []
        face_metas: List   = []
        for obj_meta in _iter_pyds_list(
            frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast
        ):
            uid = obj_meta.unique_component_id
            if (uid == PGIE_UNIQUE_ID
                    and obj_meta.confidence >= OSD_CONFIDENCE_THRESHOLD
                    and int(obj_meta.class_id) == PGIE_CLASS_PERSON):
                persons_meta.append(obj_meta)
            elif uid == SGIE_FACE_DETECT_ID:
                face_metas.append(obj_meta)

        # ── Face recognition: process SGIE detections ────────────────────────
        if _face_handler and face_metas and frame_np is not None:
            for face_obj_meta in face_metas:
                _face_handler.process_face(
                    face_obj_meta, frame_num, frame_np, persons_meta, camera_id
                )

        # ── Second pass: process each person ─────────────────────────────────
        for obj_meta in persons_meta:
            p_track_id = int(obj_meta.object_id)

            _set_osd_text(obj_meta, f"P#{p_track_id}", border_color=(0.2, 0.6, 1.0, 1.0))

            for handler in _active_handlers:
                result = handler.process(obj_meta, frame_num, frame_np=frame_np)
                if result is None:
                    continue
                if result.osd_text:
                    _set_osd_text(obj_meta, result.osd_text, border_color=result.border_color)
                if result.event_type:
                    det = _build_detection_dict(obj_meta)
                    det["type"] = result.event_type
                    det.update(result.det_extra)
                    frame_detections.append(det)
                    an = _get_analytics(pad_index)
                    au = result.analytics_update
                    if "age_gender_classes" in au:
                        lbl = au["age_gender_classes"]
                        an["age_gender_classes"][lbl] = an["age_gender_classes"].get(lbl, 0) + 1
                    if "gender_key" in au:
                        an[au["gender_key"]] += 1

            # Overlay face identity on top of existing OSD text
            if _face_handler:
                identity = _face_handler.get_identity(p_track_id)
                if identity:
                    name, conf = identity
                    if name != "Desconocido":
                        cur = obj_meta.text_params.display_text or f"P#{p_track_id}"
                        _set_osd_text(
                            obj_meta,
                            f"{cur} | {name} {conf:.0%}",
                            border_color=(0.2, 1.0, 0.4, 1.0),
                        )

            # Crop capture for dataset (always active)
            if frame_np is not None:
                last_crop = _crop_last_frame.get(p_track_id, -CROP_SAMPLE_INTERVAL)
                count = _crop_counts.get(p_track_id, 0)
                if (count < CROP_MAX_PER_PERSON
                        and (frame_num - last_crop) >= CROP_SAMPLE_INTERVAL):
                    r = obj_meta.rect_params
                    l = max(0, int(r.left))
                    t = max(0, int(r.top))
                    w, h = int(r.width), int(r.height)
                    crop = frame_np[t:t + h, l:l + w]
                    if crop.shape[0] >= CROP_MIN_HEIGHT and crop.shape[1] >= CROP_MIN_WIDTH:
                        _save_and_send_crop(
                            crop, camera_id, p_track_id, frame_num,
                            {"left": l, "top": t, "width": w, "height": h},
                        )
                        _crop_counts[p_track_id] = count + 1
                        _crop_last_frame[p_track_id] = frame_num

            # First appearance event (people_counting — always active)
            if p_track_id not in _person_notified:
                _person_notified.add(p_track_id)
                det = _build_detection_dict(obj_meta)
                det["type"] = "person"
                frame_detections.append(det)
                _get_analytics(pad_index)["person_count"] += 1

        # Reference frame for heatmap — first empty frame per camera
        if (not _reference_frame_sent.get(pad_index, False)
                and frame_np is not None
                and not frame_detections
                and frame_meta.num_obj_meta == 0):
            h, w = frame_np.shape[:2]
            _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_b64 = base64.b64encode(buf).decode("utf-8")
            api_client.post_reference_frame(camera_id, frame_num, frame_b64, w, h)
            _reference_frame_sent[pad_index] = True
            logger.info("Frame de referencia enviado camera=%s (frame=%d, %dx%d)",
                        camera_id, frame_num, w, h)

        if frame_detections:
            has_classified = any(d.get("type") not in ("person", "") for d in frame_detections)
            evt_type = frame_detections[-1].get("type", "person_detection") if has_classified \
                else "person_detection"
            api_client.post_detection_event(camera_id, frame_num, frame_detections, evt_type)

        now = time.monotonic()
        if now - _get_analytics_last_sent(pad_index) >= ANALYTICS_SEND_INTERVAL_SECS:
            an = _get_analytics(pad_index)
            api_client.post_analytics_snapshot(camera_id, {
                "people_count":       an["person_count"],
                "gender_male":        an["gender_male"],
                "gender_female":      an["gender_female"],
                "age_gender_classes": an["age_gender_classes"],
            })
            _analytics[pad_index] = {
                "person_count": 0, "gender_male": 0,
                "gender_female": 0, "age_gender_classes": {},
            }
            _analytics_last_sent[pad_index] = now

    return Gst.PadProbeReturn.OK
