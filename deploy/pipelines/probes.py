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
import dataclasses
import json
import math
import queue
import logging
import threading
import time
import uuid
import base64
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

# Sector del cliente: "comercio" | "industrial" | "hogar"
# Controla el tipo de evento emitido por face_recognition y la severidad de fall_detected.
_JETSON_SECTOR: str = "comercio"

# Pad indices que corresponden a cámaras de entrada/salida
_entry_exit_pads: set = set()


def init_channel_map(channels: list):
    """Llamar desde app.py después de load_config(), antes de arrancar el pipeline."""
    global _channel_map
    _channel_map = {idx: ch for idx, ch in enumerate(channels)}
    logger.info("Channel map: %s", _channel_map)


def init_sector(sector: str) -> None:
    global _JETSON_SECTOR
    _JETSON_SECTOR = sector
    logger.info("Sector: %s", sector)


def init_entry_exit_pads(pad_indices: set) -> None:
    global _entry_exit_pads
    _entry_exit_pads = pad_indices
    logger.info("Entry/exit pad indices: %s", pad_indices)

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

# ── Track lifecycle ───────────────────────────────────────────────────────────
TRACK_LOST_TIMEOUT_FRAMES: int = 60  # frames without detection before declaring track lost

# ── Face recognition ──────────────────────────────────────────────────────────
FACE_SAMPLE_INTERVAL: int  = 30   # frames between recognition attempts per track

# ── Analytics ─────────────────────────────────────────────────────────────────
ANALYTICS_SEND_INTERVAL_SECS: float = 60.0

# ── Crop capture ──────────────────────────────────────────────────────────────
CROPS_DIR: str            = "crops"
CROP_SAMPLE_INTERVAL: int = 15
CROP_MAX_PER_PERSON: int  = 5
CROP_MIN_WIDTH: int       = 64
CROP_MIN_HEIGHT: int      = 128


# ==============================================================================
# QA VISUAL MODE — activo solo cuando NX_QA_ENABLED=true
# Cero impacto en producción cuando la variable de entorno no está seteada.
# ==============================================================================

_IS_QA_ENABLED: bool = os.getenv("NX_QA_ENABLED", "false").lower() == "true"

# Redis client para pub/sub efímero — solo si QA activo
_redis_qa = None
if _IS_QA_ENABLED:
    try:
        import redis as _redis_lib
        _redis_qa = _redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=6379,
            db=0,
            socket_connect_timeout=2,
            socket_timeout=1,
        )
        _redis_qa.ping()
        logger.info("[QA] Redis conectado — nx:qa:detections + nx:qa:apicalls activos")
    except Exception as _qa_e:
        logger.warning("[QA] Redis no disponible (%s) — metadata no se publicará", _qa_e)
        _redis_qa = None

# Queues para el MjpegServer (pobladas desde app.py via init_qa_cameras)
tiled_frame_queue: queue.Queue = queue.Queue(maxsize=1)
camera_frame_queues: dict = {}   # camera_id → Queue(maxsize=1)

# Grid del tiler — set via init_qa_grid() desde app.py
_qa_tiler_cols: int = 1
_qa_tiler_rows: int = 1
_qa_cell_w: int = 640
_qa_cell_h: int = 360


def init_qa_grid(tiler_cols: int, tiler_rows: int, cell_w: int, cell_h: int) -> None:
    """Llamar desde app.py después de calcular el grid del tiler."""
    global _qa_tiler_cols, _qa_tiler_rows, _qa_cell_w, _qa_cell_h
    _qa_tiler_cols = tiler_cols
    _qa_tiler_rows = tiler_rows
    _qa_cell_w = cell_w
    _qa_cell_h = cell_h
    if _IS_QA_ENABLED:
        logger.info("[QA] Grid: %dx%d tiles de %dx%d px", tiler_cols, tiler_rows, cell_w, cell_h)


def init_qa_cameras(channels: list) -> None:
    """Crea una Queue por cámara en camera_frame_queues. Llamar después de init_channel_map."""
    global camera_frame_queues
    camera_frame_queues = {
        _camera_id_for(i): queue.Queue(maxsize=1)
        for i in range(len(channels))
    }
    if _IS_QA_ENABLED:
        logger.info("[QA] Queues de cámara: %s", list(camera_frame_queues.keys()))


def _qa_publish(channel: str, data: dict) -> None:
    """Fire-and-forget Redis pub/sub. Silencioso si Redis no disponible."""
    if _redis_qa:
        try:
            _redis_qa.publish(channel, json.dumps(data, default=str))
        except Exception:
            pass


def _is_capability_active(cap: str) -> bool:
    """
    En QA: lee el toggle del hash Redis nx:qa:capabilities.
    En producción (QA desactivado): siempre retorna True.
    Default cuando la key no existe: True (activo).
    """
    if not _IS_QA_ENABLED or not _redis_qa:
        return True
    try:
        val = _redis_qa.hget("nx:qa:capabilities", cap)
        return val is None or val.decode() == "1"
    except Exception:
        return True


def _draw_qa_overlays(frame_bgr: np.ndarray, qa_tracks: list) -> None:
    """
    Dibuja bboxes y labels sobre frame_bgr in-place con OpenCV (CPU).
    Las coordenadas bbox ya están en el espacio del frame tileado (640×360).
    """
    for t in qa_tracks:
        left, top, w, h = t["bbox_tiled"]
        x1 = max(0, int(left))
        y1 = max(0, int(top))
        x2 = min(frame_bgr.shape[1] - 1, int(left + w))
        y2 = min(frame_bgr.shape[0] - 1, int(top + h))
        if x2 <= x1 or y2 <= y1:
            continue

        is_fall = t.get("fall", False)
        color = (0, 0, 230) if is_fall else (0, 210, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)

        label = t.get("label", f"#{t['track_id']}")
        cv2.putText(
            frame_bgr, label,
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )


# ==============================================================================
# 0. TRACK STATE — lifecycle per person per camera
# ==============================================================================

@dataclass
class _TrackState:
    first_frame:       int
    last_frame:        int
    first_ts:          float   # time.monotonic() when first seen
    camera_id:         str
    is_entry_exit_cam: bool = False
    appearance_sent:   bool = False


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
        # QA: intercept payload antes de enviar al backend
        if _IS_QA_ENABLED:
            _qa_publish("nx:qa:apicalls", {
                "endpoint": endpoint,
                "method": method,
                "payload": payload or {},
                "ts": datetime.now(timezone.utc).isoformat(),
            })
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
    # Helpers internos
    # ------------------------------------------------------------------
    def _base_event(self, event_type: str, camera_id: str, severity: str = "info") -> dict:
        return {
            "event_id":  str(uuid.uuid4()),
            "type":      event_type,
            "sector":    _JETSON_SECTOR,
            "jetson_id": JETSON_ID,
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity":  severity,
        }

    # ------------------------------------------------------------------
    # Métodos de negocio — un evento por entidad, estructura plana
    # ------------------------------------------------------------------

    def post_person_entry(self, camera_id: str, track_id: int, bbox: dict,
                          confidence: float, is_entry_exit_cam: bool) -> None:
        payload = self._base_event("person_entry", camera_id)
        payload.update({
            "track_id": track_id,
            "bbox": bbox,
            "confidence": round(confidence, 3),
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        self.enqueue("POST", "/api/events", payload)

    def post_person_exit(self, camera_id: str, track_id: int,
                         dwell_seconds: float, is_entry_exit_cam: bool) -> None:
        payload = self._base_event("person_exit", camera_id)
        payload.update({
            "track_id": track_id,
            "dwell_seconds": round(dwell_seconds, 1),
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        self.enqueue("POST", "/api/events", payload)

    def post_person_classified(self, camera_id: str, track_id: int,
                               bbox: dict, demographics: dict) -> None:
        payload = self._base_event("person_classified", camera_id)
        payload.update({"track_id": track_id, "bbox": bbox, "demographics": demographics})
        self.enqueue("POST", "/api/events", payload)

    def post_person_appearance(self, camera_id: str, track_id: int,
                               appearance_vector: list) -> None:
        payload = self._base_event("person_appearance", camera_id)
        payload.update({"track_id": track_id, "appearance_vector": appearance_vector})
        self.enqueue("POST", "/api/events", payload)

    def post_fall_detected(self, camera_id: str, track_id: int, bbox: dict,
                           fall_score: int, avg_kp_conf: float) -> None:
        severity = "critical" if _JETSON_SECTOR == "hogar" else "high"
        payload = self._base_event("fall_detected", camera_id, severity)
        payload.update({
            "track_id": track_id,
            "bbox": bbox,
            "fall_score": fall_score,
            "avg_kp_conf": round(avg_kp_conf, 3),
        })
        self.enqueue("POST", "/api/events", payload)

    def post_employee_seen(self, camera_id: str, employee_id: str, track_id: int,
                           similarity: float, bbox: dict) -> None:
        evt = "known_person_seen" if _JETSON_SECTOR == "hogar" else "employee_seen"
        payload = self._base_event(evt, camera_id)
        payload.update({
            "track_id": track_id,
            "bbox": bbox,
            "similarity": round(similarity, 3),
            "employee_id" if _JETSON_SECTOR != "hogar" else "name": employee_id,
        })
        self.enqueue("POST", "/api/events", payload)

    def post_employee_presence(self, camera_id: str, employee_id: str, track_id: int) -> None:
        payload = self._base_event("employee_presence", camera_id)
        payload.update({"track_id": track_id,
                        "employee_id" if _JETSON_SECTOR != "hogar" else "name": employee_id})
        self.enqueue("POST", "/api/events", payload)

    def post_employee_exit(self, camera_id: str, employee_id: str,
                           track_id: int, dwell_seconds: float) -> None:
        evt = "known_person_exit" if _JETSON_SECTOR == "hogar" else "employee_exit"
        payload = self._base_event(evt, camera_id)
        payload.update({
            "track_id": track_id,
            "dwell_seconds": round(dwell_seconds, 1),
            "employee_id" if _JETSON_SECTOR != "hogar" else "name": employee_id,
        })
        self.enqueue("POST", "/api/events", payload)

    def post_unknown_person_alert(self, camera_id: str, track_id: int,
                                  face_snapshot_b64: str, bbox: dict) -> None:
        payload = self._base_event("unknown_person_alert", camera_id, "medium")
        payload.update({"track_id": track_id, "bbox": bbox,
                        "face_snapshot_b64": face_snapshot_b64})
        self.enqueue("POST", "/api/events", payload)

    def post_epp_violation(self, camera_id: str, track_id: int, bbox: dict,
                           violations: list, present: list, confidence: float) -> None:
        payload = self._base_event("epp_violation", camera_id, "high")
        payload.update({"track_id": track_id, "bbox": bbox, "violations": violations,
                        "present": present, "confidence": round(confidence, 3)})
        self.enqueue("POST", "/api/events", payload)

    def post_fire_smoke_alert(self, camera_id: str, detected: list,
                              confidence: float, frame_snapshot_b64: str = "") -> None:
        payload = self._base_event("fire_smoke_alert", camera_id, "critical")
        payload.update({"detected": detected, "confidence": round(confidence, 3),
                        "frame_snapshot_b64": frame_snapshot_b64})
        self.enqueue("POST", "/api/events", payload)

    def post_vehicle_detected(self, camera_id: str, track_id: int, bbox: dict,
                              plate: str, plate_confidence: float) -> None:
        payload = self._base_event("vehicle_detected", camera_id)
        payload.update({"track_id": track_id, "bbox": bbox,
                        "plate": plate, "plate_confidence": round(plate_confidence, 3)})
        self.enqueue("POST", "/api/events", payload)

    def post_analytics_snapshot(self, camera_id: str, stats: dict,
                                period_seconds: float = 60.0) -> None:
        payload = self._base_event("analytics_snapshot", camera_id)
        payload.update({"period_seconds": period_seconds, **stats})
        self.enqueue("POST", "/api/analytics", payload)

    def post_crop(self, camera_id: str, track_id: int, frame_num: int,
                  crop_b64: str, bbox: dict) -> None:
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

    def post_reference_frame(self, camera_id: str, frame_num: int,
                             frame_b64: str, width: int, height: int) -> None:
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
    """
    EPP (Personal Protective Equipment) compliance detection.
    Model pending integration — stub returns None until model files are available.

    When model is ready, fill in process() to read SGIE output and call:
      api_client.post_epp_violation(camera_id, track_id, bbox,
          violations=["no_helmet"], present=["gloves"], confidence=0.87)
    Severity is always "high" (set in post_epp_violation).
    """
    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        return None  # TODO: implement when epp SGIE model is integrated


class _FireSmokeHandler:
    """
    Fire and smoke frame-level classifier.
    Model pending integration — stub returns None until model files are available.

    When model is ready, fill in process() to read SGIE output and call:
      api_client.post_fire_smoke_alert(camera_id,
          detected=["fire"], confidence=0.94, frame_snapshot_b64=frame_b64)
    Severity is always "critical" (set in post_fire_smoke_alert).
    """
    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        return None  # TODO: implement when fire_smoke SGIE model is integrated


class _LicensePlateHandler:
    """
    License plate detection and reading (LPD + LPR).
    Model pending integration — stub returns None until model files are available.

    When model is ready, fill in process() to read SGIE output and call:
      api_client.post_vehicle_detected(camera_id, track_id, bbox,
          plate="ABC-1234", plate_confidence=0.93)
    """
    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        return None  # TODO: implement when license_plate SGIE model is integrated


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

    Emits sector-aware API events:
      comercio/industrial → employee_seen / employee_presence / employee_exit
      hogar               → known_person_seen / known_person_exit / unknown_person_alert

    This handler is NOT in _HANDLER_REGISTRY — it is dispatched separately via
    _face_handler because it processes SGIE_FACE_DETECT_ID objects, not PGIE objects.
    """
    PRESENCE_HEARTBEAT_SECS: float = 30.0

    def __init__(self, worker):
        self._worker = worker
        self._last_sample: Dict[int, int] = {}
        self._cache: Dict[int, Tuple[str, float]] = {}
        # tracks where we already fired employee_seen / known_person_seen
        self._identity_reported: Set[int] = set()
        # last time we sent a presence heartbeat per track
        self._last_heartbeat: Dict[int, float] = {}
        # tracks we already alerted as unknown (hogar only)
        self._unknown_alerted: Set[int] = set()

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
        fl = max(0, int(r.left))
        ft = max(0, int(r.top))
        fw = max(1, int(r.width))
        fh = max(1, int(r.height))
        face_crop = frame_np[ft:ft + fh, fl:fl + fw]
        if face_crop.size == 0:
            return

        self._last_sample[parent_track_id] = frame_num
        self._worker.enqueue(face_crop, parent_track_id, frame_num, camera_id)

        result = self._worker.get_result(parent_track_id)
        if result:
            self._cache[parent_track_id] = result

        # Find parent bbox for event payloads
        parent_obj = next(
            (p for p in persons_meta if int(p.object_id) == parent_track_id), None
        )
        bbox: dict = {}
        if parent_obj:
            pr = parent_obj.rect_params
            bbox = {"left": max(0, int(pr.left)), "top": max(0, int(pr.top)),
                    "width": int(pr.width), "height": int(pr.height)}

        identity = self._cache.get(parent_track_id)
        if identity is None:
            return

        name, conf = identity
        now = time.monotonic()

        if name != "Desconocido":
            if parent_track_id not in self._identity_reported:
                self._identity_reported.add(parent_track_id)
                api_client.post_employee_seen(camera_id, name, parent_track_id, conf, bbox)
            # heartbeat
            last_hb = self._last_heartbeat.get(parent_track_id, 0.0)
            if now - last_hb >= self.PRESENCE_HEARTBEAT_SECS:
                api_client.post_employee_presence(camera_id, name, parent_track_id)
                self._last_heartbeat[parent_track_id] = now
        elif _JETSON_SECTOR == "hogar" and parent_track_id not in self._unknown_alerted:
            # Unknown person in hogar → alert with face snapshot
            self._unknown_alerted.add(parent_track_id)
            _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            face_b64 = base64.b64encode(buf).decode("utf-8")
            api_client.post_unknown_person_alert(camera_id, parent_track_id, face_b64, bbox)

    def on_track_lost(self, track_id: int, dwell_seconds: float) -> None:
        """Called from _expire_lost_tracks. Emits employee_exit / known_person_exit."""
        identity = self._cache.get(track_id)
        if identity and identity[0] != "Desconocido":
            name, _ = identity
            # We need camera_id — look it up from active track or use empty string (best effort)
            state = _active_tracks.get(
                next((k for k in _active_tracks if k[1] == track_id), (None, None))
            )
            camera_id = state.camera_id if state else ""
            api_client.post_employee_exit(camera_id, name, track_id, dwell_seconds)
        # clean state
        self._cache.pop(track_id, None)
        self._identity_reported.discard(track_id)
        self._last_heartbeat.pop(track_id, None)
        self._unknown_alerted.discard(track_id)

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

_pose_worker      = None   # PoseWorker (fall_detection)
_face_recognizer  = None   # FaceRecognizer (face_recognition)
_appearance_worker = None  # AppearanceWorker (cross-camera re-ID, always active)
_ws_client        = None   # WsPositionClient (WebSocket position telemetry)

# Position buffer: pad_index → list of {track_id, x_norm, y_norm}
_position_buffer: Dict[int, List[dict]] = {}
_position_last_sent: Dict[int, float]  = {}
POSITION_SEND_INTERVAL: float = 10.0   # seconds between position snapshots per camera


def init_workers(
    pipeline_capabilities: List[str],
    model_dir: str,
    face_db_path: str = "",
    ws_base_url: str = "",
    api_key: str = "",
) -> None:
    global _pose_worker, _face_recognizer, _appearance_worker, _ws_client

    # AppearanceWorker — always active (cross-camera re-ID, tied to people_counting)
    osnet_path = str(Path(model_dir) / "osnet" / "osnet_x0_25_market1501.onnx")
    if Path(osnet_path).exists():
        from appearance_worker import AppearanceWorker
        _appearance_worker = AppearanceWorker(osnet_path)
    else:
        logger.warning("OSNet model not found at %s — appearance vectors disabled. "
                       "Run: python3 tools/download_models.py --reid", osnet_path)

    if "fall_detection" in pipeline_capabilities:
        from pose_worker import PoseWorker
        model_path = str(Path(model_dir) / "movenet" / "movenet_singlepose_lightning_192.onnx")
        _pose_worker = PoseWorker(model_path)

    if "face_recognition" in pipeline_capabilities:
        from face_recognizer import FaceRecognizer
        _face_recognizer = FaceRecognizer(
            db_path=face_db_path,
            model_root=str(Path(model_dir) / "insightface"),
        )

    # WebSocket position client — active only if WS_BASE_URL is configured
    if ws_base_url:
        from ws_client import WsPositionClient
        _ws_client = WsPositionClient(
            ws_url=ws_base_url,
            api_key=api_key,
            sector=_JETSON_SECTOR,
        )
    else:
        logger.info("WS_BASE_URL not set — position WebSocket disabled.")


def start_workers() -> None:
    """Start all workers. Call after pipeline.set_state(PLAYING)."""
    if _appearance_worker is not None:
        _appearance_worker.start()
    if _pose_worker is not None:
        _pose_worker.start()
    if _face_recognizer is not None:
        _face_recognizer.start()
    if _ws_client is not None:
        _ws_client.start()


def stop_workers() -> None:
    if _appearance_worker is not None:
        _appearance_worker.stop()
    if _pose_worker is not None:
        _pose_worker.stop()
    if _face_recognizer is not None:
        _face_recognizer.stop()
    if _ws_client is not None:
        _ws_client.stop()


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
            handler._cap_name = cap  # usado por _is_capability_active en QA mode
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

# Active tracks per (pad_index, track_id)
_active_tracks: Dict[Tuple[int, int], _TrackState] = {}

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


def _accumulate_positions(
    pad_index: int, camera_id: str, persons_meta: list,
    frame_width: int, frame_height: int, frame_num: int,
) -> None:
    """Accumulate normalized centroids and send position snapshot every POSITION_SEND_INTERVAL s."""
    buf = _position_buffer.setdefault(pad_index, [])
    for obj in persons_meta:
        r = obj.rect_params
        x_norm = round((r.left + r.width / 2) / frame_width, 3)
        y_norm = round((r.top + r.height / 2) / frame_height, 3)
        buf.append({"track_id": int(obj.object_id), "x_norm": x_norm, "y_norm": y_norm})

    now = time.monotonic()
    last = _position_last_sent.get(pad_index, 0.0)
    if now - last >= POSITION_SEND_INTERVAL and buf:
        _ws_client.send_positions(camera_id, buf)
        _position_buffer[pad_index] = []
        _position_last_sent[pad_index] = now


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


def _expire_lost_tracks(pad_index: int, frame_num: int,
                        visible_ids: Set[int]) -> None:
    """Emit person_exit for tracks not seen for TRACK_LOST_TIMEOUT_FRAMES frames."""
    expired = [
        key for key, state in _active_tracks.items()
        if key[0] == pad_index
        and key[1] not in visible_ids
        and (frame_num - state.last_frame) >= TRACK_LOST_TIMEOUT_FRAMES
    ]
    for key in expired:
        state = _active_tracks.pop(key)
        track_id = key[1]
        dwell = time.monotonic() - state.first_ts
        api_client.post_person_exit(
            state.camera_id, track_id, dwell, state.is_entry_exit_cam
        )
        if _face_handler:
            _face_handler.on_track_lost(track_id, dwell)
        # clean per-track caches
        _crop_counts.pop(track_id, None)
        _crop_last_frame.pop(track_id, None)
        for handler in _active_handlers:
            _cleanup_handler_cache(handler, track_id)
        logger.debug("Track lost: pad=%d track=%d dwell=%.1fs", pad_index, track_id, dwell)


def _cleanup_handler_cache(handler, track_id: int) -> None:
    """Remove track_id from any caches the handler holds."""
    for attr in ("_cache", "_votes", "_vote_last_frame", "_last_sample"):
        d = getattr(handler, attr, None)
        if isinstance(d, dict):
            d.pop(track_id, None)


def osd_sink_pad_buffer_probe(_pad, info):
    """
    Probe conectado al src-pad de caps_rgba (RGBA, post-tiler).
    Itera sobre objetos del frame: personas del PGIE y caras del FaceDetectIR SGIE.
    Despacha handlers activos y envía eventos / analytics a la API REST.
    En QA mode (NX_QA_ENABLED=true): dibuja overlays y publica a Redis/MJPEG.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # QA: acumula datos de todas las cámaras en este buffer (reset por llamada)
    _qa_frame_bgr: Optional[np.ndarray] = None   # frame tileado BGR, capturado 1 vez
    _qa_all_tracks: List[dict] = []

    for frame_meta in _iter_pyds_list(
        batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast
    ):
        frame_num = frame_meta.frame_num
        pad_index = frame_meta.pad_index
        camera_id = _camera_id_for(pad_index)
        is_entry_exit_cam = pad_index in _entry_exit_pads

        frame_np = None
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_np = np.array(n_frame, copy=True, order='C')
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            if frame_num % 30 == 0:
                logger.warning("get_nvds_buf_surface falló frame=%d: %s", frame_num, e)

        # QA: guardar el primer frame válido como referencia del tiled frame
        if _IS_QA_ENABLED and frame_np is not None and _qa_frame_bgr is None:
            _qa_frame_bgr = frame_np.copy()

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
            elif (uid == PGIE_UNIQUE_ID
                    and int(obj_meta.class_id) == PGIE_CLASS_FACE):
                face_metas.append(obj_meta)

        # ── Face recognition: process PeopleNet face detections (class 2) ────
        if _face_handler and face_metas and frame_np is not None:
            if _is_capability_active("face_recognition"):
                for face_obj_meta in face_metas:
                    _face_handler.process_face(
                        face_obj_meta, frame_num, frame_np, persons_meta, camera_id
                    )

        # ── Second pass: process each person ─────────────────────────────────
        visible_ids: Set[int] = set()
        for obj_meta in persons_meta:
            p_track_id = int(obj_meta.object_id)
            visible_ids.add(p_track_id)
            r = obj_meta.rect_params
            bbox = {
                "left":   max(0, int(r.left)),
                "top":    max(0, int(r.top)),
                "width":  int(r.width),
                "height": int(r.height),
            }

            track_key = (pad_index, p_track_id)
            now = time.monotonic()

            # ── Track entry ───────────────────────────────────────────────────
            if track_key not in _active_tracks:
                _active_tracks[track_key] = _TrackState(
                    first_frame=frame_num,
                    last_frame=frame_num,
                    first_ts=now,
                    camera_id=camera_id,
                    is_entry_exit_cam=is_entry_exit_cam,
                )
                api_client.post_person_entry(
                    camera_id, p_track_id, bbox,
                    float(obj_meta.confidence), is_entry_exit_cam,
                )
                _get_analytics(pad_index)["person_count"] += 1
            else:
                _active_tracks[track_key].last_frame = frame_num

            # ── AppearanceWorker: send appearance vector when ready ────────────
            if _appearance_worker is not None:
                state = _active_tracks[track_key]
                if not state.appearance_sent:
                    vec = _appearance_worker.get_result(p_track_id)
                    if vec is not None:
                        api_client.post_person_appearance(camera_id, p_track_id, vec.tolist())
                        state.appearance_sent = True
                # Enqueue crop for appearance extraction every 15 frames
                if (frame_np is not None
                        and frame_num % 15 == 0
                        and not state.appearance_sent
                        and bbox["width"] >= CROP_MIN_WIDTH
                        and bbox["height"] >= CROP_MIN_HEIGHT):
                    crop = frame_np[
                        bbox["top"]:bbox["top"] + bbox["height"],
                        bbox["left"]:bbox["left"] + bbox["width"],
                    ]
                    if crop.size > 0:
                        _appearance_worker.enqueue(crop, p_track_id, frame_num)

            _set_osd_text(obj_meta, f"P#{p_track_id}", border_color=(0.2, 0.6, 1.0, 1.0))

            # ── Active handlers ───────────────────────────────────────────────
            _qa_fall = False
            for handler in _active_handlers:
                # QA capability toggle: skip handler si está apagado en la UI
                if not _is_capability_active(getattr(handler, "_cap_name", "")):
                    continue
                result = handler.process(obj_meta, frame_num, frame_np=frame_np)
                if result is None:
                    continue
                if result.osd_text:
                    _set_osd_text(obj_meta, result.osd_text, border_color=result.border_color)
                if result.event_type == "person_classified":
                    api_client.post_person_classified(
                        camera_id, p_track_id, bbox, result.det_extra.get("demographics", {})
                    )
                    an = _get_analytics(pad_index)
                    au = result.analytics_update
                    if "age_gender_classes" in au:
                        lbl = au["age_gender_classes"]
                        an["age_gender_classes"][lbl] = an["age_gender_classes"].get(lbl, 0) + 1
                    if "gender_key" in au:
                        an[au["gender_key"]] += 1
                elif result.event_type == "fall_detected":
                    _qa_fall = True
                    api_client.post_fall_detected(
                        camera_id, p_track_id, bbox,
                        result.det_extra.get("fall_score", 0),
                        result.det_extra.get("avg_kp_conf", 0.0),
                    )

            # ── Face identity OSD overlay ─────────────────────────────────────
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

            # ── QA: registrar track con label final (después de todos los handlers) ──
            if _IS_QA_ENABLED:
                _qa_all_tracks.append({
                    "pad_index": pad_index,
                    "channel_id": camera_id,
                    "track_id": p_track_id,
                    "confidence": round(float(obj_meta.confidence), 3),
                    "bbox_tiled": (bbox["left"], bbox["top"], bbox["width"], bbox["height"]),
                    "label": obj_meta.text_params.display_text or f"P#{p_track_id}",
                    "fall": _qa_fall,
                })

            # ── Crop capture for dataset (always active) ──────────────────────
            if frame_np is not None:
                last_crop = _crop_last_frame.get(p_track_id, -CROP_SAMPLE_INTERVAL)
                count = _crop_counts.get(p_track_id, 0)
                if (count < CROP_MAX_PER_PERSON
                        and (frame_num - last_crop) >= CROP_SAMPLE_INTERVAL
                        and bbox["height"] >= CROP_MIN_HEIGHT
                        and bbox["width"] >= CROP_MIN_WIDTH):
                    crop = frame_np[
                        bbox["top"]:bbox["top"] + bbox["height"],
                        bbox["left"]:bbox["left"] + bbox["width"],
                    ]
                    if crop.size > 0:
                        _save_and_send_crop(
                            crop, camera_id, p_track_id, frame_num, bbox,
                        )
                        _crop_counts[p_track_id] = count + 1
                        _crop_last_frame[p_track_id] = frame_num

        # ── Expire lost tracks ────────────────────────────────────────────────
        _expire_lost_tracks(pad_index, frame_num, visible_ids)

        # ── Reference frame for heatmap ───────────────────────────────────────
        if (not _reference_frame_sent.get(pad_index, False)
                and frame_np is not None
                and len(visible_ids) == 0
                and frame_meta.num_obj_meta == 0):
            h, w = frame_np.shape[:2]
            _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_b64 = base64.b64encode(buf).decode("utf-8")
            api_client.post_reference_frame(camera_id, frame_num, frame_b64, w, h)
            _reference_frame_sent[pad_index] = True
            logger.info("Frame de referencia enviado camera=%s (frame=%d, %dx%d)",
                        camera_id, frame_num, w, h)

        # ── Position snapshot (WebSocket) ─────────────────────────────────────
        if _ws_client is not None and visible_ids and frame_np is not None:
            fh, fw = frame_np.shape[:2]
            _accumulate_positions(pad_index, camera_id, persons_meta, fw, fh, frame_num)

        # ── Analytics snapshot ────────────────────────────────────────────────
        now = time.monotonic()
        if now - _get_analytics_last_sent(pad_index) >= ANALYTICS_SEND_INTERVAL_SECS:
            an = _get_analytics(pad_index)
            api_client.post_analytics_snapshot(camera_id, {
                "people_count":       an["person_count"],
                "gender_male":        an["gender_male"],
                "gender_female":      an["gender_female"],
                "age_gender_classes": an["age_gender_classes"],
            }, period_seconds=ANALYTICS_SEND_INTERVAL_SECS)
            _analytics[pad_index] = {
                "person_count": 0, "gender_male": 0,
                "gender_female": 0, "age_gender_classes": {},
            }
            _analytics_last_sent[pad_index] = now

    # ── QA: overlays + MJPEG frames + Redis detections ───────────────────────
    if _IS_QA_ENABLED and _qa_frame_bgr is not None:
        # Dibujar bboxes y labels sobre el frame tileado
        _draw_qa_overlays(_qa_frame_bgr, _qa_all_tracks)

        # Tiled frame completo → /stream/all
        try:
            tiled_frame_queue.put_nowait(_qa_frame_bgr.copy())
        except queue.Full:
            pass

        # Crops por cámara → /stream/<camera_id> (todas, con o sin detecciones)
        for pad_idx, _ch_num in _channel_map.items():
            cam_id = _camera_id_for(pad_idx)
            q = camera_frame_queues.get(cam_id)
            if q is None:
                continue
            ox = (pad_idx % _qa_tiler_cols) * _qa_cell_w
            oy = (pad_idx // _qa_tiler_cols) * _qa_cell_h
            if (oy + _qa_cell_h <= _qa_frame_bgr.shape[0]
                    and ox + _qa_cell_w <= _qa_frame_bgr.shape[1]):
                crop = _qa_frame_bgr[oy:oy + _qa_cell_h, ox:ox + _qa_cell_w].copy()
                try:
                    q.put_nowait(crop)
                except queue.Full:
                    pass

        # Publicar detecciones a Redis (agrupadas por cámara)
        if _qa_all_tracks:
            by_cam: Dict[str, List] = {}
            for t in _qa_all_tracks:
                by_cam.setdefault(t["channel_id"], []).append({
                    "track_id": t["track_id"],
                    "confidence": t["confidence"],
                    "label": t["label"],
                    "fall": t["fall"],
                })
            for cam_id, tracks in by_cam.items():
                _qa_publish("nx:qa:detections", {
                    "cam": cam_id,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tracks": tracks,
                })

    return Gst.PadProbeReturn.OK
