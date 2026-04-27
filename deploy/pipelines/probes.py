"""
probes.py — NX Computing AI | Edge Inference Core
Probe de GStreamer para extracción de metadatos DeepStream y envío a API REST.

Arquitectura:
  PGIE (PeopleNet, gie-id=1) detecta person/bag/face en el frame completo.
  Handlers opcionales (uno por capability activa) procesan cada persona detectada.
  El pipeline de handlers es registrado en init_handlers() antes de arrancar.

Para agregar un nuevo modelo:
  1. Crear una clase que herede de _BaseHandler e implemente process().
  2. Añadirla a init_handlers() bajo su capability name.
  3. Añadir la entrada en SGIE_CONFIGS en app.py.
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

# gie-unique-id de cada modelo (deben coincidir con los configs de nvinfer)
PGIE_UNIQUE_ID: int = 1       # PeopleNet
SGIE_AGE_GENDER_ID: int = 2   # ResNet-18 Pedestrian Attr — opera sobre person(0) del PGIE

# Clases del PGIE (PeopleNet)
PGIE_CLASS_PERSON: int = 0
PGIE_CLASS_BAG: int = 1
PGIE_CLASS_FACE: int = 2

# Umbral mínimo de confianza para pintar y reportar detecciones del PGIE
OSD_CONFIDENCE_THRESHOLD: float = 0.40

# Confianza mínima para que un voto individual del SGIE sea aceptado.
MIN_CLASSIFICATION_PROB: float = 0.3

# Sistema de votación: se muestrea 1 frame cada VOTE_SAMPLE_INTERVAL frames
# por persona. Al acumular VOTES_REQUIRED votos se fija la clasificación final.
# Con intervalo=5 y 10 votos, una persona visible ~1.7s a 30fps queda clasificada.
VOTE_SAMPLE_INTERVAL: int = 5
VOTES_REQUIRED: int = 10

# Intervalo en segundos para enviar analytics de sistema (conteos agregados)
ANALYTICS_SEND_INTERVAL_SECS: float = 60.0

# Tamaño mínimo del bounding box para clasificar age/gender con el SGIE.
# Personas más pequeñas están demasiado lejos — su crop tiene poca calidad.
VOTE_MIN_WIDTH: int = 64
VOTE_MIN_HEIGHT: int = 160

# Captura de crops para dataset
CROPS_DIR: str = "crops"                # carpeta local donde se guardan los recortes
CROP_SAMPLE_INTERVAL: int = 15          # capturar 1 crop cada N frames por persona
CROP_MAX_PER_PERSON: int = 5            # máximo de crops a guardar por persona
CROP_MIN_WIDTH: int = 40                # descartar recortes más pequeños que esto
CROP_MIN_HEIGHT: int = 80


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

    def process(self, obj_meta, frame_num: int) -> Optional[_HandlerResult]:
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
    """Stub: PPE/EPP compliance detection. Model not yet integrated."""
    def process(self, obj_meta, frame_num: int) -> Optional[_HandlerResult]:
        return None


class _FireSmokeHandler:
    """Stub: fire and smoke frame-level classifier. Model not yet integrated."""
    def process(self, obj_meta, frame_num: int) -> Optional[_HandlerResult]:
        return None


class _LicensePlateHandler:
    """Stub: LPD + LPR vehicle license plate reader. Model not yet integrated."""
    def process(self, obj_meta, frame_num: int) -> Optional[_HandlerResult]:
        return None


class _FallDetectionHandler:
    """Stub: pose-based fall event detection. Model not yet integrated."""
    def process(self, obj_meta, frame_num: int) -> Optional[_HandlerResult]:
        return None


# Active handler instances — populated by init_handlers() before pipeline starts.
_active_handlers: List = []

_HANDLER_REGISTRY = {
    "age_gender":    _AgeGenderHandler,
    "epp_detection": _EppHandler,
    "fire_smoke":    _FireSmokeHandler,
    "license_plate": _LicensePlateHandler,
    "fall_detection":_FallDetectionHandler,
}


def init_handlers(pipeline_capabilities: List[str]) -> None:
    """Called from app.py after load_config() to register active handlers."""
    global _active_handlers
    _active_handlers = []
    for cap in pipeline_capabilities:
        cls = _HANDLER_REGISTRY.get(cap)
        if cls:
            _active_handlers.append(cls())
    names = [type(h).__name__ for h in _active_handlers]
    logger.info("Active handlers: %s", names if names else ["(none — people_counting only)"])


# ==============================================================================
# 5. PROBE PRINCIPAL — OSD + ENVÍO A API
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
    Itera sobre cada objeto persona del PGIE, despacha a los handlers activos
    y envía eventos / analytics a la API REST.
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

        for obj_meta in _iter_pyds_list(
            frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast
        ):
            if (obj_meta.unique_component_id != PGIE_UNIQUE_ID
                    or obj_meta.confidence < OSD_CONFIDENCE_THRESHOLD
                    or int(obj_meta.class_id) != PGIE_CLASS_PERSON):
                continue

            p_track_id = int(obj_meta.object_id)

            # Default OSD — overridden by handlers below if active
            _set_osd_text(obj_meta, f"P#{p_track_id}", border_color=(0.2, 0.6, 1.0, 1.0))

            # Dispatch to each registered handler
            for handler in _active_handlers:
                result = handler.process(obj_meta, frame_num)
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

            # Crop capture for dataset (always, regardless of handlers)
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
