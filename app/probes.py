"""
probes.py — NX Computing AI | Edge Inference Core
Probe de GStreamer para extracción de metadatos DeepStream y envío a API REST.

Arquitectura de inferencia:
  PGIE (PeopleNet, gie-id=1)  →  SGIE (Pedestrian Attr ResNet-18, gie-id=2)
  El PGIE detecta 3 clases: person(0), bag(1), face(2).
  El SGIE clasifica el recorte de CUERPO COMPLETO de cada person(0) del PGIE.
  En condiciones reales de CCTV los rostros rara vez son visibles; la clasificación
  por cuerpo aprovecha ropa, postura y complexión física.
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
CAMERA_ID: str  = os.environ.get("CAMERA_ID",  "cam-001")
JETSON_ID: str  = os.environ.get("JETSON_ID",  os.uname().nodename)
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY: str    = os.environ.get("API_KEY",    "your-api-key")

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
    def post_detection_event(self, frame_num: int, detections: List[dict],
                             event_type: str = "person_detection"):
        """
        Envía un evento de detección al backend.
        Construye el payload con la estructura esperada por el dashboard.
        event_type: "person_detection" (primera aparición) o
                    "person_classified" (clasificación demográfica confirmada).
        """
        payload = {
            "event_id": str(uuid.uuid4()),
            "camera_id": CAMERA_ID,
            "jetson_id": JETSON_ID,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "severity": "info",
            "status": "open",
            "detections": detections,
        }
        self.enqueue("POST", "/api/events", payload)

    def post_analytics_snapshot(self, stats: dict):
        """
        Envía un snapshot de analytics agregadas (edad/género, conteo).
        Alimenta el endpoint GET /api/analytics/age-gender del dashboard.
        """
        payload = {
            "camera_id": CAMERA_ID,
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

    def post_crop(self, track_id: int, frame_num: int, crop_b64: str, bbox: dict):
        """Envía un recorte de persona a la API para construcción de dataset."""
        payload = {
            "camera_id": CAMERA_ID,
            "jetson_id": JETSON_ID,
            "track_id": track_id,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_b64": crop_b64,
            "bbox": bbox,
        }
        self.enqueue("POST", "/api/crops", payload)

    def post_reference_frame(self, frame_num: int, frame_b64: str, width: int, height: int):
        """
        Envía un frame de referencia sin personas al backend.
        El dashboard lo usa como fondo para superponer el heatmap de posiciones.
        Se envía una sola vez por sesión (primer frame limpio disponible).
        """
        payload = {
            "camera_id": CAMERA_ID,
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
# 4. PROBE PRINCIPAL — OSD + ENVÍO A API
# ==============================================================================
# Resultado final de votación por track_id: (raw_label, gender_disp, age_disp, prob)
# prob = fracción de votos ganadores (ej. 7/10 → 0.70)
_person_cache: Dict[int, Tuple[str, str, str, float]] = {}

# Votos acumulados por track_id mientras no se alcanza VOTES_REQUIRED
_person_votes: Dict[int, List[str]] = {}

# Último frame_num en que se tomó muestra para cada track_id
_person_vote_last_frame: Dict[int, int] = {}

# IDs de personas ya notificadas a la API (evita spam de eventos por persona)
_person_notified: set = set()

# Crops capturados por track_id: cantidad guardada y último frame muestreado
_crop_counts: Dict[int, int] = {}
_crop_last_frame: Dict[int, int] = {}

# Frame de referencia (sin personas) para el heatmap del dashboard.
# Se envía una sola vez por sesión — el primer frame donde no hay detecciones.
_reference_frame_sent: bool = False

# Acumulador para analytics agregadas (se envía cada ANALYTICS_SEND_INTERVAL_SECS)
_analytics: Dict = {
    "person_count": 0,
    "gender_male": 0,
    "gender_female": 0,
    "age_gender_classes": {},   # e.g. {"male_adult": 12, "female_young": 3, ...}
}
_analytics_last_sent: float = time.monotonic()


def _save_and_send_crop(
    crop_bgr: np.ndarray,
    track_id: int,
    frame_num: int,
    bbox: dict,
) -> None:
    """Guarda el recorte localmente y lo envía a la API de forma no bloqueante."""
    person_dir = Path(CROPS_DIR) / str(track_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    filepath = person_dir / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filepath), crop_bgr)

    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    crop_b64 = base64.b64encode(buf).decode("utf-8")
    api_client.post_crop(track_id, frame_num, crop_b64, bbox)


def osd_sink_pad_buffer_probe(_pad, info):
    """
    Probe conectado al sink-pad del elemento nvdsosd.

    Arquitectura (2 modelos):
      PGIE (PeopleNet, gie-id=1) detecta person/bag/face en el frame completo.
      SGIE (Pedestrian Attr, gie-id=2) clasifica el recorte de cuerpo completo
      de cada person(0), inyectando NvDsClassifierMeta en el mismo objeto.

    Por cada persona:
      - Si el SGIE ya entregó resultado → muestra género/edad + confianza en OSD.
      - Si aún no hay resultado          → muestra "P#N | Analizando..." en cyan.
      - Si ya fue clasificada antes      → recupera de caché (no re-analiza).
    """
    global _analytics, _analytics_last_sent, _person_notified, _person_cache, _person_votes, _person_vote_last_frame, _crop_counts, _crop_last_frame, _reference_frame_sent

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
        frame_detections: List[dict] = []

        # Extraer frame como numpy (una sola vez por frame para todos los objetos)
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

            # Leer clasificación del SGIE en este frame
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

            # --- DIAGNÓSTICO TEMPORAL (quitar cuando funcione) ---
            if frame_num % 30 == 0:
                logger.info(
                    "DIAG P#%d frame=%d | cls_items=%d ids=%s | label=%r prob=%.3f",
                    p_track_id, frame_num, _cls_items, _seen_ids, raw_label, prob,
                )

            bbox_w = int(obj_meta.rect_params.width)
            bbox_h = int(obj_meta.rect_params.height)
            person_too_small = bbox_w < VOTE_MIN_WIDTH or bbox_h < VOTE_MIN_HEIGHT

            is_new_classification = False
            if p_track_id in _person_cache:
                # Votación completada — usar resultado fijo
                raw_label, gender_disp, age_disp, prob = _person_cache[p_track_id]
            elif person_too_small:
                # Persona demasiado pequeña — no clasificar
                raw_label, gender_disp, age_disp, prob = "", "", "", 0.0
            else:
                # Acumular voto si el SGIE entregó resultado aceptable y pasó el intervalo
                last_frame = _person_vote_last_frame.get(p_track_id, -VOTE_SAMPLE_INTERVAL)
                if (raw_label and prob >= MIN_CLASSIFICATION_PROB
                        and (frame_num - last_frame) >= VOTE_SAMPLE_INTERVAL):
                    votes = _person_votes.setdefault(p_track_id, [])
                    votes.append(raw_label)
                    _person_vote_last_frame[p_track_id] = frame_num

                    if len(votes) >= VOTES_REQUIRED:
                        winner = max(set(votes), key=votes.count)
                        vote_prob = votes.count(winner) / len(votes)
                        w_gender, w_age = _AGE_GENDER_LABEL_MAP[winner]
                        _person_cache[p_track_id] = (winner, w_gender, w_age, vote_prob)
                        raw_label, gender_disp, age_disp, prob = _person_cache[p_track_id]
                        is_new_classification = True
                        logger.debug(
                            "P#%d votación completa: %s (%.0f%% de %d votos)",
                            p_track_id, winner, vote_prob * 100, len(votes),
                        )
                    else:
                        raw_label, gender_disp, age_disp, prob = "", "", "", 0.0
                else:
                    raw_label, gender_disp, age_disp, prob = "", "", "", 0.0

            # OSD
            if raw_label:
                osd_text = f"P#{p_track_id} | {gender_disp} | {age_disp} {prob:.0%}"
                border   = (0.0, 1.0, 0.0, 1.0)    # Verde — clasificado
            elif person_too_small:
                osd_text = f"P#{p_track_id} | Muy lejos"
                border   = (0.8, 0.8, 0.0, 1.0)    # Amarillo — fuera de rango
            else:
                n_votes  = len(_person_votes.get(p_track_id, []))
                osd_text = f"P#{p_track_id} | Analizando... ({n_votes}/{VOTES_REQUIRED})"
                border   = (0.0, 0.8, 1.0, 1.0)    # Cyan — recolectando votos

            _set_osd_text(obj_meta, osd_text, border_color=border)

            # Capturar crop para dataset
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
                            crop, p_track_id, frame_num,
                            {"left": l, "top": t, "width": w, "height": h},
                        )
                        _crop_counts[p_track_id] = count + 1
                        _crop_last_frame[p_track_id] = frame_num

            # Notificar primera aparición de la persona
            if p_track_id not in _person_notified:
                _person_notified.add(p_track_id)
                det = _build_detection_dict(obj_meta)
                det["type"] = "person"
                frame_detections.append(det)
                _analytics["person_count"] += 1

            # Notificar primera clasificación confirmada
            if is_new_classification:
                det = _build_detection_dict(obj_meta)
                det["type"] = "person_classified"
                api_gender, api_age = _AGE_GENDER_API_MAP[raw_label]
                det["demographics"] = {
                    "gender":     api_gender,
                    "age_group":  api_age,
                    "label":      raw_label,
                    "confidence": round(prob, 3),
                }
                frame_detections.append(det)
                _analytics["age_gender_classes"][raw_label] = (
                    _analytics["age_gender_classes"].get(raw_label, 0) + 1
                )
                if raw_label.startswith("male"):
                    _analytics["gender_male"] += 1
                else:
                    _analytics["gender_female"] += 1

        # ----------------------------------------------------------------
        # Frame de referencia para heatmap — primer frame sin personas
        # ----------------------------------------------------------------
        if (not _reference_frame_sent
                and frame_np is not None
                and not frame_detections
                and frame_meta.num_obj_meta == 0):
            h, w = frame_np.shape[:2]
            _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_b64 = base64.b64encode(buf).decode("utf-8")
            api_client.post_reference_frame(frame_num, frame_b64, w, h)
            _reference_frame_sent = True
            logger.info("Frame de referencia enviado (frame=%d, %dx%d)", frame_num, w, h)

        # ----------------------------------------------------------------
        # Enviar evento solo cuando hay detección nueva o clasificación nueva
        # ----------------------------------------------------------------
        if frame_detections:
            has_classified = any(d.get("type") == "person_classified" for d in frame_detections)
            evt_type = "person_classified" if has_classified else "person_detection"
            api_client.post_detection_event(frame_num, frame_detections, evt_type)

        # ----------------------------------------------------------------
        # Enviar analytics agregadas cada ANALYTICS_SEND_INTERVAL_SECS
        # ----------------------------------------------------------------
        now = time.monotonic()
        if now - _analytics_last_sent >= ANALYTICS_SEND_INTERVAL_SECS:
            api_client.post_analytics_snapshot({
                "people_count":       _analytics["person_count"],
                "gender_male":        _analytics["gender_male"],
                "gender_female":      _analytics["gender_female"],
                "age_gender_classes": _analytics["age_gender_classes"],
            })
            _analytics = {
                "person_count": 0, "gender_male": 0,
                "gender_female": 0, "age_gender_classes": {},
            }
            _analytics_last_sent = now

    return Gst.PadProbeReturn.OK
