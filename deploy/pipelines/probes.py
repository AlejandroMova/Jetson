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
import json
import queue
import logging
import threading
import time
import uuid
import base64
import os
from datetime import datetime, timezone
from dataclasses import dataclass
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
_entry_exit_probe_count: int = 0
_ENTRY_EXIT_REFRESH_EVERY: int = 30   # actualizar desde Redis cada N llamadas al probe

# Tipo de cámara: pad indices externos y flags de conteo por tipo
_external_pads:             set  = set()
_count_internal:            bool = True
_count_external:            bool = True
_camera_type_probe_count:   int  = 0
_CAMERA_TYPE_REFRESH_EVERY: int  = 30

# QA: probe A escribe, probe B lee — track_id → {face_name, fall, age_gender}
# Solo se popula cuando _IS_QA_ENABLED=True. GIL garantiza acceso seguro (hilo único GStreamer).
_track_labels: Dict[int, dict] = {}

# RecordingManager: instanciado desde app.py cuando NX_QA_ENABLED=true.
# Probe A le pasa frames full-res por cámara para grabación de video.
_recording_manager = None


def set_recording_manager(rm) -> None:
    """Registra la instancia de RecordingManager para que los probes puedan pasarle frames.

    Se llama desde app.py después de instanciar RecordingManager, antes de arrancar el pipeline.
    """
    global _recording_manager
    _recording_manager = rm



def init_channel_map(channels: list):
    """Llamar desde app.py después de load_config(), antes de arrancar el pipeline."""
    global _channel_map
    _channel_map = {idx: ch for idx, ch in enumerate(channels)}
    logger.info("Channel map: %s", _channel_map)


def init_sector(sector: str) -> None:
    """Configura el sector del cliente: 'comercio', 'industrial' o 'hogar'.

    El sector afecta el tipo de evento emitido por face_recognition
    (employee_seen vs known_person_seen) y la severidad de fall_detected.
    """
    global _JETSON_SECTOR
    _JETSON_SECTOR = sector
    logger.info("Sector: %s", sector)


def init_entry_exit_pads(pad_indices: set) -> None:
    """Define qué pad indices corresponden a cámaras de entrada/salida del local.

    Los eventos person_entry/exit de estas cámaras llevan is_entry_exit_camera=True,
    lo que le permite al backend calcular tráfico de entrada y salida del negocio.
    """
    global _entry_exit_pads
    _entry_exit_pads = pad_indices
    logger.info("Entry/exit pad indices: %s", pad_indices)


def init_camera_types(external_pad_indices: set, count_internal: bool, count_external: bool) -> None:
    """Configura qué cámaras son externas y si se deben contar sus personas.

    external_pad_indices: pads de cámaras externas (ej. estacionamiento, calle).
    count_internal: si False, las cámaras internas no generan eventos ni analytics.
    count_external: si False, las cámaras externas no generan eventos ni analytics.
    """
    global _external_pads, _count_internal, _count_external
    _external_pads  = external_pad_indices
    _count_internal = count_internal
    _count_external = count_external
    logger.info(
        "Camera types — external pads: %s  count_internal=%s  count_external=%s",
        external_pad_indices, count_internal, count_external,
    )


def _refresh_entry_exit_from_redis() -> None:
    """Lee nx:qa:entry_exit de Redis y actualiza _entry_exit_pads en caliente (QA mode)."""
    global _entry_exit_pads
    if not _redis_qa:
        return
    try:
        raw = _redis_qa.get("nx:qa:entry_exit")
        if raw is None:
            return
        ee_channels = json.loads(raw)
        ch_to_pad = {ch: idx for idx, ch in _channel_map.items()}
        _entry_exit_pads = {ch_to_pad[ch] for ch in ee_channels if ch in ch_to_pad}
    except Exception:
        pass


def _refresh_camera_types_from_redis() -> None:
    """Lee external_channels y flags de conteo de Redis en caliente (QA mode)."""
    global _external_pads, _count_internal, _count_external
    if not _redis_qa:
        return
    try:
        raw = _redis_qa.get("nx:qa:external_channels")
        if raw is not None:
            ext_channels = json.loads(raw)
            ch_to_pad = {ch: idx for idx, ch in _channel_map.items()}
            _external_pads = {ch_to_pad[ch] for ch in ext_channels if ch in ch_to_pad}
        ci = _redis_qa.get("nx:qa:count_internal")
        if ci is not None:
            _count_internal = (ci if isinstance(ci, str) else ci.decode()) == "1"
        ce = _redis_qa.get("nx:qa:count_external")
        if ce is not None:
            _count_external = (ce if isinstance(ce, str) else ce.decode()) == "1"
    except Exception:
        pass


def _camera_id_for(pad_index: int) -> str:
    """Devuelve un camera_id con el canal real, ej. 'jetson-nx-001-ch03'."""
    ch = _channel_map.get(pad_index, pad_index)
    return f"{JETSON_ID}-ch{ch:02d}"

# ── GIE unique-ids ────────────────────────────────────────────────────────────
PGIE_UNIQUE_ID: int      = 1   # PeopleNet
SGIE_AGE_GENDER_ID: int  = 2   # ResNet-18 Pedestrian Attr

# ── PGIE class ids ────────────────────────────────────────────────────────────
PGIE_CLASS_PERSON: int = 0
PGIE_CLASS_BAG:    int = 1
PGIE_CLASS_FACE:   int = 2

# ── Confidence thresholds ─────────────────────────────────────────────────────
OSD_CONFIDENCE_THRESHOLD: float      = 0.30
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
CROP_MIN_WIDTH: int       = 48
CROP_MIN_HEIGHT: int      = 96
# Frames to wait for an appearance embedding before emitting person_entry anyway.
# At 30 fps: first enqueue on frame 0, result typically arrives by frame ~2.
# 30 frames ≈ 1 second — enough for OSNet on CPU even under queue pressure.
# (Was 90, which delayed person_entry 3 s on slow crops — too visible in QA app.)
ENTRY_EMIT_DEADLINE_FRAMES: int = 30


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
    # Mutar el dict in-place para que los importadores del objeto original
    # (app.py, MjpegServer) vean las queues correctas sin necesidad de reimportar.
    camera_frame_queues.clear()
    for i in range(len(channels)):
        camera_frame_queues[_camera_id_for(i)] = queue.Queue(maxsize=1)
    if _IS_QA_ENABLED:
        logger.info("[QA] Queues de cámara: %s", list(camera_frame_queues.keys()))


def init_pipeline_stats(channels: list) -> None:
    """Inicializa nx:qa:pipeline_stats en Redis con FPS en 0. Llamar desde app.py."""
    if not _IS_QA_ENABLED or not _redis_qa:
        return
    stats = {
        "fps_per_camera": {_camera_id_for(i): 0.0 for i in range(len(channels))},
        "fps_total": 0.0,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _redis_qa.set("nx:qa:pipeline_stats", json.dumps(stats))
        logger.info("[QA] nx:qa:pipeline_stats inicializado")
    except Exception:
        pass


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
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label = t.get("label", f"#{t['track_id']}")
        txt_y = max(y1 - 3, 10)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame_bgr, (x1, txt_y - th - 2), (x1 + tw, txt_y + 1), color, -1)
        cv2.putText(
            frame_bgr, label,
            (x1, txt_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA,
        )


# ==============================================================================
# 0. TRACK STATE — lifecycle per person per camera
# ==============================================================================

@dataclass
class _TrackState:
    """Estado por track activo. Vive en _active_tracks[track_id] desde el primer frame hasta el exit.

    Los campos de ReID (entry_emitted, entry_deadline, global_id, pending_bbox, pending_conf)
    solo se usan cuando _reid_manager está activo (modelo OSNet encontrado).
    entry_deadline permite emitir un person_entry de fallback si el embedding tarda demasiado
    (APPEARANCE_ENTRY_DEADLINE_FRAMES frames sin recibir la primera embedding).
    """
    first_frame:       int
    last_frame:        int
    first_ts:          float        # time.monotonic() cuando fue visto por primera vez
    camera_id:         str
    is_entry_exit_cam: bool         = False
    appearance_sent:   bool         = False
    # Campos de ReID — solo se usan cuando _reid_manager está activo
    entry_emitted:     bool         = False   # True una vez que se emitió person_entry o channel_change
    entry_deadline:    int          = 0       # emitir entry de fallback si frame_num llega aquí sin embedding
    global_id:         Optional[str] = None   # asignado tras el match de ReID
    pending_bbox:      Optional[dict] = None  # bbox guardado en la primera detección para emit diferido
    pending_conf:      float         = 0.0    # confianza guardada en la primera detección


# ==============================================================================
# 1. BUS CALL — manejo de mensajes del pipeline
# ==============================================================================
def bus_call(_bus, message, loop):
    """Maneja mensajes del bus GStreamer: EOS para salida limpia, WARNING y ERROR para logging.

    Retorna True para que GStreamer siga llamando al handler en mensajes futuros.
    Se conecta al bus en app.py con: bus.add_watch(0, bus_call, loop).
    """
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
        """Configura el cliente. La sesión HTTP se crea aquí pero el hilo worker se arranca en start().

        Se usa requests.Session para reusar la conexión TCP entre requests (keep-alive),
        reduciendo latencia en deployments con muchos eventos simultáneos.
        max_queue_size=512 evita acumulación indefinida si el backend está lento o caído.
        """
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
        """Arranca el hilo worker que drena la cola de peticiones HTTP."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="nx-api-worker",
        )
        self._worker_thread.start()
        logger.info("NxApiClient iniciado → %s", self._base_url)

    def stop(self):
        """Señaliza al worker que pare, espera hasta 5 s y cierra la sesión HTTP."""
        self._running = False
        self._queue.put(None)  # sentinel para desbloquear el get() en _worker_loop
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._session.close()
        logger.info("NxApiClient detenido.")

    # ------------------------------------------------------------------
    # Encolar peticiones (llamado desde el probe, sin bloquear)
    # ------------------------------------------------------------------
    def enqueue(self, method: str, endpoint: str, payload: Optional[dict] = None):
        """Encola una petición HTTP sin bloquear. Descarta con warning si la cola está llena."""
        try:
            self._queue.put_nowait((method, endpoint, payload))
        except queue.Full:
            logger.warning("Cola API llena — descartando: %s %s", method, endpoint)

    # ------------------------------------------------------------------
    # Worker interno
    # ------------------------------------------------------------------
    def _worker_loop(self):
        """Loop principal del hilo worker: consume la cola y envía peticiones HTTP al backend."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break  # sentinel enviado por stop()

            method, endpoint, payload = item
            self._send(method, endpoint, payload)
            self._queue.task_done()

    def _send(self, method: str, endpoint: str, payload: Optional[dict]):
        """Envía la petición HTTP al backend. En QA mode publica el payload a Redis antes de enviar.

        Timeout de 5 s por request. Los errores se loguean pero no se propagan
        — el pipeline nunca debe fallar por un error de red al backend.
        """
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
        """Construye los campos comunes a todos los eventos del backend (id, tipo, sector, timestamp)."""
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
                          confidence: float, is_entry_exit_cam: bool,
                          global_id: Optional[str] = None,
                          is_return: bool = False) -> None:
        """Emite person_entry. entry_type="return" si la persona ya fue vista antes; "new" si es nueva."""
        payload = self._base_event("person_entry", camera_id)
        payload.update({
            "track_id":            track_id,
            "bbox":                bbox,
            "confidence":          round(confidence, 3),
            "is_entry_exit_camera": is_entry_exit_cam,
            "entry_type":          "return" if is_return else "new",
        })
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_channel_change(self, camera_id: str, track_id: int, bbox: dict,
                                   confidence: float, global_id: str,
                                   prev_camera_id: Optional[str],
                                   is_entry_exit_cam: bool) -> None:
        """Emite person_channel_change cuando la misma persona (global_id conocido) cambia de cámara."""
        payload = self._base_event("person_channel_change", camera_id)
        payload.update({
            "track_id":            track_id,
            "bbox":                bbox,
            "confidence":          round(confidence, 3),
            "global_id":           global_id,
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        if prev_camera_id:
            payload["prev_camera_id"] = prev_camera_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_exit(self, camera_id: str, track_id: int,
                         dwell_seconds: float, is_entry_exit_cam: bool,
                         global_id: Optional[str] = None) -> None:
        """Emite person_exit con el tiempo total de permanencia del track en la cámara."""
        payload = self._base_event("person_exit", camera_id)
        payload.update({
            "track_id":            track_id,
            "dwell_seconds":       round(dwell_seconds, 1),
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_classified(self, camera_id: str, track_id: int,
                               bbox: dict, demographics: dict) -> None:
        """Emite person_classified con el resultado del SGIE de edad/género (tras FACE_VOTES_REQUIRED votos)."""
        payload = self._base_event("person_classified", camera_id)
        payload.update({"track_id": track_id, "bbox": bbox, "demographics": demographics})
        self.enqueue("POST", "/api/events", payload)

    def post_person_appearance(self, camera_id: str, track_id: int,
                               appearance_vector: list) -> None:
        """Emite person_appearance con el vector OSNet 512-dim L2-normalizado para re-ID en el backend."""
        payload = self._base_event("person_appearance", camera_id)
        payload.update({"track_id": track_id, "appearance_vector": appearance_vector})
        self.enqueue("POST", "/api/events", payload)

    def post_fall_detected(self, camera_id: str, track_id: int, bbox: dict,
                           fall_score: int, avg_kp_conf: float) -> None:
        """Emite fall_detected. severity="critical" en sector hogar, "high" en otros."""
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
        """Emite employee_seen (o known_person_seen en hogar) cuando se identifica un rostro conocido."""
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
        """Emite employee_presence periódicamente para empleados que siguen en cámara (heartbeat)."""
        payload = self._base_event("employee_presence", camera_id)
        payload.update({"track_id": track_id,
                        "employee_id" if _JETSON_SECTOR != "hogar" else "name": employee_id})
        self.enqueue("POST", "/api/events", payload)

    def post_employee_exit(self, camera_id: str, employee_id: str,
                           track_id: int, dwell_seconds: float) -> None:
        """Emite employee_exit (o known_person_exit en hogar) con tiempo de permanencia del empleado."""
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
        """Emite unknown_person_alert (sector hogar) cuando se detecta un rostro no reconocido."""
        payload = self._base_event("unknown_person_alert", camera_id, "medium")
        payload.update({"track_id": track_id, "bbox": bbox,
                        "face_snapshot_b64": face_snapshot_b64})
        self.enqueue("POST", "/api/events", payload)

    def post_epp_violation(self, camera_id: str, track_id: int, bbox: dict,
                           violations: list, present: list, confidence: float) -> None:
        """Emite epp_violation con los items de EPP faltantes y los presentes (sector industrial)."""
        payload = self._base_event("epp_violation", camera_id, "high")
        payload.update({"track_id": track_id, "bbox": bbox, "violations": violations,
                        "present": present, "confidence": round(confidence, 3)})
        self.enqueue("POST", "/api/events", payload)

    def post_fire_smoke_alert(self, camera_id: str, detected: list,
                              confidence: float, frame_snapshot_b64: str = "") -> None:
        """Emite fire_smoke_alert con los elementos detectados (e.g. ["fire", "smoke"])."""
        payload = self._base_event("fire_smoke_alert", camera_id, "critical")
        payload.update({"detected": detected, "confidence": round(confidence, 3),
                        "frame_snapshot_b64": frame_snapshot_b64})
        self.enqueue("POST", "/api/events", payload)

    def post_vehicle_detected(self, camera_id: str, track_id: int, bbox: dict,
                              plate: str, plate_confidence: float) -> None:
        """Emite vehicle_detected con la placa leída y su confianza (sector industrial)."""
        payload = self._base_event("vehicle_detected", camera_id)
        payload.update({"track_id": track_id, "bbox": bbox,
                        "plate": plate, "plate_confidence": round(plate_confidence, 3)})
        self.enqueue("POST", "/api/events", payload)

    def post_analytics_snapshot(self, camera_id: str, stats: dict,
                                period_seconds: float = 60.0) -> None:
        """Emite analytics_snapshot cada ANALYTICS_SEND_INTERVAL_SECS con conteos acumulados del período."""
        payload = self._base_event("analytics_snapshot", camera_id)
        payload.update({"period_seconds": period_seconds, **stats})
        self.enqueue("POST", "/api/analytics", payload)

    def post_crop(self, camera_id: str, track_id: int, frame_num: int,
                  crop_b64: str, bbox: dict) -> None:
        """Envía un crop de persona en base64 al endpoint /api/crops para inspección o re-entrenamiento."""
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
        """Envía un frame completo de referencia por cámara al backend (usado para calibración/mapa)."""
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
        """Construye el resultado del handler. event_type vacío = sin evento API adicional este frame."""
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
        """Inicializa los dicts de caché y votación por track_id. Todo vacío al iniciar."""
        self._cache: Dict[int, Tuple[str, str, str, float]] = {}  # track_id → resultado final bloqueado
        self._votes: Dict[int, List[str]] = {}                     # track_id → labels acumulados
        self._vote_last_frame: Dict[int, int] = {}                 # track_id → frame del último voto

    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        """Procesa un objeto detectado, acumula votos del SGIE y devuelve HandlerResult con el label.

        El flujo es: sin resultado SGIE → "Analizando"; con resultado pero sin suficientes votos →
        "Analizando (n/N)"; con N votos → bloquear resultado en _cache y emitir person_classified.
        Una vez en _cache, el resultado se devuelve directamente sin más inferencia.
        """
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
        """Inicializa el handler sin worker. El worker se asigna en init_handlers() vía set_worker()."""
        self._worker = None

    def set_worker(self, worker) -> None:
        self._worker = worker

    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        """Encola crop al PoseWorker cada POSE_SAMPLE_INTERVAL frames y lee el resultado más reciente.

        El crop solo se encola si la persona tiene bbox >= POSE_MIN_PERSON_WIDTH (evita crops muy pequeños).
        El resultado puede ser de un frame anterior — el worker es async y puede ir varios frames atrás.
        Si el resultado indica caída, usa pop_new_fall() para emitir el evento solo una vez por evento real.
        """
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
    Receives face detections from PeopleNet class 2 (face), crops face from
    the full frame, enqueues for embedding extraction, and caches results per track.

    Emits sector-aware API events:
      comercio/industrial → employee_seen / employee_presence / employee_exit
      hogar               → known_person_seen / known_person_exit / unknown_person_alert

    This handler is NOT in _HANDLER_REGISTRY — it is dispatched separately via
    _face_handler because it processes PeopleNet face objects (class_id=2), not person objects.
    """
    PRESENCE_HEARTBEAT_SECS: float = 30.0

    def __init__(self, worker):
        """Configura el handler con el FaceRecognizer worker y los dicts de estado por track.

        _identity_reported evita enviar employee_seen más de una vez por track.
        _last_heartbeat controla la frecuencia del heartbeat de presencia (PRESENCE_HEARTBEAT_SECS).
        _unknown_alerted evita alertas repetidas para la misma cara desconocida (sector hogar).
        """
        self._worker = worker
        self._last_sample: Dict[int, int] = {}                     # track_id → frame del último sample
        self._cache: Dict[int, Tuple[str, float]] = {}             # track_id → (nombre, similitud)
        self._identity_reported: Set[int] = set()                  # tracks con employee_seen ya emitido
        self._last_heartbeat: Dict[int, float] = {}                # track_id → ts del último heartbeat
        self._unknown_alerted: Set[int] = set()                    # tracks con unknown_person_alert ya emitido

    def process_face(
        self,
        face_obj_meta,
        frame_num: int,
        frame_np,
        persons_meta: list,
        camera_id: str,
    ) -> None:
        """Procesa una cara detectada por PeopleNet (class_id=2): extrae crop, encola al worker, emite eventos.

        Flujo:
        1. Filtrar por confianza mínima (FACE_DET_CONFIDENCE_THRESHOLD).
        2. Encontrar el track de persona padre que contiene esta cara (_find_parent_track).
        3. Encolar crop de cara al FaceRecognizer worker cada FACE_SAMPLE_INTERVAL frames.
        4. Leer resultado del worker y, según sector, emitir employee_seen o unknown_person_alert.
        """
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
        """Devuelve (nombre, similitud) si el track tiene una identidad reconocida, o None.

        Actualiza el _cache local con el resultado más reciente del worker antes de retornar.
        """
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

_pose_worker       = None  # PoseWorker (fall_detection)
_face_recognizer   = None  # FaceRecognizer (face_recognition)
_appearance_worker = None  # AppearanceWorker — generates 512-dim embeddings per person
_reid_manager      = None  # ReIdManager — local cross-camera identity DB (active when OSNet exists)
_ws_client         = None  # WsPositionClient (WebSocket position telemetry)

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
    reid_gallery_size: int = 10,
) -> None:
    """Instancia los workers async según las capacidades activas del pipeline.

    Los workers NO se arrancan aquí — solo se crean. start_workers() los arranca
    después de pipeline.set_state(PLAYING) para que el contexto CUDA de TensorRT
    esté inicializado antes de que ONNX Runtime intente usar la GPU.

    Workers creados según capacidades:
      - AppearanceWorker + ReIdManager: siempre (si existe el modelo OSNet)
      - PoseWorker: si 'fall_detection' está en pipeline_capabilities
      - FaceRecognizer: si 'face_recognition' está en pipeline_capabilities
      - WsPositionClient: si WS_BASE_URL está configurado
    """
    global _pose_worker, _face_recognizer, _appearance_worker, _reid_manager, _ws_client

    # AppearanceWorker + ReIdManager — always active when OSNet model is present.
    # ReIdManager uses the embeddings to maintain a persistent local identity DB,
    # enabling cross-camera deduplication and two event types: person_entry vs
    # person_channel_change.
    osnet_path = str(Path(model_dir) / "osnet" / "osnet_x0_25_market1501.onnx")
    if Path(osnet_path).exists():
        from appearance_worker import AppearanceWorker
        from reid_manager import ReIdManager
        _appearance_worker = AppearanceWorker(osnet_path)
        reid_db_path = str(Path(model_dir).parent / "reid_db.json")
        _reid_manager = ReIdManager(db_path=reid_db_path, gallery_max_size=reid_gallery_size)
        logger.info("ReIdManager active — DB: %s", reid_db_path)
    else:
        logger.warning("OSNet model not found at %s — appearance vectors and local ReID disabled. "
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
    """Detiene todos los workers async y persiste el estado del ReIdManager a disco.

    Se llama desde el bloque finally de main() para un shutdown limpio.
    ReIdManager.flush() fuerza la escritura de reid_db.json antes de cerrar
    para que las identidades persistan entre reinicios del pipeline.
    """
    if _appearance_worker is not None:
        _appearance_worker.stop()
    if _reid_manager is not None:
        _reid_manager.flush()          # persistir DB a reid_db.json antes de salir
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
    """Instancia y registra los handlers activos según las capacidades del pipeline.

    _active_handlers es la lista que el probe itera por cada persona detectada.
    _face_handler se maneja por separado porque procesa detecciones de cara de PeopleNet
    (class_id=2), no objetos de persona del loop principal.

    Se conecta cada handler con su worker correspondiente (ej. FallDetection → PoseWorker).
    """
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

# FPS tracking para nx:qa:pipeline_stats (solo QA mode)
_fps_frame_counts: Dict[int, int] = {}
_fps_last_publish: float = 0.0
_FPS_PUBLISH_INTERVAL: float = 5.0


def _get_analytics(pad_index: int) -> Dict:
    """Retorna (creando si no existe) el dict de analytics acumulados para una cámara.

    Acumula conteos durante ANALYTICS_SEND_INTERVAL_SECS (60 s) y luego se reinicia.
    El dict incluye: person_count, gender_male, gender_female, age_gender_classes (dict).
    """
    if pad_index not in _analytics:
        # Inicializar contadores en cero para esta cámara
        _analytics[pad_index] = {
            "person_count": 0, "gender_male": 0,
            "gender_female": 0, "age_gender_classes": {},
        }
    return _analytics[pad_index]


def _get_analytics_last_sent(pad_index: int) -> float:
    """Retorna el timestamp (monotonic) del último envío de analytics para esta cámara.

    Se inicializa al tiempo actual para que el primer envío ocurra después de un intervalo completo.
    """
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
    """Guarda un crop de persona en disco y lo envía al backend vía API.

    Uso dual: el archivo local sirve como dataset para reentrenamiento futuro,
    y el POST a /api/crops permite al backend construir un historial visual por persona.
    Máximo CROP_MAX_PER_PERSON crops por track para no saturar disco ni red.
    """
    # Guardar en disco: crops/<camera_id>/<track_id>/frame_NNNNNN.jpg
    person_dir = Path(CROPS_DIR) / camera_id / str(track_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    filepath = person_dir / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filepath), crop_bgr)

    # Codificar en base64 para el payload JSON de la API
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

        # If the entry event was deferred (ReID path) but never emitted, emit now.
        # This happens when a person disappears before their embedding was ready.
        if not state.entry_emitted:
            api_client.post_person_entry(
                state.camera_id, track_id,
                state.pending_bbox or {},
                state.pending_conf,
                state.is_entry_exit_cam,
                global_id=state.global_id,
                is_return=False,
            )
            _get_analytics(pad_index)["person_count"] += 1

        api_client.post_person_exit(
            state.camera_id, track_id, dwell, state.is_entry_exit_cam,
            global_id=state.global_id,
        )
        if _face_handler:
            _face_handler.on_track_lost(track_id, dwell)
        _crop_counts.pop(track_id, None)
        _crop_last_frame.pop(track_id, None)
        if _appearance_worker is not None:
            _appearance_worker.clear_result(track_id, pad_index)
        for handler in _active_handlers:
            _cleanup_handler_cache(handler, track_id)
        logger.debug("Track lost: pad=%d track=%d dwell=%.1fs global=%s",
                     pad_index, track_id, dwell, state.global_id)


def _cleanup_handler_cache(handler, track_id: int) -> None:
    """Remove track_id from any caches the handler holds."""
    for attr in ("_cache", "_votes", "_vote_last_frame", "_last_sample"):
        d = getattr(handler, attr, None)
        if isinstance(d, dict):
            d.pop(track_id, None)


def _handle_appearance_reid(
    track_key: Tuple[int, int],
    p_track_id: int,
    camera_id: str,
    bbox: dict,
    confidence: float,
    is_entry_exit_cam: bool,
    frame_num: int,
    frame_np,
    pad_index: int,
) -> None:
    """
    Handle AppearanceWorker + ReIdManager for one visible track.

    With ReID enabled (_reid_manager is not None):
      - Defers person_entry until embedding is available.
      - On match: emits person_entry (new/return) or person_channel_change.
      - Deadline fallback: if embedding not ready within ENTRY_EMIT_DEADLINE_FRAMES,
        emits person_entry with global_id=None so the track is never silently lost.

    Without ReID (legacy path, _reid_manager is None):
      - Sends appearance_vector to the backend for server-side matching.
    """
    if _appearance_worker is None:
        return

    state = _active_tracks[track_key]

    # ── Try to consume a ready embedding ─────────────────────────────────────
    vec = _appearance_worker.get_result(p_track_id, pad_index)
    if vec is not None:
        _appearance_worker.clear_result(p_track_id, pad_index)  # consumido

        if not state.appearance_sent:
            # Primera embedding: determinar identidad global via match_or_create
            state.appearance_sent = True

            if _reid_manager is not None:
                global_id, event_type, prev_camera = _reid_manager.match_or_create(
                    vec, camera_id
                )
                state.global_id = global_id
                logger.info(
                    "ReID track=%d cam=%s → %s gid=%s prev=%s",
                    p_track_id, camera_id, event_type, global_id, prev_camera,
                )

                # Same-camera channel_change = tracker briefly lost + re-detected same person.
                # Treat as person_return so we don't emit a spurious cross-camera event.
                if event_type == "channel_change" and prev_camera == camera_id:
                    event_type = "person_return"

                if not state.entry_emitted:
                    state.entry_emitted = True
                    if event_type == "channel_change":
                        api_client.post_person_channel_change(
                            camera_id, p_track_id, bbox, confidence,
                            global_id, prev_camera, is_entry_exit_cam,
                        )
                    else:
                        api_client.post_person_entry(
                            camera_id, p_track_id, bbox, confidence,
                            is_entry_exit_cam,
                            global_id=global_id,
                            is_return=(event_type == "person_return"),
                        )
                        _get_analytics(pad_index)["person_count"] += 1
            else:
                # Legacy path: send vector to backend for server-side re-ID
                api_client.post_person_appearance(camera_id, p_track_id, vec.tolist())

        elif state.global_id is not None and _reid_manager is not None:
            # Embeddings posteriores: actualizar el DB con EMA para mantener el
            # vector fresco. Mejora el re-ID cuando la persona regresa o cambia de
            # cámara, ya que la referencia evoluciona con su apariencia real.
            _reid_manager.update_embedding(state.global_id, vec)

    # ── Enqueue crops ─────────────────────────────────────────────────────────
    # · Primer frame: inmediatamente (inicio del track)
    # · Antes del primer match: cada 15 frames hasta recibir la embedding
    # · Después del primer match: cada 90 frames (~3 s a 30 fps) para refrescar el DB
    if (frame_np is not None
            and bbox["width"]  >= CROP_MIN_WIDTH
            and bbox["height"] >= CROP_MIN_HEIGHT):
        needs_crop = (
            frame_num == state.first_frame
            or (not state.appearance_sent and frame_num % 15 == 0)
            or (state.appearance_sent and state.global_id is not None and frame_num % 90 == 0)
        )
        if needs_crop:
            crop = frame_np[
                bbox["top"]:bbox["top"]    + bbox["height"],
                bbox["left"]:bbox["left"]  + bbox["width"],
            ]
            if crop.size > 0:
                _appearance_worker.enqueue(crop, p_track_id, pad_index, frame_num)

    # ── Deadline fallback: emit entry if embedding never arrived ─────────────
    if (_reid_manager is not None
            and not state.entry_emitted
            and frame_num >= state.entry_deadline):
        state.entry_emitted = True
        api_client.post_person_entry(
            camera_id, p_track_id,
            state.pending_bbox or bbox,
            state.pending_conf or confidence,
            is_entry_exit_cam,
            global_id=None,
            is_return=False,
        )
        _get_analytics(pad_index)["person_count"] += 1
        logger.debug("ReID deadline reached track=%d cam=%s — entry emitted without global_id",
                     p_track_id, camera_id)


def _update_fps_stats(pad_index: int) -> None:
    """Incrementa contador por cámara y publica FPS a Redis cada _FPS_PUBLISH_INTERVAL s."""
    global _fps_last_publish
    _fps_frame_counts[pad_index] = _fps_frame_counts.get(pad_index, 0) + 1
    now = time.monotonic()
    if now - _fps_last_publish >= _FPS_PUBLISH_INTERVAL:
        elapsed = now - _fps_last_publish if _fps_last_publish > 0 else _FPS_PUBLISH_INTERVAL
        fps_per_cam = {
            _camera_id_for(idx): round(count / elapsed, 1)
            for idx, count in _fps_frame_counts.items()
        }
        if _redis_qa:
            try:
                _redis_qa.set("nx:qa:pipeline_stats", json.dumps({
                    "fps_per_camera": fps_per_cam,
                    "fps_total": round(sum(fps_per_cam.values()), 1),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }))
            except Exception:
                pass
        _fps_frame_counts.clear()
        _fps_last_publish = now


# ==============================================================================
# 6. PRE-TILER ANALYTICS PROBE — helpers
#    Each function below is one named step of pre_tiler_analytics_probe.
#    Extracted to keep the probe itself readable as a table of contents.
# ==============================================================================

def _maybe_refresh_redis_state() -> None:
    """Periodically pull entry/exit pad config and camera type flags from Redis.

    Called on every probe invocation; the actual Redis fetch happens only
    every _ENTRY_EXIT_REFRESH_EVERY / _CAMERA_TYPE_REFRESH_EVERY calls.
    """
    global _entry_exit_probe_count, _camera_type_probe_count
    _entry_exit_probe_count += 1
    if _entry_exit_probe_count % _ENTRY_EXIT_REFRESH_EVERY == 0:
        _refresh_entry_exit_from_redis()
    _camera_type_probe_count += 1
    if _camera_type_probe_count % _CAMERA_TYPE_REFRESH_EVERY == 0:
        _refresh_camera_types_from_redis()


def _should_count_camera(pad_index: int) -> bool:
    """Return True if this camera's detections should be processed.

    External cameras (e.g. street-facing) and internal cameras (floor
    level) are counted independently based on client config.
    """
    is_external = pad_index in _external_pads
    if is_external and not _count_external:
        return False
    if not is_external and not _count_internal:
        return False
    return True


def _collect_detections(frame_meta) -> Tuple[List, List]:
    """Split all detected objects in one frame into persons and faces.

    Filters by PGIE unique component ID and OSD_CONFIDENCE_THRESHOLD.
    Returns (persons, faces) as lists of NvDsObjectMeta.
    """
    persons: List = []
    faces:   List = []
    for obj_meta in _iter_pyds_list(
        frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast
    ):
        if obj_meta.unique_component_id != PGIE_UNIQUE_ID:
            continue
        if obj_meta.confidence < OSD_CONFIDENCE_THRESHOLD:
            continue
        class_id = int(obj_meta.class_id)
        if class_id == PGIE_CLASS_PERSON:
            persons.append(obj_meta)
        elif class_id == PGIE_CLASS_FACE:
            faces.append(obj_meta)
    return persons, faces


def _get_frame_as_bgr(
    gst_buffer,
    frame_meta,
    camera_id: str,
    frame_num: int,
    n_persons: int,
    pad_index: int,
) -> Optional[np.ndarray]:
    """Copy the current frame from GPU to CPU as BGR, only if a worker needs it.

    Workers that need pixel data: pose (fall detection), face recognizer,
    and appearance/ReID — but only when there are new or unsettled tracks.
    Recording also triggers a copy when active.

    Pushes the frame copy to RecordingManager when recording is active
    (no second copy needed — the array is already detached from GPU memory).

    Returns None if no worker needs pixel data or if the GPU copy fails.
    """
    needs_pixel = _recording_manager is not None and _recording_manager.is_recording

    if frame_meta.num_obj_meta > 0:
        if _pose_worker is not None or _face_recognizer is not None:
            needs_pixel = True
        elif _appearance_worker is not None and not needs_pixel:
            n_known = sum(1 for k in _active_tracks if k[0] == pad_index)
            has_new_persons = n_persons > n_known
            has_unsettled_embeddings = any(
                not s.appearance_sent
                for k, s in _active_tracks.items()
                if k[0] == pad_index
            )
            needs_pixel = has_new_persons or has_unsettled_embeddings

    if not needs_pixel:
        return None

    try:
        n_frame   = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_bgr = cv2.cvtColor(np.array(n_frame, copy=True, order='C'), cv2.COLOR_RGBA2BGR)

        if frame_num == 0:
            fh, fw = frame_bgr.shape[:2]
            logger.info("[QA Probe A] Camera %s full-res frame size: %dx%d", camera_id, fw, fh)

        if _recording_manager is not None and _recording_manager.is_recording:
            _recording_manager.push_camera_frame(camera_id, frame_bgr)

        return frame_bgr

    except Exception as e:
        if frame_num % 30 == 0:
            logger.warning("pre_tiler get_nvds_buf_surface frame=%d: %s", frame_num, e)
        return None


def _dispatch_handlers(
    obj_meta,
    frame_num: int,
    frame_np: Optional[np.ndarray],
    camera_id: str,
    p_track_id: int,
    pad_index: int,
    bbox: dict,
) -> Tuple[bool, str]:
    """Run all active analytics handlers on one person and route their events.

    Calls the appropriate API endpoint for each event type and updates
    the per-camera analytics accumulators.

    Returns (qa_fall, qa_age_gender) for writing into _track_labels so
    Probe B can draw the overlay labels without re-reading classifier meta.
    qa_age_gender is the display portion only — no "P#N |" prefix.
    """
    qa_fall       = False
    qa_age_gender = ""

    for handler in _active_handlers:
        if not _is_capability_active(getattr(handler, "_cap_name", "")):
            continue
        result = handler.process(obj_meta, frame_num, frame_np=frame_np)
        if result is None:
            continue

        if result.osd_text:
            _set_osd_text(obj_meta, result.osd_text, border_color=result.border_color)
            if isinstance(handler, _AgeGenderHandler):
                # Strip "P#N | " prefix — Probe B builds base_label separately.
                osd = result.osd_text
                qa_age_gender = osd.split(" | ", 1)[1] if " | " in osd else osd

        if result.event_type == "person_classified":
            api_client.post_person_classified(
                camera_id, p_track_id, bbox,
                result.det_extra.get("demographics", {}),
            )
            an    = _get_analytics(pad_index)
            au    = result.analytics_update
            if "age_gender_classes" in au:
                label = au["age_gender_classes"]
                an["age_gender_classes"][label] = an["age_gender_classes"].get(label, 0) + 1
            if "gender_key" in au:
                an[au["gender_key"]] += 1

        elif result.event_type == "fall_detected":
            qa_fall = True
            api_client.post_fall_detected(
                camera_id, p_track_id, bbox,
                result.det_extra.get("fall_score", 0),
                result.det_extra.get("avg_kp_conf", 0.0),
            )

    return qa_fall, qa_age_gender


def _write_face_osd_and_get_label(obj_meta, p_track_id: int) -> Optional[str]:
    """Append a confirmed face identity to the OSD label and return it for Probe B.

    Returns None if no identity is confirmed yet or the person is unknown.
    """
    if not _face_handler:
        return None
    identity = _face_handler.get_identity(p_track_id)
    if not identity:
        return None
    name, conf = identity
    if name == "Desconocido":
        return None
    current_label = obj_meta.text_params.display_text or f"P#{p_track_id}"
    _set_osd_text(
        obj_meta,
        f"{current_label} | {name} {conf:.0%}",
        border_color=(0.2, 1.0, 0.4, 1.0),
    )
    return f"{name} {conf:.0%}"


def _maybe_capture_crop(
    frame_np: np.ndarray,
    camera_id: str,
    p_track_id: int,
    frame_num: int,
    bbox: dict,
) -> None:
    """Sample a person crop and send it to the API, rate-limited per track.

    Skips if the bbox is below minimum size, the per-person crop budget is
    exhausted, or CROP_SAMPLE_INTERVAL frames haven't elapsed since the last
    sample for this track.
    """
    last_crop = _crop_last_frame.get(p_track_id, -CROP_SAMPLE_INTERVAL)
    count     = _crop_counts.get(p_track_id, 0)

    too_small      = bbox["height"] < CROP_MIN_HEIGHT or bbox["width"] < CROP_MIN_WIDTH
    budget_reached = count >= CROP_MAX_PER_PERSON
    too_soon       = (frame_num - last_crop) < CROP_SAMPLE_INTERVAL

    if too_small or budget_reached or too_soon:
        return

    crop = frame_np[
        bbox["top"]: bbox["top"] + bbox["height"],
        bbox["left"]: bbox["left"] + bbox["width"],
    ]
    if crop.size == 0:
        return

    _save_and_send_crop(crop, camera_id, p_track_id, frame_num, bbox)
    _crop_counts[p_track_id]     = count + 1
    _crop_last_frame[p_track_id] = frame_num


def _process_person(
    obj_meta,
    pad_index: int,
    camera_id: str,
    is_entry_exit_cam: bool,
    frame_num: int,
    frame_np: Optional[np.ndarray],
) -> int:
    """Handle the full analytics lifecycle for one tracked person per frame.

    Steps:
      1. Register new tracks or update last-seen timestamp on existing ones.
      2. Emit person_entry immediately when ReID is disabled; otherwise defer
         until _handle_appearance_reid resolves the global identity.
      3. Dispatch analytics handlers (age/gender, fall, etc.).
      4. Overlay face identity on OSD if a confirmed match exists.
      5. Write _track_labels for Probe B (post-tiler overlay).
      6. Sample and send a person crop at CROP_SAMPLE_INTERVAL cadence.

    Returns the track_id so the caller can build visible_ids for expiry.
    """
    p_track_id = int(obj_meta.object_id)
    r    = obj_meta.rect_params
    bbox = {
        "left":   max(0, int(r.left)),
        "top":    max(0, int(r.top)),
        "width":  int(r.width),
        "height": int(r.height),
    }
    track_key = (pad_index, p_track_id)
    now       = time.monotonic()

    # Track lifecycle — register new entries or update existing track timestamp.
    if track_key not in _active_tracks:
        _active_tracks[track_key] = _TrackState(
            first_frame=frame_num,
            last_frame=frame_num,
            first_ts=now,
            camera_id=camera_id,
            is_entry_exit_cam=is_entry_exit_cam,
            entry_deadline=frame_num + ENTRY_EMIT_DEADLINE_FRAMES,
            pending_bbox=bbox,
            pending_conf=float(obj_meta.confidence),
        )
        if _reid_manager is None:
            # No ReID active — emit entry immediately.
            # When ReID is active, _handle_appearance_reid defers the emit
            # until an embedding is ready (or the deadline is reached).
            api_client.post_person_entry(
                camera_id, p_track_id, bbox,
                float(obj_meta.confidence), is_entry_exit_cam,
            )
            _get_analytics(pad_index)["person_count"] += 1
            _active_tracks[track_key].entry_emitted = True
    else:
        _active_tracks[track_key].last_frame = frame_num

    _handle_appearance_reid(
        track_key, p_track_id, camera_id, bbox,
        float(obj_meta.confidence), is_entry_exit_cam,
        frame_num, frame_np, pad_index,
    )

    # Baseline OSD label — handlers below may overwrite this with richer labels.
    _set_osd_text(obj_meta, f"P#{p_track_id}", border_color=(0.2, 0.6, 1.0, 1.0))

    qa_fall, qa_age_gender = _dispatch_handlers(
        obj_meta, frame_num, frame_np, camera_id, p_track_id, pad_index, bbox
    )

    qa_face_name = _write_face_osd_and_get_label(obj_meta, p_track_id)

    # Write labels for Probe B (post-tiler overlay).
    track_state = _active_tracks.get(track_key)
    _track_labels[p_track_id] = {
        "face_name":  qa_face_name,
        "fall":       qa_fall,
        "age_gender": qa_age_gender,
        "global_id":  track_state.global_id if track_state else None,
    }

    if frame_np is not None:
        _maybe_capture_crop(frame_np, camera_id, p_track_id, frame_num, bbox)

    return p_track_id


def _prune_stale_track_labels() -> None:
    """Remove _track_labels entries for tracks no longer in _active_tracks."""
    active_track_ids = {k[1] for k in _active_tracks}
    for tid in list(_track_labels):
        if tid not in active_track_ids:
            _track_labels.pop(tid, None)


def _maybe_send_reference_frame(
    pad_index: int,
    camera_id: str,
    frame_np: Optional[np.ndarray],
    frame_num: int,
    frame_meta,
    visible_ids: Set[int],
) -> None:
    """Send the first fully empty scene frame as a reference image, once per camera.

    Waits for a frame with zero detections (num_obj_meta covers persons,
    bags, and faces) so the reference image never contains a person.
    """
    already_sent   = _reference_frame_sent.get(pad_index, False)
    scene_is_empty = not visible_ids and frame_meta.num_obj_meta == 0

    if already_sent or frame_np is None or not scene_is_empty:
        return

    fh, fw = frame_np.shape[:2]
    _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_b64 = base64.b64encode(buf).decode("utf-8")
    api_client.post_reference_frame(camera_id, frame_num, frame_b64, fw, fh)
    _reference_frame_sent[pad_index] = True
    logger.info("Reference frame sent: camera=%s frame=%d size=%dx%d",
                camera_id, frame_num, fw, fh)


def _send_analytics_snapshot_if_due(pad_index: int, camera_id: str) -> None:
    """Flush accumulated analytics counts to the API if the interval has elapsed.

    Resets the per-camera accumulators after sending so the next window
    starts clean. Interval: ANALYTICS_SEND_INTERVAL_SECS (default 60 s).
    """
    now = time.monotonic()
    if now - _get_analytics_last_sent(pad_index) < ANALYTICS_SEND_INTERVAL_SECS:
        return

    an = _get_analytics(pad_index)
    api_client.post_analytics_snapshot(camera_id, {
        "people_count":       an["person_count"],
        "gender_male":        an["gender_male"],
        "gender_female":      an["gender_female"],
        "age_gender_classes": an["age_gender_classes"],
    }, period_seconds=ANALYTICS_SEND_INTERVAL_SECS)

    _analytics[pad_index]           = {"person_count": 0, "gender_male": 0,
                                        "gender_female": 0, "age_gender_classes": {}}
    _analytics_last_sent[pad_index] = now


# ==============================================================================
# 6. PRE-TILER ANALYTICS PROBE
# ==============================================================================

def pre_tiler_analytics_probe(_pad, info):
    """Probe A (QA mode) — connected to the caps_rgba src-pad, before the tiler.

    Receives one RGBA frame per camera at the original capture resolution
    (e.g. 1920×1080). Runs all analytics: handler dispatch, track lifecycle,
    ReID, and API events. Writes _track_labels[track_id] so Probe B (the
    post-tiler overlay probe) can draw labels without re-reading classifier meta.
    Updates per-camera FPS and publishes to nx:qa:pipeline_stats every 5 s.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    _maybe_refresh_redis_state()

    for frame_meta in _iter_pyds_list(
        batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast
    ):
        pad_index         = frame_meta.pad_index
        camera_id         = _camera_id_for(pad_index)
        frame_num         = frame_meta.frame_num
        is_entry_exit_cam = pad_index in _entry_exit_pads

        if not _should_count_camera(pad_index):
            continue

        _update_fps_stats(pad_index)

        persons, faces = _collect_detections(frame_meta)

        frame_np = _get_frame_as_bgr(
            gst_buffer, frame_meta, camera_id, frame_num,
            n_persons=len(persons), pad_index=pad_index,
        )

        if _face_handler and faces and frame_np is not None:
            if _is_capability_active("face_recognition"):
                for face_meta in faces:
                    _face_handler.process_face(
                        face_meta, frame_num, frame_np, persons, camera_id
                    )

        if _recording_manager is not None and persons:
            _recording_manager.notify_detection(len(persons))

        visible_ids: Set[int] = set()
        for obj_meta in persons:
            track_id = _process_person(
                obj_meta, pad_index, camera_id, is_entry_exit_cam, frame_num, frame_np
            )
            visible_ids.add(track_id)

        _expire_lost_tracks(pad_index, frame_num, visible_ids)
        _prune_stale_track_labels()

        _maybe_send_reference_frame(
            pad_index, camera_id, frame_np, frame_num, frame_meta, visible_ids
        )

        if _ws_client is not None and visible_ids and frame_np is not None:
            fh, fw = frame_np.shape[:2]
            _accumulate_positions(pad_index, camera_id, persons, fw, fh, frame_num)

        _send_analytics_snapshot_if_due(pad_index, camera_id)

    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe(_pad, info):
    """
    En producción (NX_QA_ENABLED=false): probe único en caps_rgba src-pad.
      Recibe frames RGBA full-res por cámara. Ejecuta todos los analytics.
    En QA mode (NX_QA_ENABLED=true): Probe B en tiler src-pad.
      Recibe frame RGBA tileado 640x360. Solo dibuja overlays y sirve MJPEG.
      pre_tiler_analytics_probe (Probe A) ya ejecutó todos los analytics.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    if _IS_QA_ENABLED:
        return _qa_overlay_probe(gst_buffer, batch_meta)
    return _production_analytics_probe(gst_buffer, batch_meta)


def _qa_overlay_probe(gst_buffer, batch_meta) -> Gst.PadProbeReturn:
    """Probe B (QA mode): lee frame tileado RGBA, dibuja overlays, sirve MJPEG.
    Los analytics ya fueron ejecutados por pre_tiler_analytics_probe (Probe A).
    Los labels vienen de _track_labels escrito por Probe A.
    """
    qa_frame_bgr: Optional[np.ndarray] = None
    qa_all_tracks: List[dict] = []

    for frame_meta in _iter_pyds_list(
        batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast
    ):
        # Leer frame tileado una sola vez (post-tiler es un frame compuesto unico)
        if qa_frame_bgr is None:
            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                qa_frame_bgr = cv2.cvtColor(
                    np.array(n_frame, copy=True, order='C'), cv2.COLOR_RGBA2BGR
                )
            except Exception as e:
                if frame_meta.frame_num % 30 == 0:
                    logger.warning("[QA Probe B] buf_surface: %s", e)

        pad_index = frame_meta.pad_index
        camera_id = _camera_id_for(pad_index)

        # Solo bboxes de personas (class_id=0) para el overlay visual
        for obj_meta in _iter_pyds_list(
            frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast
        ):
            if (obj_meta.unique_component_id != PGIE_UNIQUE_ID
                    or int(obj_meta.class_id) != PGIE_CLASS_PERSON
                    or obj_meta.confidence < OSD_CONFIDENCE_THRESHOLD):
                continue

            p_track_id = int(obj_meta.object_id)
            r = obj_meta.rect_params

            # Camera attribution: el tiler produce un frame compuesto único cuyo
            # frame_meta.pad_index es siempre 0. Calcular la cámara real desde la
            # posición del bbox en el grid tileado.
            cx = max(0, int(r.left)) + max(1, int(r.width)) // 2
            cy = max(0, int(r.top)) + max(1, int(r.height)) // 2
            tiled_col = min(cx // _qa_cell_w, _qa_tiler_cols - 1) if _qa_cell_w else 0
            tiled_row = min(cy // _qa_cell_h, _qa_tiler_rows - 1) if _qa_cell_h else 0
            obj_pad_idx = tiled_row * _qa_tiler_cols + tiled_col
            obj_camera_id = _camera_id_for(obj_pad_idx)

            bbox_tiled = (
                max(0, int(r.left)), max(0, int(r.top)),
                max(1, int(r.width)), max(1, int(r.height)),
            )

            # Age/gender directo del SGIE classifier_meta (sin pixel data)
            age_gender_text = ""
            for cls_meta in _iter_pyds_list(
                obj_meta.classifier_meta_list, pyds.NvDsClassifierMeta.cast
            ):
                if cls_meta.unique_component_id == SGIE_AGE_GENDER_ID:
                    _, gender_disp, age_disp, _ = _parse_age_gender(cls_meta)
                    if gender_disp:
                        age_gender_text = f"{gender_disp}|{age_disp}"
                    break

            # Face name, fall, global_id desde _track_labels (escrito por Probe A)
            labels = _track_labels.get(p_track_id, {})
            gid = labels.get("global_id")
            base_label = f"P#{p_track_id}" + (f"·{gid[:6]}" if gid else "")
            label_parts = [base_label]
            # Preferir el resultado votado de Probe A (_track_labels); el raw del
            # classifier_meta solo sirve de fallback cuando Probe A aún no procesó el track.
            ag_display = labels.get("age_gender") or age_gender_text
            if ag_display:
                label_parts.append(ag_display)
            if labels.get("face_name"):
                label_parts.append(labels["face_name"])

            # Respetar filtros de cámara externa/interna: si la cámara no debe
            # contar, tampoco se dibujan sus bboxes en el overlay QA.
            _obj_is_external = obj_pad_idx in _external_pads
            if _obj_is_external and not _count_external:
                continue
            if not _obj_is_external and not _count_internal:
                continue

            qa_all_tracks.append({
                "pad_index":  obj_pad_idx,
                "channel_id": obj_camera_id,
                "track_id":   p_track_id,
                "confidence": round(float(obj_meta.confidence), 3),
                "bbox_tiled": bbox_tiled,
                "label": " | ".join(label_parts),
                "fall":  labels.get("fall", False),
            })

    if qa_frame_bgr is not None:
        try:
            _draw_qa_overlays(qa_frame_bgr, qa_all_tracks)
        except Exception as _e:
            logger.debug("[QA] overlay error: %s", _e)

        try:
            tiled_frame_queue.put_nowait(qa_frame_bgr.copy())
        except queue.Full:
            pass

        for pad_idx in _channel_map:
            cam_id = _camera_id_for(pad_idx)
            q = camera_frame_queues.get(cam_id)
            if q is None:
                continue
            ox = (pad_idx % _qa_tiler_cols) * _qa_cell_w
            oy = (pad_idx // _qa_tiler_cols) * _qa_cell_h
            if (oy + _qa_cell_h <= qa_frame_bgr.shape[0]
                    and ox + _qa_cell_w <= qa_frame_bgr.shape[1]):
                try:
                    q.put_nowait(qa_frame_bgr[oy:oy + _qa_cell_h, ox:ox + _qa_cell_w].copy())
                except queue.Full:
                    pass

        if qa_all_tracks:
            by_cam: Dict[str, List] = {}
            for t in qa_all_tracks:
                by_cam.setdefault(t["channel_id"], []).append({
                    "track_id":   t["track_id"],
                    "confidence": t["confidence"],
                    "label":      t["label"],
                    "fall":       t["fall"],
                })
            for cam_id, tracks in by_cam.items():
                _qa_publish("nx:qa:detections", {
                    "cam":    cam_id,
                    "ts":     datetime.now(timezone.utc).isoformat(),
                    "tracks": tracks,
                })

    return Gst.PadProbeReturn.OK


def _production_analytics_probe(gst_buffer, batch_meta) -> Gst.PadProbeReturn:
    """Probe unico en produccion (NX_QA_ENABLED=false).
    Recibe frames RGBA full-res por camara (sin tiler en el pipeline).
    Lazy frame read: solo copia GPU->CPU cuando hay detecciones y workers de pixel."""
    global _entry_exit_probe_count, _camera_type_probe_count
    _entry_exit_probe_count += 1
    if _entry_exit_probe_count % _ENTRY_EXIT_REFRESH_EVERY == 0:
        _refresh_entry_exit_from_redis()
    _camera_type_probe_count += 1
    if _camera_type_probe_count % _CAMERA_TYPE_REFRESH_EVERY == 0:
        _refresh_camera_types_from_redis()

    for frame_meta in _iter_pyds_list(
        batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast
    ):
        frame_num         = frame_meta.frame_num
        pad_index         = frame_meta.pad_index
        camera_id         = _camera_id_for(pad_index)
        is_entry_exit_cam = pad_index in _entry_exit_pads

        # Cortocircuitar si este tipo de cámara no debe contar
        _is_external_cam = pad_index in _external_pads
        if _is_external_cam and not _count_external:
            continue
        if not _is_external_cam and not _count_internal:
            continue

        # Lazy frame read: GPU→CPU copy only when a worker genuinely needs pixel data.
        # For pose/face: always copy when detections exist.
        # For appearance/ReID: only copy when there are new tracks or tracks still awaiting
        # their first embedding — once all settled, skip the copy to avoid blocking GStreamer.
        # For recording: always copy when RecordingManager is actively recording.
        frame_np = None
        _needs_pixel = _recording_manager is not None and _recording_manager.is_recording
        if frame_meta.num_obj_meta > 0:
            if _pose_worker is not None or _face_recognizer is not None:
                _needs_pixel = True
            if not _needs_pixel and _appearance_worker is not None:
                # Count only person tracks (class 0), not bags/faces which inflate num_obj_meta.
                n_persons_in_frame = sum(
                    1 for om in _iter_pyds_list(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast)
                    if int(om.class_id) == PGIE_CLASS_PERSON
                )
                n_known = sum(1 for k in _active_tracks if k[0] == pad_index)
                _needs_pixel = (
                    n_persons_in_frame > n_known
                    or any(not s.appearance_sent
                           for k, s in _active_tracks.items() if k[0] == pad_index)
                )
        if _needs_pixel:
            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame_np = np.array(n_frame, copy=True, order='C')
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
                if _recording_manager is not None and _recording_manager.is_recording:
                    _recording_manager.push_camera_frame(camera_id, frame_np)
            except Exception as e:
                if frame_num % 30 == 0:
                    logger.warning("get_nvds_buf_surface fallo frame=%d: %s", frame_num, e)

        persons_meta: List = []
        face_metas:   List = []
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

        if _face_handler and face_metas and frame_np is not None:
            if _is_capability_active("face_recognition"):
                for face_obj_meta in face_metas:
                    _face_handler.process_face(
                        face_obj_meta, frame_num, frame_np, persons_meta, camera_id
                    )

        # Notificar al recorder cuando hay personas (no necesita Redis)
        if _recording_manager is not None and persons_meta:
            _recording_manager.notify_detection(len(persons_meta))

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

            if track_key not in _active_tracks:
                _active_tracks[track_key] = _TrackState(
                    first_frame=frame_num,
                    last_frame=frame_num,
                    first_ts=now,
                    camera_id=camera_id,
                    is_entry_exit_cam=is_entry_exit_cam,
                    entry_deadline=frame_num + ENTRY_EMIT_DEADLINE_FRAMES,
                    pending_bbox=bbox,
                    pending_conf=float(obj_meta.confidence),
                )
                if _reid_manager is None:
                    # No ReID: emit immediately (original behaviour)
                    api_client.post_person_entry(
                        camera_id, p_track_id, bbox,
                        float(obj_meta.confidence), is_entry_exit_cam,
                    )
                    _get_analytics(pad_index)["person_count"] += 1
                    _active_tracks[track_key].entry_emitted = True
            else:
                _active_tracks[track_key].last_frame = frame_num

            _handle_appearance_reid(
                track_key, p_track_id, camera_id, bbox,
                float(obj_meta.confidence), is_entry_exit_cam,
                frame_num, frame_np, pad_index,
            )

            _set_osd_text(obj_meta, f"P#{p_track_id}", border_color=(0.2, 0.6, 1.0, 1.0))

            for handler in _active_handlers:
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
                    api_client.post_fall_detected(
                        camera_id, p_track_id, bbox,
                        result.det_extra.get("fall_score", 0),
                        result.det_extra.get("avg_kp_conf", 0.0),
                    )

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
                        _save_and_send_crop(crop, camera_id, p_track_id, frame_num, bbox)
                        _crop_counts[p_track_id] = count + 1
                        _crop_last_frame[p_track_id] = frame_num

        _expire_lost_tracks(pad_index, frame_num, visible_ids)

        if (not _reference_frame_sent.get(pad_index, False)
                and frame_np is not None
                and len(visible_ids) == 0
                and frame_meta.num_obj_meta == 0):
            fh, fw = frame_np.shape[:2]
            _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_b64 = base64.b64encode(buf).decode("utf-8")
            api_client.post_reference_frame(camera_id, frame_num, frame_b64, fw, fh)
            _reference_frame_sent[pad_index] = True
            logger.info("Frame de referencia enviado camera=%s (frame=%d, %dx%d)",
                        camera_id, frame_num, fw, fh)

        if _ws_client is not None and visible_ids and frame_np is not None:
            fh, fw = frame_np.shape[:2]
            _accumulate_positions(pad_index, camera_id, persons_meta, fw, fh, frame_num)

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

    return Gst.PadProbeReturn.OK
