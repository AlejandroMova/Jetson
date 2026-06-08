"""
probes.py — NX Computing AI | Edge Inference Core

Probe de GStreamer para extracción de metadatos DeepStream y envío a API REST.
Importado por app.py y app_video_testing.py.

Arquitectura:
  PGIE (PeopleNet, gie-id=1) detecta person/bag/face en el frame completo.
  Handlers opcionales (uno por capability activa) procesan cada persona detectada.
  FaceRecognizer y AppearanceWorker corren en hilos de fondo (patrón queue+thread).

Stream mode (NX_STREAM_ENABLED=true):
  En stream mode el pipeline inserta nvmultistreamtiler (640×360) después del probe
  analytics. Un segundo probe (tiled_overlay_probe) dibuja bboxes sobre el frame
  tileado y lo sirve vía MjpegServer (:8080/viewer/all). Cero overhead en producción.

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
# ==============================================================================
JETSON_ID: str    = os.environ.get("JETSON_ID",  os.uname().nodename)
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY: str      = os.environ.get("API_KEY",    "your-api-key")

# Mapa pad_index → número de canal real del DVR.
# Se inicializa desde app.py llamando a init_channel_map(cfg.channels).
_channel_map: Dict[int, int] = {}

# Sector del cliente: "comercio" | "industrial" | "hogar"
_JETSON_SECTOR: str = "comercio"

# Pad indices que corresponden a cámaras de entrada/salida
_entry_exit_pads: set = set()

# Tipo de cámara: pad indices externos y flags de conteo por tipo
_external_pads:   set  = set()
_count_internal:  bool = True
_count_external:  bool = True


def init_channel_map(channels: list):
    """Llamar desde app.py después de load_config(), antes de arrancar el pipeline."""
    global _channel_map
    _channel_map = {idx: ch for idx, ch in enumerate(channels)}
    logger.info("Channel map: %s", _channel_map)


def init_sector(sector: str) -> None:
    """Configura el sector del cliente: 'comercio', 'industrial' o 'hogar'.

    El sector afecta el tipo de evento emitido por face_recognition
    (employee_seen vs known_person_seen).
    """
    global _JETSON_SECTOR
    _JETSON_SECTOR = sector
    logger.info("Sector: %s", sector)


def init_entry_exit_pads(pad_indices: set) -> None:
    """Define qué pad indices corresponden a cámaras de entrada/salida del local."""
    global _entry_exit_pads
    _entry_exit_pads = pad_indices
    logger.info("Entry/exit pad indices: %s", pad_indices)


def init_camera_types(external_pad_indices: set, count_internal: bool, count_external: bool) -> None:
    """Configura qué cámaras son externas y si se deben contar sus personas."""
    global _external_pads, _count_internal, _count_external
    _external_pads  = external_pad_indices
    _count_internal = count_internal
    _count_external = count_external
    logger.info(
        "Camera types — external pads: %s  count_internal=%s  count_external=%s",
        external_pad_indices, count_internal, count_external,
    )


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

# ── Track lifecycle ───────────────────────────────────────────────────────────
TRACK_LOST_TIMEOUT_FRAMES: int = 60

# ── Face recognition ──────────────────────────────────────────────────────────
FACE_SAMPLE_INTERVAL: int  = 30

# ── Analytics ─────────────────────────────────────────────────────────────────
ANALYTICS_SEND_INTERVAL_SECS: float = 60.0

# ── Reference frame ───────────────────────────────────────────────────────────
# Tiempo mínimo entre reintentos cuando el backend no ha confirmado el frame.
REFERENCE_FRAME_RETRY_SECS: float = 30.0
# Mínimo 24 h entre reenvíos — alineado con la granularidad de día del calendario del frontend.
REFERENCE_FRAME_MIN_INTERVAL_SECS: float = 86_400.0
# Fracción de diferencia normalizada (0.0-1.0) que dispara un nuevo frame de referencia.
# 0.15 ≈ 15 % de los píxeles cambian significativamente tras normalizar por iluminación media.
REFERENCE_FRAME_CHANGE_THRESHOLD: float = 0.15
# Brillo mínimo aceptable (media de píxeles en gris, escala 0-255).
# Frames por debajo de este valor (ej. noche, cámara tapada) se descartan como fondo.
# 30/255 ≈ 12 % de brillo máximo — rechaza negro puro y escenas casi a oscuras.
REFERENCE_FRAME_MIN_BRIGHTNESS: float = 30.0

# ── Crop capture ──────────────────────────────────────────────────────────────
CROPS_DIR: str            = "crops"
CROP_SAMPLE_INTERVAL: int = 15
CROP_MAX_PER_PERSON: int  = 5
CROP_MIN_WIDTH: int       = 48
CROP_MIN_HEIGHT: int      = 96
# Frames to wait for an appearance embedding before emitting person_entry anyway.
# 30 frames ≈ 1 second — enough for OSNet on CPU even under queue pressure.
ENTRY_EMIT_DEADLINE_FRAMES: int = 30


# ==============================================================================
# STREAM MODE — activo solo cuando NX_STREAM_ENABLED=true
# ==============================================================================

_IS_STREAM_ENABLED: bool = os.getenv("NX_STREAM_ENABLED", "false").lower() == "true"

# ANSI colors para stream logs. Desactivar con NO_COLOR=1 (útil para grep en logs sin escapes).
_NO_COLOR: bool = os.getenv("NO_COLOR", "0") == "1"
_C: dict = {} if (_NO_COLOR or not _IS_STREAM_ENABLED) else {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "cyan":    "\033[96m",
    "magenta": "\033[95m",
    "red":     "\033[91m",
}


def _slog(*parts: str) -> None:
    """Imprime una línea de log coloreado a stdout cuando stream mode está activo.

    Visible en `docker logs -f`. flush=True es necesario para que las líneas
    aparezcan inmediatamente sin buffer en el contexto de Docker.
    Solo emite output si NX_STREAM_ENABLED=true — cero overhead en producción.
    """
    if _IS_STREAM_ENABLED:
        print("".join(parts) + _C.get("reset", ""), flush=True)


# Acumulador para el resumen periódico de analytics_snapshot.
# _send() corre en el hilo worker de NxApiClient — el lock protege acceso concurrente.
_analytics_slog_cameras: list = []
_analytics_slog_last_t: float = time.monotonic()  # evita flush inmediato en el primer call
_ANALYTICS_SLOG_INTERVAL: float = 60.0  # segundos entre líneas de resumen
_analytics_slog_lock = threading.Lock()


def _accumulate_analytics_slog(camera_id: str) -> None:
    """Acumula cámaras con analytics_snapshot exitoso y emite una línea resumen cada 60s.

    En lugar de una línea por cámara por minuto, agrupa todas las cámaras en un solo
    mensaje periódico: [API] analytics_snapshot  ['cam1', 'cam2', ...]  200
    """
    global _analytics_slog_last_t
    cams_to_log = None
    with _analytics_slog_lock:
        if camera_id not in _analytics_slog_cameras:
            _analytics_slog_cameras.append(camera_id)
        now = time.monotonic()
        if now - _analytics_slog_last_t >= _ANALYTICS_SLOG_INTERVAL:
            _analytics_slog_last_t = now
            cams_to_log = list(_analytics_slog_cameras)
            _analytics_slog_cameras.clear()
    if cams_to_log is not None:
        _slog(
            f"{_C.get('yellow', '')}[API]{_C.get('reset', '')} ",
            f"analytics_snapshot  ",
            f"{_C.get('cyan', '')}{cams_to_log}{_C.get('reset', '')}  200",
        )


# Queue de frames tileados para MjpegServer (poblada por tiled_overlay_probe)
tiled_frame_queue: queue.Queue = queue.Queue(maxsize=1)

# Grid del tiler — seteado por init_stream_grid() desde app.py antes de arrancar
_stream_tiler_cols: int = 1
_stream_tiler_rows: int = 1
_stream_cell_w: int = 640
_stream_cell_h: int = 360

# Labels de tracks activos — Probe A (osd_sink_pad_buffer_probe) escribe,
# Probe B (tiled_overlay_probe) lee. Seguro: ambos probes corren en el mismo hilo GStreamer.
_track_labels: dict = {}   # track_id → {"label": str, "fall": bool}

# Mapeo de display: global_id (hex12) → número corto (1, 2, 3...). Solo para stream, no afecta API.
_display_ids: dict[str, int] = {}
_display_id_counter: int = 0


def init_stream_grid(cols: int, rows: int, cell_w: int, cell_h: int) -> None:
    """Setea las dimensiones del grid del tiler. Llamar desde app.py después de crear el tiler."""
    global _stream_tiler_cols, _stream_tiler_rows, _stream_cell_w, _stream_cell_h
    _stream_tiler_cols = cols
    _stream_tiler_rows = rows
    _stream_cell_w = cell_w
    _stream_cell_h = cell_h
    logger.info("[Stream] Grid: %dx%d tiles de %dx%d px", cols, rows, cell_w, cell_h)


def _draw_tiled_overlays(frame_bgr: np.ndarray, tracks: list) -> None:
    """Dibuja bboxes y labels sobre frame_bgr in-place (coordenadas ya en espacio tileado).

    tracks: lista de {"bbox_tiled": (x, y, w, h), "label": str, "fall": bool}.
    """
    for t in tracks:
        x1, y1, w, h = t["bbox_tiled"]
        x2 = min(frame_bgr.shape[1] - 1, x1 + w)
        y2 = min(frame_bgr.shape[0] - 1, y1 + h)
        x1 = max(0, x1)
        y1 = max(0, y1)
        if x2 <= x1 or y2 <= y1:
            continue
        color = (0, 0, 230) if t.get("fall") else (0, 210, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = t["label"]
        txt_y = max(y1 - 3, 12)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame_bgr, (x1, txt_y - th - 2), (x1 + tw, txt_y + 1), color, -1)
        cv2.putText(frame_bgr, label, (x1, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def tiled_overlay_probe(_pad, info):
    """Probe B — adjuntado al src pad del nvmultistreamtiler (solo en stream mode).

    Recibe el frame compuesto 640×360 RGBA. Lee _track_labels para obtener los
    labels ya calculados por Probe A, mapea las coordenadas de cada bbox al espacio
    tileado usando la geometría del grid, dibuja bboxes+labels y encola el frame
    en tiled_frame_queue para que MjpegServer lo sirva como MJPEG.
    """
    gst_buffer = info.get_buffer()
    if gst_buffer is None:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    cols = _stream_tiler_cols
    rows = _stream_tiler_rows
    cw   = _stream_cell_w
    ch   = _stream_cell_h

    # El tiler produce un único frame compuesto (batch_id=0)
    frame_meta = None
    for fm in _iter_pyds_list(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
        frame_meta = fm
        break
    if frame_meta is None:
        return Gst.PadProbeReturn.OK

    # Leer frame RGBA del tiler (640×360 → pequeño, GPU→CPU rápido)
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_bgr = cv2.cvtColor(np.array(n_frame, copy=True, order='C'), cv2.COLOR_RGBA2BGR)
    except Exception:
        return Gst.PadProbeReturn.OK

    # Recolectar tracks del frame tileado con coordenadas mapeadas
    overlay_tracks = []
    for obj_meta in _iter_pyds_list(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast):
        if int(obj_meta.class_id) != PGIE_CLASS_PERSON:
            continue
        r = obj_meta.rect_params
        # Centro del bbox en el frame tileado → índice de celda → pad_idx original
        cx = int(r.left + r.width / 2)
        cy = int(r.top + r.height / 2)
        tile_col = min(cx // cw, cols - 1)
        tile_row = min(cy // ch, rows - 1)
        # Obtener label calculado por Probe A para este track_id
        track_id = int(obj_meta.object_id)
        info_dict = _track_labels.get(track_id, {})
        label = info_dict.get("label") or f"P#{track_id}"
        overlay_tracks.append({
            "bbox_tiled": (int(r.left), int(r.top), int(r.width), int(r.height)),
            "label":      label,
            "fall":       info_dict.get("fall", False),
        })

    if overlay_tracks:
        _draw_tiled_overlays(frame_bgr, overlay_tracks)

    try:
        tiled_frame_queue.put_nowait(frame_bgr)
    except queue.Full:
        pass  # MjpegServer no alcanzó a consumir — OK descartar frame anterior

    return Gst.PadProbeReturn.OK


# ==============================================================================
# 0. TRACK STATE
# ==============================================================================

@dataclass
class _TrackState:
    """Estado por track activo. Vive en _active_tracks[track_id] desde el primer frame hasta el exit.

    Los campos de ReID (entry_emitted, entry_deadline, global_id, pending_bbox, pending_conf)
    solo se usan cuando _reid_manager está activo (modelo OSNet encontrado).
    """
    first_frame:       int
    last_frame:        int
    first_ts:          float
    camera_id:         str
    is_entry_exit_cam: bool          = False
    appearance_sent:   bool          = False
    entry_emitted:     bool          = False
    entry_deadline:    int           = 0
    global_id:         Optional[str] = None
    pending_bbox:      Optional[dict] = None
    pending_conf:      float          = 0.0


# ==============================================================================
# 1. BUS CALL
# ==============================================================================
def bus_call(_bus, message, loop):
    """Maneja mensajes del bus GStreamer: EOS para salida limpia, WARNING y ERROR para logging."""
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
# ==============================================================================
class NxApiClient:
    """
    Envía peticiones HTTP al backend en un hilo de fondo independiente.
    El probe de GStreamer solo hace enqueue() (O(1), sin I/O), garantizando
    que las llamadas de red nunca impacten los FPS del pipeline.
    """

    def __init__(self, base_url: str, api_key: str, max_queue_size: int = 512):
        """Configura el cliente con keep-alive HTTP y cola FIFO."""
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
        # Callbacks invocados desde el worker thread tras un 2xx exitoso.
        # Clave: endpoint exacto (ej. "/api/cameras/reference-frame").
        self._success_callbacks: Dict[str, "Callable[[dict], None]"] = {}

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
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._session.close()
        logger.info("NxApiClient detenido.")

    def register_success_callback(self, endpoint: str, cb: "Callable[[dict], None]") -> None:
        """Registra un callback invocado por el worker thread cuando el endpoint retorna 2xx.

        El callback recibe el payload original enviado. Se llama desde el hilo worker,
        por lo que debe ser thread-safe y no bloquear.

        Args:
            endpoint: Ruta exacta, ej. "/api/cameras/reference-frame".
            cb:       Función que acepta el dict del payload enviado.
        """
        self._success_callbacks[endpoint] = cb

    def enqueue(self, method: str, endpoint: str, payload: Optional[dict] = None):
        """Encola una petición HTTP sin bloquear. Descarta con warning si la cola está llena."""
        try:
            self._queue.put_nowait((method, endpoint, payload))
        except queue.Full:
            logger.warning("Cola API llena — descartando: %s %s", method, endpoint)

    def _worker_loop(self):
        """Loop principal del hilo worker: consume la cola y envía peticiones HTTP al backend."""
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
        """Envía la petición HTTP al backend. Timeout 5 s — errores se loguean, nunca se propagan."""
        url = f"{self._base_url}{endpoint}"
        try:
            resp = self._session.request(method=method, url=url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.debug("%s %s → %d", method, endpoint, resp.status_code)
            if endpoint == "/api/analytics":
                # Analytics snapshots son uno por cámara cada 60s — agrupa en una línea resumen.
                _accumulate_analytics_slog((payload or {}).get("camera_id", "?"))
            else:
                _slog(
                    f"{_C.get('yellow', '')}[API]{_C.get('reset', '')} ",
                    f"{method} {endpoint}  ",
                    f"{_C.get('bold', '')}{resp.status_code}{_C.get('reset', '')}",
                )
            # Invocar callback de éxito si está registrado para este endpoint.
            cb = self._success_callbacks.get(endpoint)
            if cb is not None:
                try:
                    cb(payload or {})
                except Exception as exc:
                    logger.warning("Success callback error (%s): %s", endpoint, exc)
        except requests.exceptions.Timeout:
            logger.warning("Timeout: %s %s", method, url)
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP %d: %s %s → %s",
                         e.response.status_code, method, url, e.response.text[:300])
        except requests.exceptions.ConnectionError:
            logger.debug("Sin conexión: %s", url)

    def _base_event(self, event_type: str, camera_id: str, severity: str = "info") -> dict:
        """Construye los campos comunes a todos los eventos del backend."""
        return {
            "event_id":  str(uuid.uuid4()),
            "type":      event_type,
            "sector":    _JETSON_SECTOR,
            "jetson_id": JETSON_ID,
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity":  severity,
        }

    # ── Eventos MVP ──────────────────────────────────────────────────────────

    def post_person_entry(self, camera_id: str, track_id: int, bbox: dict,
                          confidence: float, is_entry_exit_cam: bool,
                          global_id: Optional[str] = None,
                          is_return: bool = False) -> None:
        """Emite person_entry. entry_type="return" si la persona ya fue vista antes."""
        payload = self._base_event("person_entry", camera_id)
        payload.update({
            "track_id":             track_id,
            "bbox":                 bbox,
            "confidence":           round(confidence, 3),
            "is_entry_exit_camera": is_entry_exit_cam,
            "entry_type":           "return" if is_return else "new",
        })
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_channel_change(self, camera_id: str, track_id: int, bbox: dict,
                                   confidence: float, global_id: str,
                                   prev_camera_id: Optional[str],
                                   is_entry_exit_cam: bool) -> None:
        """Emite person_channel_change cuando la misma persona cambia de cámara (ReID)."""
        payload = self._base_event("person_channel_change", camera_id)
        payload.update({
            "track_id":             track_id,
            "bbox":                 bbox,
            "confidence":           round(confidence, 3),
            "global_id":            global_id,
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        if prev_camera_id:
            payload["prev_camera_id"] = prev_camera_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_exit(self, camera_id: str, track_id: int,
                         dwell_seconds: float, is_entry_exit_cam: bool,
                         global_id: Optional[str] = None) -> None:
        """Emite person_exit con el tiempo total de permanencia del track."""
        payload = self._base_event("person_exit", camera_id)
        payload.update({
            "track_id":             track_id,
            "dwell_seconds":        round(dwell_seconds, 1),
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_classified(self, camera_id: str, track_id: int,
                               bbox: dict, demographics: dict) -> None:
        """Emite person_classified con el resultado de edad/género (tras VOTES_REQUIRED votos)."""
        payload = self._base_event("person_classified", camera_id)
        payload.update({"track_id": track_id, "bbox": bbox, "demographics": demographics})
        self.enqueue("POST", "/api/events", payload)

    def post_person_appearance(self, camera_id: str, track_id: int,
                               appearance_vector: list) -> None:
        """Emite person_appearance con el vector OSNet 512-dim L2-normalizado."""
        payload = self._base_event("person_appearance", camera_id)
        payload.update({"track_id": track_id, "appearance_vector": appearance_vector})
        self.enqueue("POST", "/api/events", payload)

    def post_employee_seen(self, camera_id: str, employee_id: str, track_id: int,
                           similarity: float, bbox: dict) -> None:
        """Emite employee_seen (o known_person_seen en hogar) cuando se identifica un rostro conocido."""
        evt = "known_person_seen" if _JETSON_SECTOR == "hogar" else "employee_seen"
        payload = self._base_event(evt, camera_id)
        payload.update({
            "track_id":   track_id,
            "bbox":       bbox,
            "similarity": round(similarity, 3),
            "employee_id" if _JETSON_SECTOR != "hogar" else "name": employee_id,
        })
        self.enqueue("POST", "/api/events", payload)

    def post_employee_presence(self, camera_id: str, employee_id: str, track_id: int) -> None:
        """Emite employee_presence periódicamente para empleados en cámara (heartbeat)."""
        payload = self._base_event("employee_presence", camera_id)
        payload.update({"track_id": track_id,
                        "employee_id" if _JETSON_SECTOR != "hogar" else "name": employee_id})
        self.enqueue("POST", "/api/events", payload)

    def post_employee_exit(self, camera_id: str, employee_id: str,
                           track_id: int, dwell_seconds: float) -> None:
        """Emite employee_exit (o known_person_exit en hogar) con tiempo de permanencia."""
        evt = "known_person_exit" if _JETSON_SECTOR == "hogar" else "employee_exit"
        payload = self._base_event(evt, camera_id)
        payload.update({
            "track_id":     track_id,
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

    def post_analytics_snapshot(self, camera_id: str, stats: dict,
                                period_seconds: float = 60.0) -> None:
        """Emite analytics_snapshot cada ANALYTICS_SEND_INTERVAL_SECS con conteos acumulados."""
        payload = self._base_event("analytics_snapshot", camera_id)
        payload.update({"period_seconds": period_seconds, **stats})
        self.enqueue("POST", "/api/analytics", payload)

    def post_crop(self, camera_id: str, track_id: int, frame_num: int,
                  crop_b64: str, bbox: dict) -> None:
        """Envía un crop de persona en base64 al endpoint /api/crops."""
        payload = {
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "track_id":  track_id,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_b64": crop_b64,
            "bbox":      bbox,
        }
        self.enqueue("POST", "/api/crops", payload)

    def post_reference_frame(self, camera_id: str, frame_num: int,
                             frame_b64: str, width: int, height: int) -> None:
        """Envía un frame de referencia (escena vacía) por cámara al backend."""
        payload = {
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_b64": frame_b64,
            "width":     width,
            "height":    height,
        }
        self.enqueue("POST", "/api/cameras/reference-frame", payload)


# Instancia global — se inicializa en main() antes de arrancar el pipeline
api_client = NxApiClient(base_url=API_BASE_URL, api_key=API_KEY)


# ==============================================================================
# 3. HELPERS DE METADATOS DEEPSTREAM
# ==============================================================================
def _iter_pyds_list(pyds_list, cast_fn):
    """Generador seguro sobre listas enlazadas de pyds."""
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

# Mismo mapa pero para el payload de la API
_AGE_GENDER_API_MAP: Dict[str, Tuple[str, str]] = {
    "female_adult":  ("female", "adult"),
    "female_senior": ("female", "senior"),
    "female_young":  ("female", "young"),
    "male_adult":    ("male",   "adult"),
    "male_senior":   ("male",   "senior"),
    "male_young":    ("male",   "young"),
}


def _parse_age_gender(classifier_meta) -> Tuple[str, str, str, float]:
    """Extrae el label y probabilidad del SGIE ResNet-18.

    Retorna (raw_label, gender_display, age_display, prob).
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
    """Aplica estilo y texto al OSD de un objeto."""
    obj_meta.text_params.display_text = text

    fp = obj_meta.text_params.font_params
    fp.font_name = "Sans"
    fp.font_size = 12
    fp.font_color.set(1.0, 1.0, 1.0, 1.0)

    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

    obj_meta.rect_params.border_color.set(*border_color)
    obj_meta.rect_params.border_width = 2


# ==============================================================================
# 4. HANDLERS DE PIPELINE
# ==============================================================================

class _HandlerResult:
    """Resultado que un handler devuelve al probe para OSD y API."""
    __slots__ = ("osd_text", "border_color", "event_type", "det_extra", "analytics_update")

    def __init__(
        self,
        osd_text: str = "",
        border_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        event_type: str = "",
        det_extra: Optional[dict] = None,
        analytics_update: Optional[dict] = None,
    ):
        """Construye el resultado del handler. event_type vacío = sin evento API adicional."""
        self.osd_text = osd_text
        self.border_color = border_color
        self.event_type = event_type
        self.det_extra = det_extra or {}
        self.analytics_update = analytics_update or {}


class _AgeGenderHandler:
    """Clasificación de género y grupo de edad por votación sobre el SGIE ResNet-18.

    Acumula VOTES_REQUIRED muestras antes de fijar el resultado y emitir person_classified.
    """

    def __init__(self):
        """Inicializa dicts de caché y votación por track_id."""
        self._cache: Dict[int, Tuple[str, str, str, float]] = {}
        self._votes: Dict[int, List[str]] = {}
        self._vote_last_frame: Dict[int, int] = {}

    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        """Acumula votos del SGIE y devuelve HandlerResult cuando hay suficientes votos."""
        p_track_id = int(obj_meta.object_id)
        r = obj_meta.rect_params

        # Filtro de tamaño mínimo — personas muy lejanas producen clasificaciones ruidosas
        if int(r.width) < VOTE_MIN_WIDTH or int(r.height) < VOTE_MIN_HEIGHT:
            if p_track_id in self._cache:
                raw, gender_disp, age_disp, prob = self._cache[p_track_id]
                prefix = str(obj_meta.text_params.display_text) or "..."
                return _HandlerResult(
                    osd_text=f"{prefix} | {gender_disp} | {age_disp} {prob:.0%}",
                    border_color=(0.0, 1.0, 0.0, 1.0),
                )
            return None

        # Si ya está bloqueado en caché, devolver el resultado sin volver a inferir
        if p_track_id in self._cache:
            raw, gender_disp, age_disp, prob = self._cache[p_track_id]
            prefix = str(obj_meta.text_params.display_text) or "..."
            return _HandlerResult(
                osd_text=f"{prefix} | {gender_disp} | {age_disp} {prob:.0%}",
                border_color=(0.0, 1.0, 0.0, 1.0),
            )

        # Leer resultado del SGIE si hay classifier_meta disponible
        last = self._vote_last_frame.get(p_track_id, -VOTE_SAMPLE_INTERVAL)
        if frame_num - last < VOTE_SAMPLE_INTERVAL:
            return None

        for cls_meta in _iter_pyds_list(
            obj_meta.classifier_meta_list, pyds.NvDsClassifierMeta.cast
        ):
            if cls_meta.unique_component_id != SGIE_AGE_GENDER_ID:
                continue
            raw_label, gender_disp, age_disp, prob = _parse_age_gender(cls_meta)
            if not raw_label or prob < MIN_CLASSIFICATION_PROB:
                break
            self._vote_last_frame[p_track_id] = frame_num
            votes = self._votes.setdefault(p_track_id, [])
            votes.append(raw_label)
            n = len(votes)
            if n < VOTES_REQUIRED:
                prefix = str(obj_meta.text_params.display_text) or "..."
                return _HandlerResult(
                    osd_text=f"{prefix} | Analizando ({n}/{VOTES_REQUIRED})",
                    border_color=(0.2, 0.6, 1.0, 1.0),
                )
            # Suficientes votos: bloquear con la moda
            from collections import Counter
            winner = Counter(votes).most_common(1)[0][0]
            winner_gd, winner_ad = _AGE_GENDER_LABEL_MAP.get(winner, ("?", "?"))
            winner_prob = votes.count(winner) / len(votes)
            self._cache[p_track_id] = (winner, winner_gd, winner_ad, winner_prob)
            gender_api, age_api = _AGE_GENDER_API_MAP.get(winner, ("unknown", "unknown"))
            event_type = "person_classified"
            det_extra = {
                "demographics": {
                    "gender":     gender_api,
                    "age_group":  age_api,
                    "label":      winner,
                    "confidence": round(winner_prob, 3),
                }
            }
            analytics = {
                "age_gender_classes": winner,
                "gender_key": "gender_male" if winner.startswith("male") else "gender_female",
            }
            prefix = str(obj_meta.text_params.display_text) or "..."
            return _HandlerResult(
                osd_text=f"{prefix} | {winner_gd} | {winner_ad} {winner_prob:.0%}",
                border_color=(0.0, 1.0, 0.0, 1.0),
                event_type=event_type,
                det_extra=det_extra,
                analytics_update=analytics,
            )

        prefix = str(obj_meta.text_params.display_text) or "..."
        return _HandlerResult(
            osd_text=f"{prefix} | Analizando",
            border_color=(0.2, 0.6, 1.0, 1.0),
        )


class _FaceRecognitionHandler:
    """Face recognition via async FaceRecognizer (InsightFace ArcFace).

    Recibe detecciones de cara de PeopleNet class 2 (face), extrae crop, encola
    al worker y emite eventos sector-aware (employee_seen, unknown_person_alert, etc.).

    Este handler NO está en _HANDLER_REGISTRY — se despacha separadamente porque
    procesa objetos de cara (class_id=2), no personas del loop principal.
    """
    PRESENCE_HEARTBEAT_SECS: float = 30.0

    def __init__(self, worker):
        """Configura el handler con el FaceRecognizer worker."""
        self._worker = worker
        self._last_sample: Dict[int, int] = {}
        self._cache: Dict[int, Tuple[str, float]] = {}
        self._identity_reported: Set[int] = set()
        self._last_heartbeat: Dict[int, float] = {}
        self._unknown_alerted: Set[int] = set()
        # Rastrea qué tracks ya recibieron una línea ROSTRO Desconocido en stream log.
        self._unknown_face_logged: Set[int] = set()

    def process_face(
        self,
        face_obj_meta,
        frame_num: int,
        frame_np,
        persons_meta: list,
        camera_id: str,
    ) -> None:
        """Procesa una cara detectada por PeopleNet (class_id=2): extrae crop, encola y emite eventos."""
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

        # identity key is now a UUID string (or "Unknown" if below threshold)
        identity_key, conf = identity
        display_name = self._worker.get_display_name(identity_key)
        now = time.monotonic()

        if identity_key != "Unknown":
            if parent_track_id not in self._identity_reported:
                self._identity_reported.add(parent_track_id)
                # identity_key is the backend-assigned UUID — used in events for FK join
                api_client.post_employee_seen(camera_id, identity_key, parent_track_id, conf, bbox)
                _slog(
                    f"{_C.get('cyan', '')}[{camera_id}]{_C.get('reset', '')} ",
                    f"{_C.get('green', '')}{_C.get('bold', '')}EMPLEADO{_C.get('reset', '')}   ",
                    f"track={parent_track_id:<4} ",
                    f"nombre={_C.get('bold', '')}{display_name}{_C.get('reset', '')}  sim={conf:.2f}",
                )
            last_hb = self._last_heartbeat.get(parent_track_id, 0.0)
            if now - last_hb >= self.PRESENCE_HEARTBEAT_SECS:
                api_client.post_employee_presence(camera_id, identity_key, parent_track_id)
                self._last_heartbeat[parent_track_id] = now
        else:
            # Cara no reconocida — loguear una vez por track en stream mode.
            if parent_track_id not in self._unknown_face_logged:
                self._unknown_face_logged.add(parent_track_id)
                _slog(
                    f"{_C.get('cyan', '')}[{camera_id}]{_C.get('reset', '')} ",
                    f"ROSTRO     ",
                    f"track={parent_track_id:<4} ",
                    f"Desconocido  sim={conf:.2f}",
                )
            if _JETSON_SECTOR == "hogar" and parent_track_id not in self._unknown_alerted:
                self._unknown_alerted.add(parent_track_id)
                _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                face_b64 = base64.b64encode(buf).decode("utf-8")
                api_client.post_unknown_person_alert(camera_id, parent_track_id, face_b64, bbox)

    def on_track_lost(self, track_id: int, dwell_seconds: float) -> None:
        """Llamado desde _expire_lost_tracks. Emite employee_exit / known_person_exit."""
        identity = self._cache.get(track_id)
        if identity and identity[0] != "Unknown":
            identity_key, _ = identity  # UUID string assigned by the backend
            state = _active_tracks.get(
                next((k for k in _active_tracks if k[1] == track_id), (None, None))
            )
            camera_id = state.camera_id if state else ""
            api_client.post_employee_exit(camera_id, identity_key, track_id, dwell_seconds)
        self._cache.pop(track_id, None)
        self._identity_reported.discard(track_id)
        self._last_heartbeat.pop(track_id, None)
        self._unknown_alerted.discard(track_id)

    def get_identity(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Devuelve (nombre, similitud) si hay identidad reconocida, o None."""
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

_face_recognizer    = None  # FaceRecognizer (face_recognition)
_appearance_worker  = None  # AppearanceWorker — embeddings 512-dim por persona
_reid_manager       = None  # ReIdManager — DB local cross-cámara
_ws_client          = None  # WsPositionClient (WebSocket de posiciones / heatmaps)
_jetson_sync_client = None  # JetsonSyncClient (Socket.IO face roster sync)

# Buffer de posiciones: pad_index → list of {track_id, x_norm, y_norm}
_position_buffer: Dict[int, List[dict]] = {}
_position_last_sent: Dict[int, float]  = {}
POSITION_SEND_INTERVAL: float = 10.0


def init_workers(
    pipeline_capabilities: List[str],
    model_dir: str,
    face_db_path: str = "",
    ws_base_url: str = "",
    api_key: str = "",
    reid_gallery_size: int = 10,
) -> None:
    """Instancia los workers async según las capacidades activas del pipeline.

    Workers creados según capacidades:
      - AppearanceWorker + ReIdManager: siempre (si existe el modelo OSNet)
      - FaceRecognizer: si 'face_recognition' está en pipeline_capabilities
      - WsPositionClient: si WS_BASE_URL está configurado (heatmaps)
      - JetsonSyncClient: si face_recognition activo y API_BASE_URL configurado
    """
    global _face_recognizer, _appearance_worker, _reid_manager, _ws_client, _jetson_sync_client

    # AppearanceWorker + ReIdManager — siempre activo si existe el modelo OSNet.
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

    if "face_recognition" in pipeline_capabilities:
        from face_recognizer import FaceRecognizer
        _face_recognizer = FaceRecognizer(
            db_path=face_db_path,
            model_root=str(Path(model_dir) / "insightface"),
            api_base_url=API_BASE_URL,
            api_key=api_key,
        )

        # JetsonSyncClient — Socket.IO listener for face_update events from backend.
        # When the backend activates or revokes an employee, this triggers a roster pull.
        if API_BASE_URL:
            from jetson_sync_client import JetsonSyncClient
            _jetson_sync_client = JetsonSyncClient(
                api_base_url=API_BASE_URL,
                api_key=api_key,
                sync_callback=_face_recognizer.sync_from_backend,
            )
        else:
            logger.info("API_BASE_URL not set — JetsonSyncClient disabled (no face roster push).")

    # WebSocket de posiciones para heatmaps en el backend
    if ws_base_url:
        from ws_client import WsPositionClient
        _ws_client = WsPositionClient(
            ws_url=ws_base_url,
            api_key=api_key,
            sector=_JETSON_SECTOR,
        )
    else:
        logger.info("WS_BASE_URL not set — position WebSocket disabled.")

    # ── Callback de confirmación de reference frame ───────────────────────────
    def _on_reference_frame_confirmed(payload: dict) -> None:
        """Llamado por el worker de NxApiClient cuando el backend confirma (2xx) el frame.

        Almacena el frame como baseline para la detección de cambio visual futura.
        Se ejecuta en el hilo worker del NxApiClient, no en el probe de GStreamer.

        Args:
            payload: Payload original enviado al backend (incluye image_b64 y camera_id).
        """
        cam = payload.get("camera_id", "")
        b64 = payload.get("image_b64", "")
        if not cam or not b64:
            return
        try:
            buf  = base64.b64decode(b64)
            arr  = np.frombuffer(buf, dtype=np.uint8)
            gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return
            # Reducir a 64×36 float32 para comparaciones futuras con _scene_changed().
            small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            _reference_frame_confirmed_np[cam] = small
            _reference_frame_confirmed_ts[cam] = time.monotonic()
            logger.info("Frame de referencia confirmado por backend: camera=%s", cam)
        except Exception as exc:
            logger.warning("Error procesando confirmación de reference frame camera=%s: %s", cam, exc)

    api_client.register_success_callback("/api/cameras/reference-frame", _on_reference_frame_confirmed)


def start_workers() -> None:
    """Start all workers. Call after pipeline.set_state(PLAYING)."""
    if _appearance_worker is not None:
        _appearance_worker.start()
    if _face_recognizer is not None:
        _face_recognizer.start()
    if _ws_client is not None:
        _ws_client.start()
    if _jetson_sync_client is not None:
        _jetson_sync_client.start()


def stop_workers() -> None:
    """Detiene todos los workers y persiste el estado del ReIdManager a disco."""
    if _jetson_sync_client is not None:
        _jetson_sync_client.stop()
    if _appearance_worker is not None:
        _appearance_worker.stop()
    if _reid_manager is not None:
        _reid_manager.flush()   # escribir reid_db.json antes de cerrar
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
    "age_gender": _AgeGenderHandler,
    # face_recognition is NOT here — handled separately via _face_handler
}


def _frame_is_bright_enough(frame_np: "np.ndarray") -> bool:
    """Devuelve True si el frame tiene suficiente iluminación para usarse como fondo.

    Descarta frames nocturnos, cámaras tapadas o escenas casi a oscuras que
    resultarían en un fondo negro inútil para el heatmap.

    La comprobación es barata: convierte a gris y calcula la media sobre una
    miniatura 64×36 (2 304 píxeles), no sobre el frame completo.

    Args:
        frame_np: Frame BGR o gris del probe (ya en RAM).

    Returns:
        True si la media de brillo (0-255) supera REFERENCE_FRAME_MIN_BRIGHTNESS.
    """
    gray = frame_np if frame_np.ndim == 2 else cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    return float(small.mean()) >= REFERENCE_FRAME_MIN_BRIGHTNESS


def _scene_changed(current_np: "np.ndarray", prev_np: "np.ndarray") -> bool:
    """Compara el frame actual con el último frame de referencia confirmado.

    Normaliza por iluminación media antes de comparar, de modo que un simple
    cambio de luz (día/noche) no dispare un reenvío. Solo los cambios estructurales
    (productos reordenados, zona reorganizada) superan el umbral.

    Args:
        current_np: Frame BGR o gris full-res del probe (ya en RAM).
        prev_np:    Último frame confirmado almacenado en _reference_frame_confirmed_np
                    — siempre gris, 64×36, float32.

    Returns:
        True si la diferencia normalizada supera REFERENCE_FRAME_CHANGE_THRESHOLD.
    """
    # Convertir a gris si viene en BGR y reducir a miniatura para cómputo rápido.
    gray = current_np if current_np.ndim == 2 else cv2.cvtColor(current_np, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Normalizar por iluminación media (evita falsos positivos por cambio de luz).
    mean_a = small.mean() or 1.0
    mean_b = prev_np.mean() or 1.0
    diff = np.abs(small / mean_a - prev_np / mean_b).mean()
    return diff > REFERENCE_FRAME_CHANGE_THRESHOLD


def init_handlers(pipeline_capabilities: List[str]) -> None:
    """Instancia y registra los handlers activos según las capacidades del pipeline."""
    global _active_handlers, _face_handler
    _active_handlers = []
    _face_handler = None

    for cap in pipeline_capabilities:
        cls = _HANDLER_REGISTRY.get(cap)
        if cls:
            handler = cls()
            _active_handlers.append(handler)

        if cap == "face_recognition" and _face_recognizer is not None:
            _face_handler = _FaceRecognitionHandler(_face_recognizer)
            logger.info("FaceRecognitionHandler → FaceRecognizer")

    names = [type(h).__name__ for h in _active_handlers]
    if _face_handler:
        names.append("_FaceRecognitionHandler")
    logger.info("Active handlers: %s", names if names else ["(none — people_counting only)"])


# ==============================================================================
# 7. PROBE PRINCIPAL
# ==============================================================================

_active_tracks: Dict[Tuple[int, int], _TrackState] = {}
_crop_counts: Dict[int, int] = {}
_crop_last_frame: Dict[int, int] = {}

# ── Reference frame state ─────────────────────────────────────────────────────
# Frame confirmado por el backend (grayscale 64×36 float32) por camera_id.
# None significa que aún no se recibió confirmación 2xx del backend.
_reference_frame_confirmed_np: Dict[str, "np.ndarray"] = {}
# Timestamp monotónico del último 2xx confirmado por cámara.
_reference_frame_confirmed_ts: Dict[str, float] = {}
# Timestamp monotónico del último intento de envío (para controlar el retry).
_reference_frame_last_attempt: Dict[int, float] = {}

_analytics: Dict[int, Dict] = {}
_analytics_last_sent: Dict[int, float] = {}


def _get_analytics(pad_index: int) -> Dict:
    """Retorna (creando si no existe) el dict de analytics acumulados para una cámara."""
    if pad_index not in _analytics:
        _analytics[pad_index] = {
            "person_count": 0, "gender_male": 0,
            "gender_female": 0, "age_gender_classes": {},
        }
    return _analytics[pad_index]


def _get_analytics_last_sent(pad_index: int) -> float:
    """Retorna el timestamp del último envío de analytics para esta cámara."""
    if pad_index not in _analytics_last_sent:
        _analytics_last_sent[pad_index] = time.monotonic()
    return _analytics_last_sent[pad_index]


def _accumulate_positions(
    pad_index: int, camera_id: str, persons_meta: list,
    frame_width: int, frame_height: int, frame_num: int,
) -> None:
    """Acumula centroides normalizados y envía snapshot de posiciones cada POSITION_SEND_INTERVAL s."""
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
    """Guarda un crop en disco y lo envía al backend vía API."""
    person_dir = Path(CROPS_DIR) / camera_id / str(track_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    filepath = person_dir / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filepath), crop_bgr)
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    crop_b64 = base64.b64encode(buf).decode("utf-8")
    api_client.post_crop(camera_id, track_id, frame_num, crop_b64, bbox)


def _expire_lost_tracks(pad_index: int, frame_num: int,
                        visible_ids: Set[int]) -> None:
    """Emite person_exit para tracks no vistos por TRACK_LOST_TIMEOUT_FRAMES frames."""
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

        # Si el entry event fue diferido (ReID) y nunca se emitió, emitirlo ahora
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
        _track_labels.pop(track_id, None)
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
    """Handle AppearanceWorker + ReIdManager para un track visible.

    Con ReID activo: difiere person_entry hasta tener embedding; emite
    person_entry (new/return) o person_channel_change según el match.
    Sin ReID: envía el vector de apariencia al backend para re-ID server-side.
    """
    if _appearance_worker is None:
        return

    state = _active_tracks[track_key]

    # ── Consumir embedding disponible ────────────────────────────────────────
    vec = _appearance_worker.get_result(p_track_id, pad_index)
    if vec is not None:
        _appearance_worker.clear_result(p_track_id, pad_index)

        if not state.appearance_sent:
            state.appearance_sent = True

            if _reid_manager is not None:
                global_id, event_type, prev_camera = _reid_manager.match_or_create(
                    vec, camera_id
                )
                state.global_id = global_id
                logger.info("ReID track=%d cam=%s → %s gid=%s prev=%s",
                            p_track_id, camera_id, event_type, global_id, prev_camera)
                _slog(
                    f"{_C.get('cyan', '')}[{camera_id}]{_C.get('reset', '')} ",
                    f"{_C.get('bold', '')}DETECCIÓN{_C.get('reset', '')}  ",
                    f"track={p_track_id:<4} ",
                    f"gid={_C.get('green', '')}{global_id[:8]}{_C.get('reset', '')}  ",
                    f"tipo={_C.get('yellow', '')}{event_type}{_C.get('reset', '')}",
                    f"  prev={prev_camera}" if prev_camera else "",
                )

                # Same-camera re-detection → tratar como return, no channel_change
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
                # Legacy: enviar vector al backend para re-ID server-side
                api_client.post_person_appearance(camera_id, p_track_id, vec.tolist())

        elif state.global_id is not None and _reid_manager is not None:
            # Embeddings posteriores: actualizar DB para mantener el vector fresco
            _reid_manager.update_embedding(state.global_id, vec)

    # ── Enqueue crops para el worker ─────────────────────────────────────────
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
                bbox["top"]:bbox["top"] + bbox["height"],
                bbox["left"]:bbox["left"] + bbox["width"],
            ]
            if crop.size > 0:
                _appearance_worker.enqueue(crop, p_track_id, pad_index, frame_num)

    # ── Deadline fallback: emitir entry si el embedding no llegó ─────────────
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


def _should_count_camera(pad_index: int) -> bool:
    """Return True si los analytics de esta cámara deben procesarse."""
    is_external = pad_index in _external_pads
    if is_external and not _count_external:
        return False
    if not is_external and not _count_internal:
        return False
    return True


# ==============================================================================
# 8. PROBE ÚNICO — OSD + ANALYTICS + STREAM OVERLAY
# ==============================================================================

def osd_sink_pad_buffer_probe(_pad, info):
    """Probe único en caps_rgba src-pad (frames RGBA full-res por cámara, sin tiler).

    Lazy frame read: solo copia GPU→CPU cuando un worker necesita pixels o
    cuando stream mode está activo (para dibujar bboxes y servir MJPEG).
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

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

        # ── Lazy frame read ───────────────────────────────────────────────────
        # GPU→CPU copy solo cuando face_recognizer o appearance_worker necesitan crops.
        # Stream mode (tiler) no requiere frame aquí — Probe B lo lee del tiler output.
        frame_np = None
        _needs_pixel = False

        if frame_meta.num_obj_meta > 0:
            if _face_recognizer is not None:
                _needs_pixel = True
            if not _needs_pixel and _appearance_worker is not None:
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
            except Exception as e:
                if frame_num % 30 == 0:
                    logger.warning("get_nvds_buf_surface fallo frame=%d: %s", frame_num, e)

        # ── Separar personas y caras ──────────────────────────────────────────
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

        # ── Face recognition: procesar detecciones de cara ───────────────────
        if _face_handler and face_metas and frame_np is not None:
            for face_obj_meta in face_metas:
                _face_handler.process_face(
                    face_obj_meta, frame_num, frame_np, persons_meta, camera_id
                )

        # ── Tracks de personas ────────────────────────────────────────────────
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

            # OSD base label — número corto de display si el ReID ya resolvió, "..." si espera.
            # _display_ids es solo para el stream; no toca global_id ni los payloads de API.
            state = _active_tracks[track_key]
            if state.global_id:
                global _display_id_counter
                if state.global_id not in _display_ids:
                    _display_id_counter += 1
                    _display_ids[state.global_id] = _display_id_counter
                base_label = f"#{_display_ids[state.global_id]}"
            else:
                base_label = "..."
            _set_osd_text(obj_meta, base_label, border_color=(0.2, 0.6, 1.0, 1.0))

            # ── Handler dispatch ──────────────────────────────────────────────
            for handler in _active_handlers:
                result = handler.process(obj_meta, frame_num, frame_np=frame_np)
                if result is None:
                    continue
                if result.osd_text:
                    _set_osd_text(obj_meta, result.osd_text, border_color=result.border_color)
                if result.event_type == "person_classified":
                    demo = result.det_extra.get("demographics", {})
                    gd, ad = _AGE_GENDER_LABEL_MAP.get(demo.get("label", ""), ("?", "?"))
                    _slog(
                        f"{_C.get('cyan', '')}[{camera_id}]{_C.get('reset', '')} ",
                        f"{_C.get('magenta', '')}DEMOGRAFÍA{_C.get('reset', '')} ",
                        f"track={p_track_id:<4} ",
                        f"{gd} | {ad}  conf={demo.get('confidence', 0.0):.0%}",
                    )
                    api_client.post_person_classified(
                        camera_id, p_track_id, bbox, demo
                    )
                    an = _get_analytics(pad_index)
                    au = result.analytics_update
                    if "age_gender_classes" in au:
                        lbl = au["age_gender_classes"]
                        an["age_gender_classes"][lbl] = an["age_gender_classes"].get(lbl, 0) + 1
                    if "gender_key" in au:
                        an[au["gender_key"]] += 1

            # ── Face identity overlay ─────────────────────────────────────────
            if _face_handler:
                identity = _face_handler.get_identity(p_track_id)
                if identity:
                    name, conf = identity
                    if name != "Desconocido":
                        cur = str(obj_meta.text_params.display_text) or "..."
                        _set_osd_text(
                            obj_meta,
                            f"{cur} | {name} {conf:.0%}",
                            border_color=(0.2, 1.0, 0.4, 1.0),
                        )

            # ── Crop capture ──────────────────────────────────────────────────
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

            # Guardar label final en _track_labels para que Probe B lo lea en el tiler.
            # str() es necesario: display_text en pyds es un tipo ctypes, no Python str.
            if _IS_STREAM_ENABLED:
                raw = obj_meta.text_params.display_text
                _track_labels[p_track_id] = {
                    "label": str(raw) if raw else f"P#{p_track_id}",
                    "fall":  False,
                }

        # ── Expirar tracks perdidos ───────────────────────────────────────────
        _expire_lost_tracks(pad_index, frame_num, visible_ids)

        # ── Posiciones para heatmaps ──────────────────────────────────────────
        if _ws_client is not None and visible_ids and frame_np is not None:
            fh, fw = frame_np.shape[:2]
            _accumulate_positions(pad_index, camera_id, persons_meta, fw, fh, frame_num)

        # ── Analytics snapshot periódico ──────────────────────────────────────
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

        # ── Frame de referencia: retry hasta confirmar + detección de cambio ──
        # Solo se evalúa cuando la escena está vacía y tiene suficiente iluminación.
        # Frames nocturnos o con cámara tapada se descartan para no enviar un fondo negro.
        if (frame_np is not None
                and len(visible_ids) == 0
                and frame_meta.num_obj_meta == 0
                and _frame_is_bright_enough(frame_np)):
            _ref_confirmed_np = _reference_frame_confirmed_np.get(camera_id)
            _ref_confirmed_ts = _reference_frame_confirmed_ts.get(camera_id, 0.0)
            _ref_last_attempt = _reference_frame_last_attempt.get(pad_index, 0.0)

            # Caso 1 — nunca confirmado: reintentar cada REFERENCE_FRAME_RETRY_SECS.
            _needs_initial = (
                _ref_confirmed_np is None
                and now - _ref_last_attempt >= REFERENCE_FRAME_RETRY_SECS
            )
            # Caso 2 — ya confirmado: verificar si la escena cambió significativamente
            # y han pasado al menos 24 h desde el último frame confirmado.
            _needs_update = (
                _ref_confirmed_np is not None
                and now - _ref_confirmed_ts >= REFERENCE_FRAME_MIN_INTERVAL_SECS
                and _scene_changed(frame_np, _ref_confirmed_np)
            )

            if _needs_initial or _needs_update:
                fh, fw = frame_np.shape[:2]
                _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_b64 = base64.b64encode(buf).decode("utf-8")
                api_client.post_reference_frame(camera_id, frame_num, frame_b64, fw, fh)
                _reference_frame_last_attempt[pad_index] = now
                reason = "inicial/retry" if _needs_initial else "cambio visual detectado"
                logger.info(
                    "Frame de referencia encolado camera=%s frame=%d %dx%d [%s]",
                    camera_id, frame_num, fw, fh, reason,
                )

    return Gst.PadProbeReturn.OK
