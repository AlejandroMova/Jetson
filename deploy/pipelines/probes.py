"""
GStreamer probe for DeepStream metadata extraction and REST API delivery.

Imported by app.py and app_video_testing.py.

Architecture:
  PGIE (PeopleNet, gie-id=1) detects person/bag/face in the full frame.
  Optional handlers (one per active capability) process each detected person.
  FaceRecognizer runs on a background thread (queue+thread pattern).
  OSNet embeddings are read synchronously from the SGIE (gie-id=3) via NvDsInferTensorMeta.

Stream mode (NX_STREAM_ENABLED=true):
  The pipeline inserts nvmultistreamtiler (640x360) after the analytics probe.
  A second probe (tiled_overlay_probe) draws bboxes on the tiled frame and serves
  it via MjpegServer (:8080/viewer/all). Zero overhead in production.

How to add a new model:
  1. Create a class implementing process(obj_meta, frame_num, frame_np).
  2. Add it to _HANDLER_REGISTRY under its capability name.
  3. If it needs an async worker, add it in init_workers() and wire it in init_handlers().
  4. Add the entry to SGIE_CONFIGS in app.py (or None if it's a Python worker).
"""
import base64
import ctypes
import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
import pyds
import requests

logger = logging.getLogger(__name__)


# ── Environment config ────────────────────────────────────────────────────────

JETSON_ID: str = os.environ.get("JETSON_ID", os.uname().nodename)
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY: str = os.environ.get("API_KEY", "your-api-key")


def _get_tailscale_ip() -> Optional[str]:
    """Return the Tailscale IPv4 address of this device, or None if unavailable."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=2,
        )
        ip = result.stdout.strip()
        return ip if ip else None
    except Exception:
        # Tailscale not installed or not running — non-fatal, field is optional.
        return None


TAILSCALE_IP: Optional[str] = _get_tailscale_ip()

# pad_index → real DVR channel number.
# Populated from app.py by calling init_channel_map(cfg.channels).
_channel_map: Dict[int, int] = {}

_JETSON_SECTOR: str = "comercio"

# Pad indices for entry/exit cameras — used to tag person_entry/exit events.
_entry_exit_pads: set = set()

# Camera type classification — controls which cameras contribute to analytics counts.
_external_pads: set = set()
_count_internal: bool = True
_count_external: bool = True


def init_channel_map(channels: list) -> None:
    """Call from app.py after load_config(), before starting the pipeline."""
    global _channel_map
    _channel_map = {idx: ch for idx, ch in enumerate(channels)}
    logger.info("Channel map: %s", _channel_map)


def init_sector(sector: str) -> None:
    """Set the client sector: 'comercio', 'industrial', or 'hogar'.

    The sector controls the event type emitted by face_recognition
    (employee_seen vs known_person_seen).
    """
    global _JETSON_SECTOR
    _JETSON_SECTOR = sector
    logger.info("Sector: %s", sector)


def init_entry_exit_pads(pad_indices: set) -> None:
    """Define which pad indices correspond to entry/exit cameras."""
    global _entry_exit_pads
    _entry_exit_pads = pad_indices
    logger.info("Entry/exit pad indices: %s", pad_indices)


def init_camera_types(
    external_pad_indices: set, count_internal: bool, count_external: bool
) -> None:
    """Configure which cameras are external and whether to count their detections."""
    global _external_pads, _count_internal, _count_external
    _external_pads = external_pad_indices
    _count_internal = count_internal
    _count_external = count_external
    logger.info(
        "Camera types — external pads: %s  count_internal=%s  count_external=%s",
        external_pad_indices, count_internal, count_external,
    )


def _camera_id_for(pad_index: int) -> str:
    """Return a camera_id using the real DVR channel number, e.g. 'jetson-nx-001-ch03'."""
    ch = _channel_map.get(pad_index, pad_index)
    return f"{JETSON_ID}-ch{ch:02d}"


# ── GIE unique-ids ────────────────────────────────────────────────────────────
PGIE_UNIQUE_ID: int = 1      # PeopleNet
SGIE_AGE_GENDER_ID: int = 2  # ResNet-18 Pedestrian Attr
OSNET_GIE_ID: int = 3        # OSNet-x1.0 appearance SGIE — 512-dim embedding per person

# ── Partial-body ReID floor ──────────────────────────────────────────────────
# Standing person bbox: height/width ≈ 3–4. Below this ratio only legs/feet or a
# small sliver of torso are visible (camera-edge crop or heavy occlusion) — too
# degraded to be worth even attempting a match. Skip and wait for a better view.
# Removed 2026-07-08: a separate, lower similarity threshold for the mid-range
# (ratio 1.3-1.8) used to exist (PARTIAL_BODY_REID_THRESHOLD=0.64) — calibration
# against real DEMOONE crops showed it merged different people just as often as
# the original 0.68 full-body threshold did. Every detection above this floor
# now goes through the same SIMILARITY_THRESHOLD as full-body views; a partial
# view that doesn't match yet just extends the deadline (see below) instead of
# being accepted on a lower bar.
PARTIAL_BODY_MIN_RATIO: float = 1.3

# ── PGIE class ids ────────────────────────────────────────────────────────────
PGIE_CLASS_PERSON: int = 0
PGIE_CLASS_BAG: int = 1
PGIE_CLASS_FACE: int = 2

# ── Confidence thresholds ─────────────────────────────────────────────────────
OSD_CONFIDENCE_THRESHOLD: float = 0.30
MIN_CLASSIFICATION_PROB: float = 0.3
FACE_DET_CONFIDENCE_THRESHOLD: float = 0.40

# ── Age/gender voting ─────────────────────────────────────────────────────────
VOTE_SAMPLE_INTERVAL: int = 5
VOTES_REQUIRED: int = 10
VOTE_MIN_WIDTH: int = 64
VOTE_MIN_HEIGHT: int = 160

# ── Track lifecycle ───────────────────────────────────────────────────────────
TRACK_LOST_TIMEOUT_FRAMES: int = 60

# ── Face recognition ──────────────────────────────────────────────────────────
FACE_SAMPLE_INTERVAL: int = 30

# Persistent per-client CSV log of every face-recognition sample (match or unknown),
# for offline threshold/precision analysis. Independent of NX_STREAM_ENABLED — unlike
# the console _slog lines, this always writes in production.
# Columns (no header row in the file — see init_workers()):
#   timestamp,camera_id,track_id,global_id,identity,similarity,status
FACE_LOG_MAX_BYTES: int = 20 * 1024 * 1024  # 20 MB per file
FACE_LOG_BACKUP_COUNT: int = 5              # ~100 MB max on disk, oldest rotated out

# ── Analytics ─────────────────────────────────────────────────────────────────
ANALYTICS_SEND_INTERVAL_SECS: float = 60.0

# ── Reference frame ───────────────────────────────────────────────────────────
# Minimum time between retries when the backend has not confirmed the frame yet.
REFERENCE_FRAME_RETRY_SECS: float = 30.0
# Minimum 24h between resends — aligned with the frontend calendar's day granularity.
REFERENCE_FRAME_MIN_INTERVAL_SECS: float = 86_400.0
# Normalized diff fraction (0.0-1.0) that triggers a new reference frame send.
# 0.15 ≈ 15% of pixels changed significantly after normalizing for mean illumination.
REFERENCE_FRAME_CHANGE_THRESHOLD: float = 0.15
# Minimum acceptable brightness (mean pixel value in grayscale, 0-255).
# Frames below this value (night, covered camera) are discarded as background.
# 30/255 ≈ 12% of max brightness — rejects pure black and near-dark scenes.
REFERENCE_FRAME_MIN_BRIGHTNESS: float = 30.0

# we wait until we have certain frames without a person, this to prevent that we 
# consider a frame "empty" from people, when in reality there was a lack of detection 
# for a few frames
# consider FPS
MIN_REFERENCE_FRAME_SPACE: int = 30
# we initialize the frame space, from 0 and we add them up every frame that there are no people
# once the frame is sent, restart from 0 
CURRENT_FRAME_SPACE: dict[str, int] = {}


# ── Crop capture ──────────────────────────────────────────────────────────────
CROPS_DIR: str = "crops"
CROP_SAMPLE_INTERVAL: int = 15
CROP_MAX_PER_PERSON: int = 5
CROP_MIN_WIDTH: int = 48
CROP_MIN_HEIGHT: int = 96
# Frames to wait for an appearance embedding before emitting person_entry anyway.
# 30 frames ≈ 1 second — enough for OSNet on CPU even under queue pressure.
ENTRY_EMIT_DEADLINE_FRAMES: int = 30


# ── Stream mode — active only when NX_STREAM_ENABLED=true ────────────────────

_IS_STREAM_ENABLED: bool = os.getenv("NX_STREAM_ENABLED", "false").lower() == "true"

# ANSI color codes for stream logs. Disable with NO_COLOR=1 (useful for grepping
# docker logs without escape sequences). Empty dict = no-op for _C.get() calls.
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
    """Print a colored log line to stdout when stream mode is active.

    Visible in `docker logs -f`. flush=True is required for immediate output
    without Docker buffering. Zero output in production (NX_STREAM_ENABLED=false).
    """
    if _IS_STREAM_ENABLED:
        print("".join(parts) + _C.get("reset", ""), flush=True)


# Accumulator for the periodic analytics_snapshot summary line.
# _send() runs in the NxApiClient worker thread — the lock guards concurrent access.
_analytics_slog_cameras: list = []
_analytics_slog_last_t: Optional[float] = None  # None = timer not started; starts on first call
_ANALYTICS_SLOG_INTERVAL: float = 60.0
_analytics_slog_lock = threading.Lock()


def _accumulate_analytics_slog(camera_id: str) -> None:
    """Accumulate cameras with a successful analytics_snapshot and emit a summary line every 60s.

    The timer starts on the first real call (not at import time) to avoid being
    skewed by model load time. The first line fires ~60s after the pipeline starts,
    with all cameras already accumulated.
    """
    global _analytics_slog_last_t
    cams_to_log = None
    with _analytics_slog_lock:
        if camera_id not in _analytics_slog_cameras:
            _analytics_slog_cameras.append(camera_id)
        now = time.monotonic()
        if _analytics_slog_last_t is None:
            # First call — start the timer without flushing yet.
            _analytics_slog_last_t = now
        elif now - _analytics_slog_last_t >= _ANALYTICS_SLOG_INTERVAL:
            _analytics_slog_last_t = now
            cams_to_log = list(_analytics_slog_cameras)
            _analytics_slog_cameras.clear()
    if cams_to_log is not None:
        _slog(
            f"{_C.get('yellow', '')}[API]{_C.get('reset', '')} ",
            f"analytics_snapshot  ",
            f"{_C.get('cyan', '')}{cams_to_log}{_C.get('reset', '')}  200",
        )


# Tiled frame queue for MjpegServer, populated by tiled_overlay_probe.
tiled_frame_queue: queue.Queue = queue.Queue(maxsize=1)

# Tiler grid dimensions — set by init_stream_grid() from app.py before pipeline start.
_stream_tiler_cols: int = 1
_stream_tiler_rows: int = 1
_stream_cell_w: int = 640
_stream_cell_h: int = 360

# Track labels written by Probe A (osd_sink_pad_buffer_probe) and read by Probe B
# (tiled_overlay_probe). Safe: both probes run on the same GStreamer thread.
_track_labels: dict = {}  # track_id → {"label": str, "fall": bool}

# Display ID mapping: global_id (hex12) → short display number (1, 2, 3...).
# Stream-only — does not affect global_id or API payloads.
_display_ids: dict[str, int] = {}
_display_id_counter: int = 0


def init_stream_grid(cols: int, rows: int, cell_w: int, cell_h: int) -> None:
    """Set tiler grid dimensions. Call from app.py after creating the tiler element."""
    global _stream_tiler_cols, _stream_tiler_rows, _stream_cell_w, _stream_cell_h
    _stream_tiler_cols = cols
    _stream_tiler_rows = rows
    _stream_cell_w = cell_w
    _stream_cell_h = cell_h
    logger.info("[Stream] Grid: %dx%d tiles of %dx%d px", cols, rows, cell_w, cell_h)


def _draw_tiled_overlays(frame_bgr: np.ndarray, tracks: list) -> None:
    """Draw bboxes and labels on frame_bgr in-place (coordinates already in tiled space).

    tracks: list of {"bbox_tiled": (x, y, w, h), "label": str, "fall": bool}.
    """
    for t in tracks:
        x1, y1, w, h = t["bbox_tiled"]
        x2 = min(frame_bgr.shape[1] - 1, x1 + w)
        y2 = min(frame_bgr.shape[0] - 1, y1 + h)
        x1 = max(0, x1)
        y1 = max(0, y1)
        if x2 <= x1 or y2 <= y1:
            continue
        if t.get("fall"):
            color = (0, 0, 230)  # red
        elif t.get("face"):
            color = (0, 200, 255)  # orange — distinguishes face boxes from person boxes
        else:
            color = (0, 210, 0)  # green
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = t["label"]
        txt_y = max(y1 - 3, 12)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame_bgr, (x1, txt_y - th - 2), (x1 + tw, txt_y + 1), color, -1)
        cv2.putText(frame_bgr, label, (x1, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def tiled_overlay_probe(_pad, info):
    """Probe B — attached to the nvmultistreamtiler src pad (stream mode only).

    Receives the composited 640x360 RGBA frame. Reads _track_labels for labels
    computed by Probe A, maps each bbox to tiled space using the grid geometry,
    draws bboxes+labels, and pushes the frame to tiled_frame_queue for MjpegServer.
    """
    gst_buffer = info.get_buffer()
    if gst_buffer is None:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if batch_meta is None:
        return Gst.PadProbeReturn.OK

    cols = _stream_tiler_cols
    rows = _stream_tiler_rows
    cw = _stream_cell_w
    ch = _stream_cell_h

    # The tiler produces a single composited frame (batch_id=0).
    frame_meta = None
    for fm in _iter_pyds_list(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):
        frame_meta = fm
        break
    if frame_meta is None:
        return Gst.PadProbeReturn.OK

    try:
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_bgr = cv2.cvtColor(np.array(n_frame, copy=True, order='C'), cv2.COLOR_RGBA2BGR)
    except Exception:
        # GPU surface read can fail transiently — drop the frame rather than crash.
        return Gst.PadProbeReturn.OK

    overlay_tracks = []
    seen_face_ids: Set[int] = set()
    for obj_meta in _iter_pyds_list(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast):
        cls = int(obj_meta.class_id)
        if cls not in (PGIE_CLASS_PERSON, PGIE_CLASS_FACE):
            continue
        r = obj_meta.rect_params
        # Map bbox center to tile cell to identify the original stream.
        cx = int(r.left + r.width / 2)
        cy = int(r.top + r.height / 2)
        tile_col = min(cx // cw, cols - 1)  # noqa: F841 — reserved for future per-stream logic
        tile_row = min(cy // ch, rows - 1)  # noqa: F841
        track_id = int(obj_meta.object_id)
        if cls == PGIE_CLASS_FACE:
            seen_face_ids.add(track_id)
        info_dict = _track_labels.get(track_id, {})
        default_label = f"F#{track_id}" if cls == PGIE_CLASS_FACE else f"P#{track_id}"
        label = info_dict.get("label") or default_label
        overlay_tracks.append({
            "bbox_tiled": (int(r.left), int(r.top), int(r.width), int(r.height)),
            "label": label,
            "fall": info_dict.get("fall", False),
            "face": info_dict.get("face", cls == PGIE_CLASS_FACE),
        })

    if overlay_tracks:
        _draw_tiled_overlays(frame_bgr, overlay_tracks)

    # Face tracks have no _active_tracks/_expire_lost_tracks equivalent, so
    # prune labels for faces no longer present this frame here instead —
    # otherwise _track_labels grows unbounded over a long stream session.
    stale_face_ids = [
        tid for tid, info in _track_labels.items()
        if info.get("face") and tid not in seen_face_ids
    ]
    for tid in stale_face_ids:
        _track_labels.pop(tid, None)

    try:
        tiled_frame_queue.put_nowait(frame_bgr)
    except queue.Full:
        pass  # MjpegServer did not consume in time — drop the older frame, not the new one.

    return Gst.PadProbeReturn.OK


# ── Track state ───────────────────────────────────────────────────────────────

@dataclass
class _TrackState:
    """Per-track state. Lives in _active_tracks[track_key] from first frame to exit.

    The ReID fields (entry_emitted, entry_deadline, global_id, pending_bbox,
    pending_conf) are only populated when _reid_manager is active (OSNet found).
    """
    first_frame: int
    last_frame: int
    first_ts: float
    camera_id: str
    is_entry_exit_cam: bool = False
    appearance_sent: bool = False
    entry_emitted: bool = False
    entry_deadline: int = 0
    global_id: Optional[str] = None
    pending_bbox: Optional[dict] = None
    pending_conf: float = 0.0


# ── GStreamer bus handler ─────────────────────────────────────────────────────

def bus_call(_bus, message, loop):
    """Handle GStreamer bus messages: EOS for clean shutdown, WARNING and ERROR for logging."""
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End of video stream (EOS).")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("GStreamer WARNING: %s — %s", err, debug)
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("GStreamer ERROR: %s — %s", err, debug)
        loop.quit()
    return True


# ── Non-blocking REST API client ──────────────────────────────────────────────

class NxApiClient:
    """Send HTTP requests to the backend on a background thread.

    The GStreamer probe only calls enqueue() (O(1), no I/O), guaranteeing
    that network calls never affect pipeline FPS.
    """

    def __init__(self, base_url: str, api_key: str, max_queue_size: int = 512):
        """Set up the client with HTTP keep-alive and a FIFO request queue."""
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
        # Callbacks invoked from the worker thread after a successful 2xx response.
        # Key: exact endpoint path, e.g. "/api/cameras/reference-frame".
        self._success_callbacks: Dict[str, "Callable[[dict], None]"] = {}

    def start(self) -> None:
        """Start the worker thread that drains the HTTP request queue."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="nx-api-worker",
        )
        self._worker_thread.start()
        logger.info("NxApiClient started → %s", self._base_url)

    def stop(self) -> None:
        """Signal the worker to stop, wait up to 5s, then close the HTTP session."""
        self._running = False
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._session.close()
        logger.info("NxApiClient stopped.")

    def register_success_callback(self, endpoint: str, cb: "Callable[[dict], None]") -> None:
        """Register a callback invoked by the worker thread when an endpoint returns 2xx.

        The callback receives the original payload sent. It runs from the worker thread
        and must be thread-safe and non-blocking.

        Args:
            endpoint: Exact path, e.g. "/api/cameras/reference-frame".
            cb: Function that accepts the sent payload dict.
        """
        self._success_callbacks[endpoint] = cb

    def enqueue(self, method: str, endpoint: str, payload: Optional[dict] = None) -> None:
        """Enqueue an HTTP request without blocking. Drops with a warning if the queue is full."""
        try:
            self._queue.put_nowait((method, endpoint, payload))
        except queue.Full:
            logger.warning("API queue full — dropping: %s %s", method, endpoint)

    def _worker_loop(self) -> None:
        """Drain the request queue and send HTTP requests to the backend."""
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

    def _send(self, method: str, endpoint: str, payload: Optional[dict]) -> None:
        """Send one HTTP request. 5s timeout — errors are logged, never propagated."""
        url = f"{self._base_url}{endpoint}"
        try:
            resp = self._session.request(method=method, url=url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.debug("%s %s → %d", method, endpoint, resp.status_code)
            if endpoint == "/api/analytics":
                # Analytics snapshots are one per camera per 60s — group into a summary line.
                _accumulate_analytics_slog((payload or {}).get("camera_id", "?"))
            else:
                _slog(
                    f"{_C.get('yellow', '')}[API]{_C.get('reset', '')} ",
                    f"{method} {endpoint}  ",
                    f"{_C.get('bold', '')}{resp.status_code}{_C.get('reset', '')}",
                )
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
            logger.debug("No connection: %s", url)

    def _base_event(self, event_type: str, camera_id: str, severity: str = "info") -> dict:
        """Build the fields common to all backend events."""
        return {
            "event_id": str(uuid.uuid4()),
            "type": event_type,
            "sector": _JETSON_SECTOR,
            "jetson_id": JETSON_ID,
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
        }

    def post_person_entry(self, camera_id: str, track_id: int, bbox: dict,
                          confidence: float, is_entry_exit_cam: bool,
                          global_id: Optional[str] = None,
                          is_return: bool = False) -> None:
        """Emit person_entry. entry_type='return' if the person was seen before."""
        payload = self._base_event("person_entry", camera_id)
        payload.update({
            "track_id": track_id,
            "bbox": bbox,
            "confidence": round(confidence, 3),
            "is_entry_exit_camera": is_entry_exit_cam,
            "entry_type": "return" if is_return else "new",
        })
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_channel_change(self, camera_id: str, track_id: int, bbox: dict,
                                   confidence: float, global_id: str,
                                   prev_camera_id: Optional[str],
                                   is_entry_exit_cam: bool) -> None:
        """Emit person_channel_change when the same person switches cameras (ReID)."""
        payload = self._base_event("person_channel_change", camera_id)
        payload.update({
            "track_id": track_id,
            "bbox": bbox,
            "confidence": round(confidence, 3),
            "global_id": global_id,
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        if prev_camera_id:
            payload["prev_camera_id"] = prev_camera_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_exit(self, camera_id: str, track_id: int,
                         dwell_seconds: float, is_entry_exit_cam: bool,
                         global_id: Optional[str] = None) -> None:
        """Emit person_exit with the total dwell time for the track."""
        payload = self._base_event("person_exit", camera_id)
        payload.update({
            "track_id": track_id,
            "dwell_seconds": round(dwell_seconds, 1),
            "is_entry_exit_camera": is_entry_exit_cam,
        })
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/events", payload)

    def post_person_classified(self, camera_id: str, global_id: str,
                               bbox: dict, demographics: dict) -> None:
        """Emit person_classified with age/gender result (after VOTES_REQUIRED votes)."""
        payload = self._base_event("person_classified", camera_id)
        payload.update({"global_id": global_id, "bbox": bbox, "demographics": demographics})
        self.enqueue("POST", "/api/events", payload)

    def post_person_appearance(self, camera_id: str, track_id: int,
                               appearance_vector: list) -> None:
        """Emit person_appearance with the L2-normalized 512-dim OSNet vector."""
        payload = self._base_event("person_appearance", camera_id)
        payload.update({"track_id": track_id, "appearance_vector": appearance_vector})
        self.enqueue("POST", "/api/events", payload)

    # post_employee_seen/presence/exit removed — employee identity now rides
    # along in positions_snapshot (see WsPositionClient.send_positions and
    # _accumulate_positions below) instead of discrete REST events.

    def post_unknown_person_alert(self, camera_id: str, track_id: int,
                                  face_snapshot_b64: str, bbox: dict) -> None:
        """Emit unknown_person_alert (hogar sector) on an unrecognized face detection."""
        payload = self._base_event("unknown_person_alert", camera_id, "medium")
        payload.update({
            "track_id": track_id,
            "bbox": bbox,
            "face_snapshot_b64": face_snapshot_b64,
        })
        self.enqueue("POST", "/api/events", payload)

    def post_analytics_snapshot(self, camera_id: str, stats: dict,
                                period_seconds: float = 60.0) -> None:
        """Emit analytics_snapshot every ANALYTICS_SEND_INTERVAL_SECS with accumulated counts."""
        payload = self._base_event("analytics_snapshot", camera_id)
        payload.update({"period_seconds": period_seconds, **stats})
        self.enqueue("POST", "/api/analytics", payload)

    def post_crop(self, camera_id: str, track_id: int, frame_num: int,
                  crop_b64: str, bbox: dict, global_id: Optional[str] = None) -> None:
        """Send a base64-encoded person crop to /api/crops.

        global_id is the cross-camera ReID identity for this track, if
        already resolved when the crop was captured (None otherwise —
        crops taken before ReID resolves still ship, just without it)."""
        payload = {
            "camera_id": camera_id,
            "jetson_id": JETSON_ID,
            "track_id": track_id,
            "frame_num": frame_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_b64": crop_b64,
            "bbox": bbox,
        }
        if global_id:
            payload["global_id"] = global_id
        self.enqueue("POST", "/api/crops", payload)

    def post_reference_frame(self, camera_id: str, frame_num: int,
                             frame_b64: str, width: int, height: int) -> None:
        """Send a reference frame (empty scene) per camera to the backend."""
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


# Global instance — initialized in main() before starting the pipeline.
api_client = NxApiClient(base_url=API_BASE_URL, api_key=API_KEY)


# ── DeepStream metadata helpers ───────────────────────────────────────────────

def _iter_pyds_list(pyds_list, cast_fn):
    """Safely iterate over a pyds linked list."""
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


# Pedestrian Attr model outputs 6 classes — maps raw label → (display gender, display age group).
_AGE_GENDER_LABEL_MAP: Dict[str, Tuple[str, str]] = {
    "female_adult":  ("Mujer",  "Adulta"),
    "female_senior": ("Mujer",  "Mayor"),
    "female_young":  ("Mujer",  "Joven"),
    "male_adult":    ("Hombre", "Adulto"),
    "male_senior":   ("Hombre", "Mayor"),
    "male_young":    ("Hombre", "Joven"),
}

# Same mapping for API payloads.
_AGE_GENDER_API_MAP: Dict[str, Tuple[str, str]] = {
    "female_adult":  ("female", "adult"),
    "female_senior": ("female", "senior"),
    "female_young":  ("female", "young"),
    "male_adult":    ("male",   "adult"),
    "male_senior":   ("male",   "senior"),
    "male_young":    ("male",   "young"),
}


def _parse_age_gender(classifier_meta) -> Tuple[str, str, str, float]:
    """Extract the label and probability from the ResNet-18 SGIE classifier metadata.

    Returns:
        Tuple of (raw_label, gender_display, age_display, prob).
        Returns ("", "", "", 0.0) if the model has not yet run inference.
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
        logger.debug("Unknown SGIE label: %r (gie-id=%d)", raw,
                     classifier_meta.unique_component_id)
    return "", "", "", 0.0


def _set_osd_text(
    obj_meta,
    text: str,
    border_color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 1.0),
) -> None:
    """Apply text and style to the OSD overlay for a detected object."""
    obj_meta.text_params.display_text = text

    fp = obj_meta.text_params.font_params
    fp.font_name = "Sans"
    fp.font_size = 12
    fp.font_color.set(1.0, 1.0, 1.0, 1.0)

    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

    obj_meta.rect_params.border_color.set(*border_color)
    obj_meta.rect_params.border_width = 2


# ── Pipeline handlers ─────────────────────────────────────────────────────────

class _HandlerResult:
    """Value returned by a handler to the probe for OSD update and API dispatch."""
    __slots__ = ("osd_text", "border_color", "event_type", "det_extra", "analytics_update")

    def __init__(
        self,
        osd_text: str = "",
        border_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        event_type: str = "",
        det_extra: Optional[dict] = None,
        analytics_update: Optional[dict] = None,
    ):
        """event_type empty means no additional API event is emitted."""
        self.osd_text = osd_text
        self.border_color = border_color
        self.event_type = event_type
        self.det_extra = det_extra or {}
        self.analytics_update = analytics_update or {}


class _AgeGenderHandler:
    """Classify gender and age group by voting over the ResNet-18 SGIE.

    Accumulates VOTES_REQUIRED samples before locking the result and emitting
    person_classified. Voting reduces false positives from single-frame noise.
    """

    def __init__(self):
        self._cache: Dict[int, Tuple[str, str, str, float]] = {}
        self._votes: Dict[int, List[str]] = {}
        self._vote_last_frame: Dict[int, int] = {}

    def process(self, obj_meta, frame_num: int, frame_np=None) -> Optional[_HandlerResult]:
        """Accumulate SGIE votes and return a HandlerResult once enough votes are in."""
        p_track_id = int(obj_meta.object_id)
        r = obj_meta.rect_params

        # Persons too small in frame produce noisy classifications — skip until large enough.
        if int(r.width) < VOTE_MIN_WIDTH or int(r.height) < VOTE_MIN_HEIGHT:
            if p_track_id in self._cache:
                raw, gender_disp, age_disp, prob = self._cache[p_track_id]
                prefix = str(obj_meta.text_params.display_text) or "..."
                return _HandlerResult(
                    osd_text=f"{prefix} | {gender_disp} | {age_disp} {prob:.0%}",
                    border_color=(0.0, 1.0, 0.0, 1.0),
                )
            return None

        if p_track_id in self._cache:
            raw, gender_disp, age_disp, prob = self._cache[p_track_id]
            prefix = str(obj_meta.text_params.display_text) or "..."
            return _HandlerResult(
                osd_text=f"{prefix} | {gender_disp} | {age_disp} {prob:.0%}",
                border_color=(0.0, 1.0, 0.0, 1.0),
            )

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
            # Enough votes — lock the winner by majority.
            winner = Counter(votes).most_common(1)[0][0]
            winner_gd, winner_ad = _AGE_GENDER_LABEL_MAP.get(winner, ("?", "?"))
            winner_prob = votes.count(winner) / len(votes)
            self._cache[p_track_id] = (winner, winner_gd, winner_ad, winner_prob)
            gender_api, age_api = _AGE_GENDER_API_MAP.get(winner, ("unknown", "unknown"))
            det_extra = {
                "demographics": {
                    "gender": gender_api,
                    "age_group": age_api,
                    "label": winner,
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
                event_type="person_classified",
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

    Receives face detections from PeopleNet class 2 (face), extracts crops,
    and enqueues them to the worker keyed by global_id (not track_id) — see
    process_face. Employee identity no longer goes out as discrete REST
    events (employee_seen/presence/exit): it rides along in positions_snapshot
    instead, via _employee_by_global_id + _face_confirmed_this_cycle, both
    consumed by _accumulate_positions. Only the hogar unknown_person_alert
    stays as a discrete event, since it's about intrusion detection, not
    employee attendance.

    Not in _HANDLER_REGISTRY — dispatched separately because it processes
    face objects (class_id=2), not the main person loop.
    """

    def __init__(self, worker):
        self._worker = worker
        self._last_sample: Dict[int, int] = {}
        self._cache: Dict[int, Tuple[str, float]] = {}
        # Track ids already logged as EMPLEADO in stream mode — cosmetic
        # dedup only, no REST side effect anymore.
        self._identity_reported: Set[int] = set()
        self._unknown_alerted: Set[int] = set()
        # Tracks that already received a stream log line for an unknown face — one per track.
        self._unknown_face_logged: Set[int] = set()

    def process_face(
        self,
        face_obj_meta,
        frame_num: int,
        frame_np,
        persons_meta: list,
        camera_id: str,
        pad_index: int,
    ) -> None:
        """Process a face detected by PeopleNet (class_id=2): extract crop,
        enqueue to the worker keyed by global_id, and cache the result."""
        if self._worker is None or frame_np is None:
            return
        if face_obj_meta.confidence < FACE_DET_CONFIDENCE_THRESHOLD:
            return

        parent_track_id = self._find_parent_track(face_obj_meta, persons_meta)
        if parent_track_id is None:
            return

        # Don't feed the recognizer until ReID has resolved a global_id for
        # this track — indexing votes by track_id would reset every camera
        # change. The wait is a few frames at most, negligible against the
        # FACE_VOTES_REQUIRED sampling cycle. This also gates the hogar
        # unknown_person_alert path below on Jetsons without OSNet installed
        # — an accepted, documented limitation (see CLAUDE.md).
        state = _active_tracks.get((pad_index, parent_track_id))
        if state is None or state.global_id is None:
            return
        global_id = state.global_id

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
        self._worker.enqueue(face_crop, global_id, frame_num, camera_id)

        result = self._worker.get_result(global_id)
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

        # identity_key is a backend-assigned UUID string (or "Unknown" if below threshold).
        identity_key, conf = identity
        display_name = self._worker.get_display_name(identity_key)

        # Persistent, always-on record of every processed sample (not deduped per
        # track like the console log below) — similarity drift over time is what
        # later threshold tuning needs. identity_key (not display_name) is logged,
        # so rows join directly on employees.id without a name-collision risk.
        if _face_csv_logger is not None:
            _face_csv_logger.info(
                "%s,%s,%s,%s,%.4f,%s",
                camera_id, parent_track_id, global_id, identity_key, conf,
                "matched" if identity_key != "Unknown" else "unknown",
            )

        if identity_key != "Unknown":
            # Tag the global_id (permanent for its lifetime) and mark this
            # camera's current buffering cycle as face-confirmed — consumed
            # by _accumulate_positions to decide employee_id/face_confirmed
            # on the outgoing positions_snapshot entry.
            _employee_by_global_id[global_id] = identity_key
            _face_confirmed_this_cycle.setdefault(pad_index, set()).add(global_id)
            if parent_track_id not in self._identity_reported:
                self._identity_reported.add(parent_track_id)
                _slog(
                    f"{_C.get('cyan', '')}[{camera_id}]{_C.get('reset', '')} ",
                    f"{_C.get('green', '')}{_C.get('bold', '')}EMPLEADO{_C.get('reset', '')}   ",
                    f"track={parent_track_id:<4} ",
                    f"nombre={_C.get('bold', '')}{display_name}{_C.get('reset', '')}  sim={conf:.2f}",
                )
        else:
            # This global_id may have been a tagged employee before — the sliding
            # vote window (or a reload()/revoke sync clearing FaceRecognizer's
            # _locked) can flip the winner back to "Unknown". Drop the stale tag
            # immediately so positions_snapshot stops attributing it to someone
            # it no longer resolves to (no-op if it was never tagged).
            _employee_by_global_id.pop(global_id, None)
            # Log once per track in stream mode.
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

    def on_track_lost(self, track_id: int) -> None:
        """Called from _expire_lost_tracks. Cleans up this handler's local,
        track_id-keyed caches only — _employee_by_global_id and
        FaceRecognizer's own _locked/_votes are keyed by global_id and are
        NOT touched here: the same physical person may still be tracked
        under a different track/camera via ReID continuity. That state is
        only cleared once ReIdManager itself expires the global_id (see
        _handle_appearance_reid's use of match_or_create's expired_ids)."""
        self._cache.pop(track_id, None)
        self._identity_reported.discard(track_id)
        self._unknown_alerted.discard(track_id)

    def get_identity(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Return (name, similarity) if an identity has been recognized, else None.
        Reads only the local cache populated by process_face — the worker
        itself is keyed by global_id, not track_id, so it can't be queried
        directly with a track_id here."""
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


# ── Probe state globals ───────────────────────────────────────────────────────

_active_tracks: Dict[Tuple[int, int], _TrackState] = {}
_crop_counts: Dict[int, int] = {}
_crop_last_frame: Dict[int, int] = {}

# global_id → employee UUID, once face recognition confirms it (see
# _FaceRecognitionHandler.process_face). Cleared when: ReIdManager expires
# that global_id (see _handle_appearance_reid's use of match_or_create's
# expired_ids), or the sliding vote window/a reload() revoke sync flips the
# winner back to "Unknown" (see process_face's else branch).
_employee_by_global_id: Dict[str, str] = {}
# pad_index → set of global_ids that got an actual face crop processed this
# buffering cycle (cleared on each camera's own _accumulate_positions flush,
# same cadence as _position_buffer/_position_last_sent below). Per-camera,
# not global — each camera flushes on its own timer, so a shared set would
# let one camera's flush wipe out another camera's still-pending confirmation.
_face_confirmed_this_cycle: Dict[int, Set[str]] = {}

# Confirmed reference frame (grayscale 64x36 float32) per camera_id.
# None means no 2xx confirmation received from the backend yet.
_reference_frame_confirmed_np: Dict[str, "np.ndarray"] = {}
# Monotonic timestamp of the last backend-confirmed frame per camera_id.
_reference_frame_confirmed_ts: Dict[str, float] = {}
# Monotonic timestamp of the last send attempt per pad_index (controls retry pacing).
_reference_frame_last_attempt: Dict[int, float] = {}

_analytics: Dict[int, Dict] = {}
_analytics_last_sent: Dict[int, float] = {}


def _on_reference_frame_confirmed(payload: dict) -> None:
    """Called by NxApiClient worker thread when the backend confirms a reference frame (2xx).

    Stores the confirmed frame as the baseline for future visual-change detection.
    Runs on the NxApiClient worker thread, not the GStreamer probe thread.

    Args:
        payload: Original payload sent to the backend (includes image_b64 and camera_id).
    """
    cam = payload.get("camera_id", "")
    b64 = payload.get("image_b64", "")
    if not cam or not b64:
        return
    try:
        buf = base64.b64decode(b64)
        arr = np.frombuffer(buf, dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return
        # Resize to 64x36 float32 for fast comparisons with _scene_changed().
        small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        _reference_frame_confirmed_np[cam] = small
        _reference_frame_confirmed_ts[cam] = time.monotonic()
        logger.info("Reference frame confirmed by backend: camera=%s", cam)
    except Exception as exc:
        logger.warning("Error processing reference frame confirmation camera=%s: %s", cam, exc)


# ── Async worker globals + lifecycle ──────────────────────────────────────────

_face_recognizer = None     # FaceRecognizer (InsightFace ArcFace)
_face_csv_logger: Optional[logging.Logger] = None  # always-on persistent face-recognition log
_reid_manager = None        # ReIdManager — local cross-camera identity DB
_ws_client = None           # WsPositionClient (position/heatmap WebSocket)
_jetson_sync_client = None  # JetsonSyncClient (Socket.IO face roster sync)

# Position buffer: pad_index → {global_id → latest position entry}.
# The inner dict is keyed by global_id (12-char hex from ReIdManager) so each person
# contributes exactly one entry per snapshot regardless of how many frames they appear in.
# The backend compares consecutive snapshot timestamps to calculate dwell per person.
_position_buffer: Dict[int, Dict[str, dict]] = {}
_position_last_sent: Dict[int, float] = {}
# 1-second flush gives 1-second timestamp resolution for dwell tracking.
# At a 5-second threshold, a person needs ~5 consecutive snapshots in the same cell.
POSITION_SEND_INTERVAL: float = 1.0


def init_workers(
    pipeline_capabilities: List[str],
    model_dir: str,
    face_db_path: str = "",
    ws_base_url: str = "",
    api_key: str = "",
    reid_gallery_size: int = 10,
) -> None:
    """Instantiate async workers based on active pipeline capabilities.

    Workers initialized per capability:
      - ReIdManager: always (if OSNet ONNX exists on disk — SGIE handles the inference)
      - FaceRecognizer: if 'face_recognition' is in pipeline_capabilities
      - WsPositionClient: if ws_base_url is configured (heatmaps)
      - JetsonSyncClient: if face_recognition is active and API_BASE_URL is set
    """
    global _face_recognizer, _reid_manager, _ws_client, _jetson_sync_client, _face_csv_logger

    osnet_path = str(Path(model_dir) / "osnet" / "osnet_x1_0_market1501.onnx")
    if Path(osnet_path).exists():
        # ReIdManager stores and matches 512-dim embeddings across cameras.
        # The embeddings themselves come from the OSNet SGIE (app.py), not a Python worker.
        from reid_manager import ReIdManager
        reid_db_path = str(Path(model_dir).parent / "reid_db.json")
        # Same clients/<cliente>/logs/ dir as face_recognition.csv, so both analysis
        # logs live together regardless of whether face_recognition is purchased.
        osnet_csv_dir = str(Path(face_db_path).parent / "logs") if face_db_path else None
        _reid_manager = ReIdManager(
            db_path=reid_db_path, gallery_max_size=reid_gallery_size, csv_log_dir=osnet_csv_dir,
        )
        logger.info("ReIdManager active — DB: %s", reid_db_path)
    else:
        logger.warning("OSNet model not found at %s — appearance SGIE and local ReID disabled. "
                       "Run: python3 tools/download_models.py --reid --github-token $TOKEN", osnet_path)

    if "face_recognition" in pipeline_capabilities:
        # Deferred — InsightFace is a large optional dependency not needed in other packages.
        from face_recognizer import FaceRecognizer
        _face_recognizer = FaceRecognizer(
            db_path=face_db_path,
            model_root=str(Path(model_dir) / "insightface"),
            api_base_url=API_BASE_URL,
            api_key=api_key,
        )

        # Persistent CSV log — same client dir as known_faces.json, always-on
        # (unlike the console _slog lines, not gated by NX_STREAM_ENABLED).
        log_dir = Path(face_db_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _face_csv_logger = logging.getLogger("nx.face_csv")
        _face_csv_logger.setLevel(logging.INFO)
        _face_csv_logger.propagate = False  # keep separate from stdout/docker logs
        if not _face_csv_logger.handlers:   # idempotent if init_workers ever re-runs
            csv_handler = RotatingFileHandler(
                log_dir / "face_recognition.csv",
                maxBytes=FACE_LOG_MAX_BYTES, backupCount=FACE_LOG_BACKUP_COUNT,
            )
            csv_handler.setFormatter(logging.Formatter("%(asctime)s,%(message)s"))
            _face_csv_logger.addHandler(csv_handler)

        if API_BASE_URL:
            # Deferred — only needed when face_recognition is active.
            from jetson_sync_client import JetsonSyncClient
            _jetson_sync_client = JetsonSyncClient(
                api_base_url=API_BASE_URL,
                api_key=api_key,
                sync_callback=_face_recognizer.sync_from_backend,
            )
        else:
            logger.info("API_BASE_URL not set — JetsonSyncClient disabled (no face roster push).")

    if ws_base_url:
        # Deferred — WsPositionClient is only needed when heatmap telemetry is configured.
        from ws_client import WsPositionClient
        _ws_client = WsPositionClient(
            ws_url=ws_base_url,
            api_key=api_key,
            sector=_JETSON_SECTOR,
        )
    else:
        logger.info("WS_BASE_URL not set — position WebSocket disabled.")

    api_client.register_success_callback(
        "/api/cameras/reference-frame", _on_reference_frame_confirmed
    )


def start_workers() -> None:
    """Start all workers. Call after pipeline.set_state(PLAYING)."""
    if _face_recognizer is not None:
        _face_recognizer.start()
    if _ws_client is not None:
        _ws_client.start()
    if _jetson_sync_client is not None:
        _jetson_sync_client.start()


def stop_workers() -> None:
    """Stop all workers and persist ReIdManager state to disk."""
    if _jetson_sync_client is not None:
        _jetson_sync_client.stop()
    if _reid_manager is not None:
        _reid_manager.flush()  # write reid_db.json before shutdown
    if _face_recognizer is not None:
        _face_recognizer.stop()
    if _ws_client is not None:
        _ws_client.stop()


# ── Handler registry ──────────────────────────────────────────────────────────

_active_handlers: List = []
_face_handler: Optional[_FaceRecognitionHandler] = None

_HANDLER_REGISTRY = {
    "age_gender": _AgeGenderHandler,
    # face_recognition is NOT here — handled separately via _face_handler
}


def _frame_is_bright_enough(frame_np: "np.ndarray") -> bool:
    """Return True if the frame has enough illumination to use as a background reference.

    Rejects night frames, covered cameras, and near-dark scenes that would produce
    a useless black background for heatmap queries.

    Checks a 64x36 thumbnail (2,304 pixels) rather than the full frame.

    Args:
        frame_np: BGR or grayscale frame from the probe (already in RAM).

    Returns:
        True if mean brightness (0-255) exceeds REFERENCE_FRAME_MIN_BRIGHTNESS.
    """
    gray = frame_np if frame_np.ndim == 2 else cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    return float(small.mean()) >= REFERENCE_FRAME_MIN_BRIGHTNESS


def _scene_changed(current_np: "np.ndarray", prev_np: "np.ndarray") -> bool:
    """Compare the current frame against the last confirmed reference frame.

    Normalizes by mean illumination before comparing, so a lighting change
    (day/night cycle) does not trigger a resend. Only structural changes
    (rearranged products, reorganized zones) exceed the threshold.

    Args:
        current_np: BGR or grayscale full-res frame from the probe.
        prev_np: Last confirmed frame in _reference_frame_confirmed_np
                 — always grayscale, 64x36, float32.

    Returns:
        True if normalized difference exceeds REFERENCE_FRAME_CHANGE_THRESHOLD.
    """
    gray = current_np if current_np.ndim == 2 else cv2.cvtColor(current_np, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Normalize by mean illumination to suppress false positives from lighting changes.
    mean_a = small.mean() or 1.0
    mean_b = prev_np.mean() or 1.0
    diff = np.abs(small / mean_a - prev_np / mean_b).mean()
    return diff > REFERENCE_FRAME_CHANGE_THRESHOLD


def init_handlers(pipeline_capabilities: List[str]) -> None:
    """Instantiate and register active handlers based on pipeline capabilities."""
    global _active_handlers, _face_handler
    _active_handlers = []
    _face_handler = None

    for cap in pipeline_capabilities:
        cls = _HANDLER_REGISTRY.get(cap)
        if cls:
            _active_handlers.append(cls())

        if cap == "face_recognition" and _face_recognizer is not None:
            _face_handler = _FaceRecognitionHandler(_face_recognizer)
            logger.info("FaceRecognitionHandler → FaceRecognizer")

    names = [type(h).__name__ for h in _active_handlers]
    if _face_handler:
        names.append("_FaceRecognitionHandler")
    logger.info("Active handlers: %s", names if names else ["(none — people_counting only)"])


# ── Probe helpers ─────────────────────────────────────────────────────────────

def _get_analytics(pad_index: int) -> Dict:
    """Return (creating if absent) the accumulated analytics dict for a camera."""
    if pad_index not in _analytics:
        _analytics[pad_index] = {
            "person_count": 0, "gender_male": 0,
            "gender_female": 0, "age_gender_classes": {},
        }
    return _analytics[pad_index]


def _get_analytics_last_sent(pad_index: int) -> float:
    """Return the timestamp of the last analytics send for this camera."""
    if pad_index not in _analytics_last_sent:
        _analytics_last_sent[pad_index] = time.monotonic()
    return _analytics_last_sent[pad_index]


def _accumulate_positions(
    pad_index: int, camera_id: str, persons_meta: list, frame_meta,
) -> None:
    """Update the per-camera position buffer and flush to the backend every POSITION_SEND_INTERVAL s.

    Only persons with a resolved global_id are included — tracks without a cross-camera
    identity are skipped so the backend always receives a stable dwell-time key.

    Each person contributes exactly one entry per snapshot (the latest position within
    that second). The backend compares consecutive snapshot timestamps to calculate how
    long each global_id stayed in a given heatmap cell.

    employee_id/face_confirmed ride along per entry so the backend can route
    employee positions to attendance instead of anonymous customer stats — see
    _employee_by_global_id/_face_confirmed_this_cycle (populated by
    _FaceRecognitionHandler.process_face) and app/socket/positions.py on the
    backend.
    """
    fw = frame_meta.source_frame_width
    fh = frame_meta.source_frame_height
    buf = _position_buffer.setdefault(pad_index, {})
    confirmed_this_cycle = _face_confirmed_this_cycle.setdefault(pad_index, set())

    for obj in persons_meta:
        track_key = (pad_index, int(obj.object_id))
        state = _active_tracks.get(track_key)
        if state is None or state.global_id is None:
            # Skip until ReID resolves a stable cross-camera identity.
            continue
        r = obj.rect_params
        x_norm = round((r.left + r.width / 2) / fw, 3)
        y_norm = round((r.top + r.height / 2) / fh, 3)
        # Overwrite any earlier entry for this person within the current second.
        buf[state.global_id] = {
            "global_id": state.global_id, "x_norm": x_norm, "y_norm": y_norm,
            "employee_id": _employee_by_global_id.get(state.global_id),
            "face_confirmed": state.global_id in confirmed_this_cycle,
        }

    now = time.monotonic()
    last = _position_last_sent.get(pad_index, 0.0)
    if now - last >= POSITION_SEND_INTERVAL and buf:
        _ws_client.send_positions(camera_id, list(buf.values()))
        _position_buffer[pad_index] = {}
        _position_last_sent[pad_index] = now
        # New buffering cycle starts for this camera — clear only this
        # camera's confirmed set (per-pad_index, not global — see the
        # docstring on _face_confirmed_this_cycle above).
        _face_confirmed_this_cycle[pad_index] = set()


def _save_and_send_crop(
    crop_bgr: np.ndarray,
    camera_id: str,
    track_id: int,
    frame_num: int,
    bbox: dict,
    global_id: Optional[str] = None,
) -> None:
    """Save a person crop to disk and send it to the backend via API."""
    person_dir = Path(CROPS_DIR) / camera_id / str(track_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    filepath = person_dir / f"frame_{frame_num:06d}.jpg"
    cv2.imwrite(str(filepath), crop_bgr)
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    crop_b64 = base64.b64encode(buf).decode("utf-8")
    api_client.post_crop(camera_id, track_id, frame_num, crop_b64, bbox, global_id=global_id)


def _expire_lost_tracks(pad_index: int, frame_num: int, visible_ids: Set[int]) -> None:
    """Emit person_exit for tracks not seen within TRACK_LOST_TIMEOUT_FRAMES frames."""
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

        # If entry was deferred (ReID) and never emitted, emit it now before the exit.
        if not state.entry_emitted:
            api_client.post_person_entry(
                state.camera_id, track_id,
                state.pending_bbox or {},
                state.pending_conf,
                state.is_entry_exit_cam,
                global_id=state.global_id,
                is_return=False,
            )

        api_client.post_person_exit(
            state.camera_id, track_id, dwell, state.is_entry_exit_cam,
            global_id=state.global_id,
        )
        if _face_handler:
            _face_handler.on_track_lost(track_id)
        _crop_counts.pop(track_id, None)
        _crop_last_frame.pop(track_id, None)
        _track_labels.pop(track_id, None)
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


def _extract_osnet_embedding(obj_meta) -> Optional[np.ndarray]:
    """Read the 512-dim OSNet embedding from the SGIE tensor attached to this object.

    DeepStream attaches one NvDsUserMeta per SGIE to each detected object when
    output-tensor-meta=1 is set in the nvinfer config. Each entry in obj_user_meta_list
    has a meta_type field — NVDSINFER_TENSOR_OUTPUT_META identifies it as nvinfer output.
    We additionally filter by unique_id == OSNET_GIE_ID (3) to distinguish OSNet from
    other SGIEs (AgeGender is gie-id=2 and also attaches tensor metadata).

    Returns an L2-normalized float32 (512,) vector, or None if:
    - The SGIE has not yet processed this object (first frame latency).
    - The bbox is smaller than input-object-min-width/height=96x192 in the nvinfer config.
    """
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        # Check that this user metadata entry is an nvinfer tensor output
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            # Filter by gie-unique-id to get only the OSNet output (not AgeGender)
            if tensor_meta.unique_id == OSNET_GIE_ID:
                # Layer 0 is the single output blob "output" — shape (512,) per object
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                ptr = ctypes.cast(
                    pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float)
                )
                # Copy out of DeepStream-managed memory before the buffer is recycled
                emb = np.ctypeslib.as_array(ptr, shape=(512,)).copy()
                norm = np.linalg.norm(emb)
                return emb / norm if norm > 0 else emb

        try:
            l_user = l_user.next
        except StopIteration:
            break
    return None


def _handle_appearance_reid(
    track_key: Tuple[int, int],
    p_track_id: int,
    camera_id: str,
    bbox: dict,
    confidence: float,
    is_entry_exit_cam: bool,
    frame_num: int,
    pad_index: int,
    obj_meta,
) -> None:
    """Drive ReIdManager for a visible track using the OSNet SGIE embedding.

    Reads the 512-dim appearance vector synchronously from NvDsInferTensorMeta —
    no background thread, no queue, no frame_np copy. The embedding arrives in the
    same frame as the detection (SGIE runs before the probe). Defers person_entry
    until the first embedding arrives; falls back after ENTRY_EMIT_DEADLINE_FRAMES
    if the SGIE never fires (bbox consistently below min-width/height threshold).
    """
    if _reid_manager is None:
        return

    state = _active_tracks[track_key]

    # ── Read embedding from OSNet SGIE metadata (synchronous) ─────────────────
    # The SGIE runs before the probe in the GStreamer pipeline, so the tensor is
    # already attached to obj_meta when we get here. Returns None if the bbox was
    # below the min-size threshold in the nvinfer config (96×192 px).
    vec = _extract_osnet_embedding(obj_meta)

    if vec is not None:
        # ── Aspect-ratio floor: skip detections too degraded to be worth trying ─
        # A standing person has height/width ≈ 3–4. Below PARTIAL_BODY_MIN_RATIO
        # only legs/feet or a sliver of torso are visible — extend the deadline
        # and wait for a better view instead of matching against noise. Anything
        # at or above the floor goes through the same SIMILARITY_THRESHOLD as any
        # other detection (see PARTIAL_BODY_MIN_RATIO comment for why there's no
        # longer a separate lower threshold for partial views).
        ratio = bbox["height"] / bbox["width"] if bbox["width"] > 0 else 0.0
        if ratio < PARTIAL_BODY_MIN_RATIO:
            # Too partial to be useful — extend deadline and wait for a better view
            state.entry_deadline = frame_num + ENTRY_EMIT_DEADLINE_FRAMES
            logger.debug("ReID: ratio=%.2f too low track=%d — skip", ratio, p_track_id)
        else:
            if not state.appearance_sent:
                # ── First embedding for this track — run ReID match ───────────
                global_id, event_type, prev_camera, expired_ids = _reid_manager.match_or_create(
                    vec, camera_id, track_id=p_track_id,
                )
                # Nothing else clears FaceRecognizer's own vote/lock state or
                # _employee_by_global_id by global_id — do it here, the only
                # place a global_id's expiry is ever observed.
                for gid in expired_ids:
                    if _face_recognizer is not None:
                        _face_recognizer.forget(gid)
                    _employee_by_global_id.pop(gid, None)
                # match_or_create() always resolves a global_id now (matches or
                # creates) — no more "partial body, no create" case to handle here.
                state.appearance_sent = True
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

                # Same-camera re-detection: demote to person_return to avoid a
                # spurious channel_change when the tracker re-detects on the same camera.
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
                        if event_type == "new_person":
                            _get_analytics(pad_index)["person_count"] += 1

            elif state.global_id is not None and frame_num % 90 == 0:
                # ── Subsequent frames — refresh the gallery periodically ──────
                # Adds new angles/poses so cross-camera matching stays accurate.
                _reid_manager.update_embedding(state.global_id, vec, track_id=p_track_id)

    # ── Deadline fallback: emit entry if SGIE never returned an embedding ─────
    # Covers the case where the person's bbox never met the min-size threshold
    # (96×192 px) — an intentional quality gate, not just an edge case: too-small
    # detections shouldn't get counted, ReID'd, face-recognized, or positioned.
    # person_entry still goes out without global_id rather than never going out.
    if not state.entry_emitted and frame_num >= state.entry_deadline:
        state.entry_emitted = True
        api_client.post_person_entry(
            camera_id, p_track_id,
            state.pending_bbox or bbox,
            state.pending_conf or confidence,
            is_entry_exit_cam,
            global_id=None,
            is_return=False,
        )
        logger.debug("ReID deadline reached track=%d cam=%s — entry emitted without global_id",
                     p_track_id, camera_id)


def _should_count_camera(pad_index: int) -> bool:
    # NOTE: this function duplicates the inline camera-type guard in osd_sink_pad_buffer_probe.
    # Consider replacing the inline block with a call to this function.
    """Return True if analytics for this camera should be processed."""
    is_external = pad_index in _external_pads
    if is_external and not _count_external:
        return False
    if not is_external and not _count_internal:
        return False
    return True


# ── Main GStreamer probe ───────────────────────────────────────────────────────

def osd_sink_pad_buffer_probe(_pad, info):
    """Single probe on the caps_rgba src-pad (full-res RGBA frames per camera, no tiler).

    Lazy frame read: GPU->CPU copy only when a worker needs pixels or when the
    scene is empty and it's time to capture a reference frame (at most every 30s).
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
        frame_num = frame_meta.frame_num
        pad_index = frame_meta.pad_index
        camera_id = _camera_id_for(pad_index)
        is_entry_exit_cam = pad_index in _entry_exit_pads

        _is_external_cam = pad_index in _external_pads
        if _is_external_cam and not _count_external:
            continue
        if not _is_external_cam and not _count_internal:
            continue

        # ── Lazy frame read ───────────────────────────────────────────────────
        # GPU->CPU copy only when face_recognizer needs crops (for InsightFace embedding).
        # OSNet embeddings come from the SGIE directly — no pixel copy needed for ReID.
        # In stream mode the tiled frame is read by Probe B from the tiler output, not here.
        frame_np = None
        _needs_pixel = False

        if frame_meta.num_obj_meta > 0:
            if _face_recognizer is not None:
                _needs_pixel = True

        # Pre-scan: detect if any person-class object is present before deciding
        # whether to decode for a reference frame. Bags and faces (non-person
        # PeopleNet classes) should not block the reference frame path — the send
        # condition uses visible_ids == 0 (persons only), not num_obj_meta == 0.
        _has_person_detection = frame_meta.num_obj_meta > 0 and any(
            int(o.class_id) == PGIE_CLASS_PERSON
            and o.unique_component_id == PGIE_UNIQUE_ID
            and o.confidence >= OSD_CONFIDENCE_THRESHOLD
            for o in _iter_pyds_list(frame_meta.obj_meta_list, pyds.NvDsObjectMeta.cast)
        )

        # Decode pixels for reference frame when no persons visible, even if bags
        # or faces are detected. Overhead: at most one GPU→CPU copy every 30 s
        # (initial retry) or 24 h (scene-change update).
        if not _needs_pixel and not _has_person_detection:
            _rc_confirmed = _reference_frame_confirmed_np.get(camera_id)
            _rc_ts = _reference_frame_confirmed_ts.get(camera_id, 0.0)
            _rc_attempt = _reference_frame_last_attempt.get(pad_index, 0.0)
            _now_rc = time.monotonic()
            _needs_pixel = (
                (_rc_confirmed is None
                 and _now_rc - _rc_attempt >= REFERENCE_FRAME_RETRY_SECS)
                or (_rc_confirmed is not None
                    and _now_rc - _rc_ts >= REFERENCE_FRAME_MIN_INTERVAL_SECS)
            )

        if _needs_pixel:
            try:
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frame_np = np.array(n_frame, copy=True, order='C')
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR)
            except Exception as e:
                if frame_num % 30 == 0:
                    logger.warning("get_nvds_buf_surface failed frame=%d: %s", frame_num, e)

        # ── Separate persons and faces ────────────────────────────────────────
        persons_meta: List = []
        face_metas: List = []
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

        # Debug: expose every raw face detection in stream mode, regardless of
        # recognition state — lets you see what PeopleNet is finding before any
        # ReID/recognition gating happens downstream in _face_handler.
        if _IS_STREAM_ENABLED:
            for face_obj_meta in face_metas:
                f_track_id = int(face_obj_meta.object_id)
                _track_labels[f_track_id] = {
                    "label": f"Cara {face_obj_meta.confidence:.0%}",
                    "fall": False,
                    "face": True,
                }

        if _face_handler and face_metas and frame_np is not None:
            for face_obj_meta in face_metas:
                _face_handler.process_face(
                    face_obj_meta, frame_num, frame_np, persons_meta, camera_id, pad_index
                )

        # ── Person track processing ───────────────────────────────────────────
        visible_ids: Set[int] = set()

        for obj_meta in persons_meta:
            p_track_id = int(obj_meta.object_id)
            visible_ids.add(p_track_id)
            r = obj_meta.rect_params
            bbox = {
                "left": max(0, int(r.left)),
                "top": max(0, int(r.top)),
                "width": int(r.width),
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
                frame_num, pad_index, obj_meta,
            )

            # OSD label: short display number once ReID resolves, "..." while waiting.
            # _display_ids is stream-only and does not affect global_id or API payloads.
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
                    api_client.post_person_classified(camera_id, state.global_id, bbox, demo)
                    an = _get_analytics(pad_index)
                    au = result.analytics_update
                    if "age_gender_classes" in au:
                        lbl = au["age_gender_classes"]
                        an["age_gender_classes"][lbl] = an["age_gender_classes"].get(lbl, 0) + 1
                    if "gender_key" in au:
                        an[au["gender_key"]] += 1

            if _face_handler:
                identity = _face_handler.get_identity(p_track_id)
                if identity:
                    # identity_key is the raw UUID or the literal "Unknown" sentinel
                    # (see face_recognizer.py) — resolve it to a human name via the
                    # same lookup the console EMPLEADO log already uses, and only
                    # draw the overlay for actual matches.
                    identity_key, conf = identity
                    if identity_key != "Unknown" and _face_recognizer is not None:
                        display_name = _face_recognizer.get_display_name(identity_key)
                        cur = str(obj_meta.text_params.display_text) or "..."
                        _set_osd_text(
                            obj_meta,
                            f"{cur} | {display_name} {conf:.0%}",
                            border_color=(0.2, 1.0, 0.4, 1.0),
                        )

            # if there is a frame
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
                            global_id=state.global_id,
                        )
                        _crop_counts[p_track_id] = count + 1
                        _crop_last_frame[p_track_id] = frame_num

            # str() required: display_text in pyds is a ctypes type, not a Python str.
            if _IS_STREAM_ENABLED:
                raw = obj_meta.text_params.display_text
                _track_labels[p_track_id] = {
                    "label": str(raw) if raw else f"P#{p_track_id}",
                    "fall": False,
                }

        _expire_lost_tracks(pad_index, frame_num, visible_ids)

        if _ws_client is not None and visible_ids:
            _accumulate_positions(pad_index, camera_id, persons_meta, frame_meta)

        # ── Periodic analytics snapshot ───────────────────────────────────────
        now = time.monotonic()
        if now - _get_analytics_last_sent(pad_index) >= ANALYTICS_SEND_INTERVAL_SECS:
            an = _get_analytics(pad_index)
            api_client.post_analytics_snapshot(camera_id, {
                "people_count": an["person_count"],
                "gender_male": an["gender_male"],
                "gender_female": an["gender_female"],
                "age_gender_classes": an["age_gender_classes"],
                "tailscale_ip": TAILSCALE_IP,
            }, period_seconds=ANALYTICS_SEND_INTERVAL_SECS)
            _analytics[pad_index] = {
                "person_count": 0, "gender_male": 0,
                "gender_female": 0, "age_gender_classes": {},
            }
            _analytics_last_sent[pad_index] = now

        # ── Reference frame: retry until confirmed + visual change detection ──
        # Only evaluated when no persons are visible and the frame is bright enough.
        # Non-person objects (bags, faces without a body — PeopleNet BAG/FACE classes)
        # are part of the background and do not block the send; only dark frames are excluded.

        # if there are people, reset the count
        global CURRENT_FRAME_SPACE
        if (len(visible_ids) != 0): 
            CURRENT_FRAME_SPACE[camera_id] = 0
        if (frame_np is not None
                and len(visible_ids) == 0
                and _frame_is_bright_enough(frame_np)):
            _ref_confirmed_np = _reference_frame_confirmed_np.get(camera_id)
            _ref_confirmed_ts = _reference_frame_confirmed_ts.get(camera_id, 0.0)
            _ref_last_attempt = _reference_frame_last_attempt.get(pad_index, 0.0)


            # Add to CURRENT_FRAME_SPACE
            CURRENT_FRAME_SPACE[camera_id] = CURRENT_FRAME_SPACE.get(camera_id, 0) + 1
            
            # Case 1 — never confirmed: retry every REFERENCE_FRAME_RETRY_SECS.
            _needs_initial = (
                _ref_confirmed_np is None
                and now - _ref_last_attempt >= REFERENCE_FRAME_RETRY_SECS
            )
            # Case 2 — already confirmed: resend only if the scene changed significantly
            # and at least 24h have passed since the last confirmed frame.
            _needs_update = (
                _ref_confirmed_np is not None
                and now - _ref_confirmed_ts >= REFERENCE_FRAME_MIN_INTERVAL_SECS
                and _scene_changed(frame_np, _ref_confirmed_np)
            )

            if (_needs_initial or _needs_update) and CURRENT_FRAME_SPACE[camera_id] >= MIN_REFERENCE_FRAME_SPACE:
                fh, fw = frame_np.shape[:2]
                _, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_b64 = base64.b64encode(buf).decode("utf-8")
                api_client.post_reference_frame(camera_id, frame_num, frame_b64, fw, fh)
                _reference_frame_last_attempt[pad_index] = now
                reason = "initial/retry" if _needs_initial else "visual change detected"
                logger.info(
                    "Reference frame queued camera=%s frame=%d %dx%d [%s]",
                    camera_id, frame_num, fw, fh, reason,
                )
                CURRENT_FRAME_SPACE[camera_id] = 0

    return Gst.PadProbeReturn.OK
