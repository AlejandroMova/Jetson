"""
recording_manager.py — NX Computing AI | QA Recording

Automatically records video clips whenever people are detected in the QA pipeline.
Only instantiated when NX_QA_ENABLED=true — zero impact in production.

Frame sources:
  - Tiled (640×360)    : push_tiled_frame()   — called from MjpegServer._encode_loop
  - Per-camera full-res: push_camera_frame()  — called from Probe A (pre_tiler_analytics_probe)

Detection trigger: notify_detection() called directly from the probes (no Redis pub/sub).
"""

import json
import logging
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# mp4v (MPEG-4 Part 2) — clips are played back via decodebin in app_video_testing.py,
# which auto-detects the codec. Do NOT change this without updating the playback pipeline.
_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


class RecordingManager:
    """Records video clips automatically when people are detected in the QA pipeline.

    State machine: IDLE → (detection) → RECORDING → (cooldown or max duration) → IDLE

    Two video streams are recorded in parallel:
      - tiled.mp4    : 640×360 composite view of all cameras (from MjpegServer)
      - <cam_id>.mp4 : full-resolution per-camera view (from Probe A)

    Storage is self-managing: oldest clips are pruned automatically when total
    size exceeds max_storage_gb.

    Thread safety:
      push_tiled_frame(), push_camera_frame(), and notify_detection() are called
      from probe/server threads. They write only to queue.Queue instances (thread-safe)
      or to _last_det_time/_max_people via atomic float/int assignment (CPython GIL).
      _record_loop is the sole writer for _recording, _clip_dir, and all VideoWriters.
    """

    def __init__(
        self,
        recordings_dir: str,
        redis_client,
        fps: float = 25.0,
        max_clip_minutes: float = 5.0,
        min_clip_seconds: float = 5.0,
        cooldown_seconds: float = 10.0,
        max_storage_gb: float = 10.0,
    ):
        self._dir = Path(recordings_dir)
        self._redis = redis_client
        self._fps = fps
        self._max_clip_secs = max_clip_minutes * 60.0
        self._min_clip_secs = min_clip_seconds
        self._cooldown_secs = cooldown_seconds
        self._max_bytes = int(max_storage_gb * 1024 ** 3)

        # maxsize=2: drop frames if the record loop falls behind rather than buffering
        # unboundedly. One dropped frame is less harmful than unbounded memory growth.
        self._tiled_q: queue.Queue = queue.Queue(maxsize=2)
        self._cam_qs: Dict[str, queue.Queue] = {}

        # Written by probe threads via notify_detection(); read by _record_loop.
        # Float/int assignment is atomic in CPython (GIL + STORE_ATTR).
        self._last_det_time: Optional[float] = None
        self._max_people: int = 0

        # All state below is owned exclusively by _record_loop.
        self._clip_dir: Optional[Path] = None
        self._clip_start: Optional[float] = None
        self._clip_ts: str = ""
        self._tiled_writer: Optional[cv2.VideoWriter] = None
        self._cam_writers: Dict[str, cv2.VideoWriter] = {}
        self._frame_count: int = 0
        self._thumbnail_saved: bool = False

        self._recording = False
        self._running = False

        self._rec_thread = threading.Thread(
            target=self._record_loop, daemon=True, name="RecordLoop"
        )

    # ── Public API (called from external threads) ─────────────────────────────

    @property
    def is_recording(self) -> bool:
        """True while a clip is in progress (between _start() and _finish()/_cancel()).

        Read by Probe A and MjpegServer to decide whether to forward frames here.
        Bool assignment is atomic in CPython (GIL protects the read).
        """
        return self._recording

    def push_tiled_frame(self, frame: np.ndarray) -> None:
        """Accept a 640×360 tiled frame from MjpegServer._encode_loop."""
        if not self._recording:
            return
        try:
            self._tiled_q.put_nowait(frame)
        except queue.Full:
            pass

    def push_camera_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Accept a full-resolution per-camera frame from Probe A."""
        if not self._recording:
            return
        if camera_id not in self._cam_qs:
            self._cam_qs[camera_id] = queue.Queue(maxsize=2)
        try:
            self._cam_qs[camera_id].put_nowait(frame)
        except queue.Full:
            pass

    def notify_detection(self, people_count: int = 1) -> None:
        """Signal that people are present. Called directly from the probes."""
        self._last_det_time = time.monotonic()
        if people_count > self._max_people:
            self._max_people = people_count

    def start(self) -> None:
        """Create the recordings directory and start the background recording thread."""
        self._running = True
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rec_thread.start()
        logger.info("[Recording] Started — directory: %s", self._dir)

    def stop(self) -> None:
        """Stop the recording thread, finishing any clip currently in progress."""
        self._running = False
        if self._recording:
            self._finish()

    # ── Recording loop ────────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        """Main loop for the background recording thread.

        Simple state machine: IDLE ↔ RECORDING.

        IDLE: polls every 50 ms for a recent detection that should start a new clip.
        RECORDING: drains frame queues each iteration and checks two stop conditions:
          - Cooldown expired (no detections for cooldown_seconds) → save the clip.
          - Max duration reached (elapsed > max_clip_minutes)     → save the clip.
        Clips shorter than min_clip_seconds when stopped are discarded, not saved.
        """
        while self._running:
            if not self._recording:
                if (self._last_det_time is not None
                        and time.monotonic() - self._last_det_time < 1.0):
                    self._start()
                else:
                    time.sleep(0.05)  # lightweight poll — avoids busy-waiting the CPU
                continue

            now = time.monotonic()
            idle_secs    = now - (self._last_det_time or 0)
            elapsed_secs = now - (self._clip_start or now)

            if idle_secs > self._cooldown_secs or elapsed_secs > self._max_clip_secs:
                if elapsed_secs >= self._min_clip_secs:
                    self._finish()
                else:
                    self._cancel()
                continue

            self._drain_frame_queues()

    def _drain_frame_queues(self) -> None:
        """Drain one tiled frame and any pending per-camera frames.

        The tiled queue uses a 40 ms timeout to pace tiled.mp4 at ~25 fps.
        Camera queues use non-blocking get so a slow camera doesn't stall the loop.
        """
        try:
            frame = self._tiled_q.get(timeout=0.04)
            self._write_tiled(frame)
        except queue.Empty:
            pass

        for cam_id, q in list(self._cam_qs.items()):
            try:
                frame = q.get_nowait()
                self._write_camera(cam_id, frame)
            except queue.Empty:
                pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _start(self) -> None:
        """Open a new clip directory and reset per-clip counters.

        _max_people is intentionally NOT reset here — it already holds the count
        from the notify_detection() call that triggered this recording. Resetting it
        here would produce clips with max_people=0 in metadata. It is cleared in
        _reset_writers() after the clip is fully closed.
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._clip_dir = self._dir / ts
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        self._clip_start = time.monotonic()
        self._clip_ts = ts
        self._frame_count = 0
        self._thumbnail_saved = False
        self._tiled_writer = None   # created lazily in _write_tiled on first frame
        self._cam_writers = {}      # created lazily in _write_camera on first frame
        self._cam_qs = {}           # clear queues carried over from the previous clip
        self._recording = True      # set before _publish_state so Streamlit sees it immediately
        self._publish_state(active=True, clip_ts=ts)
        logger.info("[Recording] Clip started: %s", ts)

    def _write_tiled(self, frame: np.ndarray) -> None:
        """Write one frame to tiled.mp4. Creates the VideoWriter lazily on the first frame.

        The first frame is also saved as thumbnail.jpg for the Streamlit clip gallery.
        """
        h, w = frame.shape[:2]
        if self._tiled_writer is None:
            self._tiled_writer = cv2.VideoWriter(
                str(self._clip_dir / "tiled.mp4"), _FOURCC, self._fps, (w, h)
            )

        if not self._thumbnail_saved:
            cv2.imwrite(
                str(self._clip_dir / "thumbnail.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            self._thumbnail_saved = True

        self._tiled_writer.write(frame)
        self._frame_count += 1

    def _write_camera(self, cam_id: str, frame: np.ndarray) -> None:
        """Write one full-resolution frame to <cam_id>.mp4. Creates the VideoWriter lazily."""
        h, w = frame.shape[:2]
        if cam_id not in self._cam_writers:
            self._cam_writers[cam_id] = cv2.VideoWriter(
                str(self._clip_dir / f"{cam_id}.mp4"), _FOURCC, self._fps, (w, h)
            )
        self._cam_writers[cam_id].write(frame)

    def _finish(self) -> None:
        """Close the current clip, write metadata.json, and prune storage if needed."""
        self._recording = False
        elapsed = time.monotonic() - (self._clip_start or 0)
        if self._tiled_writer:
            self._tiled_writer.release()
        for w in self._cam_writers.values():
            w.release()
        meta = {
            "timestamp":   self._clip_ts,
            "duration_s":  round(elapsed, 1),
            "channels":    list(self._cam_writers.keys()),
            "frame_count": self._frame_count,
            "max_people":  self._max_people,
        }
        try:
            with open(self._clip_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning("[Recording] Failed to write metadata: %s", e)
        logger.info(
            "[Recording] Clip saved: %s (%.1f s, %d tiled frames)",
            self._clip_ts, elapsed, self._frame_count,
        )
        self._publish_state(active=False)
        self._prune()
        self._reset_writers()

    def _cancel(self) -> None:
        """Discard the current clip — called when the clip is shorter than min_clip_seconds.

        Short clips are typically false-positive detections (a person crossed the frame
        in under 5 s). Discarding them keeps the clip library clean.
        """
        self._recording = False
        elapsed = time.monotonic() - (self._clip_start or 0)
        if self._tiled_writer:
            self._tiled_writer.release()
        for w in self._cam_writers.values():
            w.release()
        if self._clip_dir and self._clip_dir.exists():
            shutil.rmtree(self._clip_dir, ignore_errors=True)
        logger.info(
            "[Recording] Clip discarded (%.1f s < minimum %.1f s)",
            elapsed, self._min_clip_secs,
        )
        self._publish_state(active=False)
        self._reset_writers()

    def _reset_writers(self) -> None:
        """Clear all VideoWriters and clip state after _finish() or _cancel()."""
        self._tiled_writer = None
        self._cam_writers = {}
        self._cam_qs = {}
        self._clip_dir = None
        self._clip_start = None
        self._max_people = 0    # reset here, not in _start() — see _start() docstring

    def _publish_state(self, active: bool, clip_ts: str = "") -> None:
        """Publish recording state to Redis for the Streamlit dashboard.

        nx:qa:recording_active → "1" while recording, "0" otherwise.
        nx:qa:recording_info   → JSON with clip metadata (only set while active=True).
        Silently skips if Redis is unavailable — recording continues without it.
        """
        if not self._redis:
            return
        try:
            self._redis.set("nx:qa:recording_active", "1" if active else "0")
            if active:
                self._redis.set("nx:qa:recording_info", json.dumps({
                    "started":      clip_ts,
                    "clip_name":    clip_ts,
                    "people_count": self._max_people,
                }))
            else:
                self._redis.delete("nx:qa:recording_info")
        except Exception:
            pass  # Redis being down must never interrupt the recording itself

    def _prune(self) -> None:
        """Delete oldest clips until total storage falls below max_storage_gb."""
        try:
            clips = sorted(d for d in self._dir.iterdir() if d.is_dir())
            total = sum(
                f.stat().st_size
                for clip in clips
                for f in clip.rglob("*")
                if f.is_file()
            )
            for clip in clips:
                if total <= self._max_bytes:
                    break
                size = sum(f.stat().st_size for f in clip.rglob("*") if f.is_file())
                shutil.rmtree(clip, ignore_errors=True)
                total -= size
                logger.info("[Recording] Pruned oldest clip to free space: %s", clip.name)
        except Exception as e:
            logger.warning("[Recording] Storage prune failed: %s", e)
