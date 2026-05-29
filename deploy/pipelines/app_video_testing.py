"""
app_video_testing.py — NX Computing AI | Testing Pipeline (MP4 / file source)

Usage:
  python pipelines/app_video_testing.py [video.mp4] [--capabilities cap1,cap2,...]

Examples:
  python pipelines/app_video_testing.py test_videos/clip.mp4
  python pipelines/app_video_testing.py test_videos/fall.mp4 --capabilities people_counting,fall_detection
  python pipelines/app_video_testing.py test_videos/office.mp4 --capabilities people_counting,age_gender,face_recognition --client demo

QA mode (NX_QA_ENABLED=true):
  Launched by docker-entrypoint.sh for dashboard playback. Uses the same dual-probe
  QA path as app.py: tiler + Probe A (full-res analytics) + Probe B (overlays) +
  MjpegServer on :8080. The Streamlit dashboard shows live inference on the recording.

Dev mode (NX_QA_ENABLED not set):
  Uses nvrtspoutsinkbin as sink — view the output with VLC or any RTSP viewer.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import gi
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import cv2
import pyds
from app import SGIE_CONFIGS
from probes import (
    osd_sink_pad_buffer_probe,
    pre_tiler_analytics_probe,
    api_client,
    init_channel_map,
    init_sector,
    init_handlers,
    init_workers,
    start_workers,
    stop_workers,
    init_qa_cameras,
    init_qa_grid,
    init_pipeline_stats,
    set_recording_manager,
    tiled_frame_queue,
    camera_frame_queues,
    _IS_QA_ENABLED,
    _redis_qa,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_MODELS_DIR     = Path(__file__).resolve().parent.parent / "models"
TEST_VIDEOS_DIR = "test_videos"

# Fallback dimensions when cv2.VideoCapture cannot probe the clip.
# Happens with some non-standard formats or corrupted files.
_VIDEO_WIDTH_DEFAULT  = 1920
_VIDEO_HEIGHT_DEFAULT = 1080

# nvtracker input resolution — downscaled to reduce GPU memory pressure.
# Full-res frames are processed directly by the probe, not the tracker.
# Increasing these improves tracking of small/distant objects at higher memory cost.
_TRACKER_INPUT_WIDTH  = 640
_TRACKER_INPUT_HEIGHT = 384

# Dev mode RTSP sink settings (VLC: rtsp://localhost:8554/ds-test).
# Only active when NX_QA_ENABLED is not set.
_DEV_RTSP_PORT    = 8554
_DEV_RTSP_BITRATE = 4_000_000

# QA tiler output — single stream fills the full composited frame.
# Must match the tiler dimensions configured in app.py.
_QA_TILER_WIDTH  = 640
_QA_TILER_HEIGHT = 360


def _find_video(path_arg: str | None) -> str:
    """Return the video path to use.

    Uses path_arg directly if provided. Otherwise finds the first video
    alphabetically inside TEST_VIDEOS_DIR. Exits with code 1 if no video
    is found.
    """
    if path_arg:
        return path_arg

    try:
        videos = sorted(
            os.path.join(TEST_VIDEOS_DIR, f)
            for f in os.listdir(TEST_VIDEOS_DIR)
            if f.lower().endswith((".mp4", ".h264", ".h265", ".mkv"))
        )
    except FileNotFoundError:
        logger.error("Directory '%s' not found.", TEST_VIDEOS_DIR)
        sys.exit(1)

    if not videos:
        logger.error("No videos found in %s", TEST_VIDEOS_DIR)
        sys.exit(1)

    return videos[0]


def main() -> None:
    """Entry point for the file-based testing pipeline.

    Builds a GStreamer pipeline using filesrc instead of rtspsrc. The inference
    chain (PGIE, tracker, SGIEs, Python workers) is identical to production.

    Loop modes:
      loop=True (default): seeks back to the start on EOS for continuous testing.
      loop=False (--no-loop): exits on EOS — used by docker-entrypoint.sh for QA playback.

    Sink by environment:
      QA mode (NX_QA_ENABLED=true): tiler + fakesink + MjpegServer on :8080.
      Dev mode: nvrtspoutsinkbin on :8554 for VLC inspection.
    """
    parser = argparse.ArgumentParser(description="NX video testing pipeline")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--input", "-i", default=None,
                        help="Path to video file (alias for positional arg, used by entrypoint)")
    parser.add_argument("--capabilities", "-c", default="people_counting,age_gender",
                        help="Comma-separated capabilities (default: people_counting,age_gender)")
    parser.add_argument("--client", default=None,
                        help="Client name for face DB (default: /etc/nx_client or 'demo')")
    parser.add_argument("--no-loop", action="store_true",
                        help="Exit when video ends instead of looping (QA playback mode)")
    args = parser.parse_args()

    # --input takes priority over the positional arg
    video_path = _find_video(args.input or args.video)
    loop_video = not args.no_loop

    pipeline_caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    if "people_counting" not in pipeline_caps:
        pipeline_caps = ["people_counting"] + pipeline_caps

    client_name = args.client
    if not client_name:
        try:
            client_name = Path("/etc/nx_client").read_text().strip()
        except FileNotFoundError:
            client_name = "demo"

    face_db_path = str(
        Path(__file__).resolve().parent.parent / "clients" / client_name / "known_faces.json"
    )

    logger.info("Video       : %s", video_path)
    logger.info("Capabilities: %s", pipeline_caps)
    logger.info("Client      : %s", client_name)
    logger.info("QA mode     : %s", _IS_QA_ENABLED)

    # Probe actual clip dimensions before building the pipeline so nvstreammux
    # uses the real resolution. Without this, sub-stream or tiled clips (e.g. 640×360)
    # get stretched to 1920×1080 inside the muxer.
    _video_cap = cv2.VideoCapture(video_path)
    video_width  = int(_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or _VIDEO_WIDTH_DEFAULT
    video_height = int(_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or _VIDEO_HEIGHT_DEFAULT
    _video_cap.release()
    logger.info("Video dims  : %dx%d", video_width, video_height)

    # Read sector from /etc/nx_sector so playback events (entry/exit signals,
    # face recognition event names) are consistent with the live pipeline.
    try:
        sector = Path("/etc/nx_sector").read_text().strip()
    except FileNotFoundError:
        sector = "comercio"
    init_sector(sector)

    init_channel_map([1])
    init_workers(pipeline_caps, model_dir=str(_MODELS_DIR), face_db_path=face_db_path)
    init_handlers(pipeline_caps)

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ── Source ────────────────────────────────────────────────────────────────
    # decodebin auto-detects the codec (mp4v, h264, h265, etc.), replacing the old
    # qtdemux + h264parse chain that only handled H.264 and broke on mp4v clips.
    # nvvidconv_src + caps_nvmm normalize frames to NVMM NV12 regardless of whether
    # decodebin decoded in hardware (NVMM) or software (SystemMemory).
    source        = Gst.ElementFactory.make("filesrc",        "file-source")
    decodebin     = Gst.ElementFactory.make("decodebin",      "decoder")
    nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", "pre-mux-convert")
    caps_nvmm     = Gst.ElementFactory.make("capsfilter",     "capsfilter-nvmm")
    caps_nvmm.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
    source.set_property("location", video_path)

    # ── Streammux ─────────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property("width",                video_width)
    streammux.set_property("height",               video_height)
    streammux.set_property("batch-size",           1)
    # 33333 µs ≈ 1 frame at 30 fps — avoids the 4 s startup latency of the default.
    streammux.set_property("batched-push-timeout", 33333)

    # ── PGIE — PeopleNet ──────────────────────────────────────────────────────
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path",
                      "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt")

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width",  _TRACKER_INPUT_WIDTH)
    tracker.set_property("tracker-height", _TRACKER_INPUT_HEIGHT)
    tracker.set_property("gpu-id",         0)
    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file",
        "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    tracker.set_property("display-tracking-id", 1)

    # ── SGIEs — one per GStreamer-backed capability ───────────────────────────
    sgie_elements = []
    for cap in pipeline_caps:
        if cap == "people_counting":
            continue
        cfg_path = SGIE_CONFIGS.get(cap)
        if cfg_path is None:
            # Python-worker capability — no GStreamer inference element needed.
            logger.info("Capability '%s' uses a Python worker — skipping SGIE", cap)
            continue
        if not Path(cfg_path).exists():
            logger.warning("SGIE config not found for '%s': %s — skipping", cap, cfg_path)
            continue
        sgie = Gst.ElementFactory.make("nvinfer", f"sgie-{cap}")
        sgie.set_property("config-file-path", cfg_path)
        sgie_elements.append(sgie)
        logger.info("SGIE loaded: %s → %s", cap, cfg_path)

    # ── NV12→RGBA (probes require RGBA to extract crops) ──────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",     "capsfilter-rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    if _IS_QA_ENABLED:
        # QA mode: 1×1 tiler + fakesink — mirrors app.py QA pipeline.
        # Probe A (pre-tiler, full-res) runs all analytics.
        # Probe B (post-tiler, 640×360) draws overlays and feeds MjpegServer.
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows",    1)
        tiler.set_property("columns", 1)
        tiler.set_property("width",   _QA_TILER_WIDTH)
        tiler.set_property("height",  _QA_TILER_HEIGHT)

        fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
        fakesink.set_property("sync", False)

        pipeline_elements = (
            [source, decodebin, nvvidconv_src, caps_nvmm, streammux, pgie, tracker]
            + sgie_elements
            + [nvvidconv1, caps_rgba, tiler, fakesink]
        )
    else:
        # Dev mode: nvrtspoutsinkbin on :8554 — useful for inspection with VLC.
        nvosd      = Gst.ElementFactory.make("nvdsosd",        "onscreendisplay")
        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
        sink = Gst.ElementFactory.make("nvrtspoutsinkbin", "rtsp-renderer")
        sink.set_property("rtsp-port", _DEV_RTSP_PORT)
        sink.set_property("enc-type",  0)
        sink.set_property("codec",     0)
        sink.set_property("bitrate",   _DEV_RTSP_BITRATE)
        sink.set_property("sync",      False)

        pipeline_elements = (
            [source, decodebin, nvvidconv_src, caps_nvmm, streammux, pgie, tracker]
            + sgie_elements
            + [nvvidconv1, caps_rgba, nvosd, nvvidconv2, sink]
        )

    if not all(pipeline_elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)

    for el in pipeline_elements:
        pipeline.add(el)

    # ── Linking ───────────────────────────────────────────────────────────────
    def _on_decode_pad_added(_decodebin: Gst.Element, pad: Gst.Pad) -> None:
        """Connect decodebin to the NVMM converter when a video pad becomes available.

        decodebin creates pads dynamically after probing the file codec.
        Audio pads are discarded here — only the video pad is linked.
        Compatible with mp4v (MPEG-4 Part 2), h264, h265, vp9, and others.
        """
        caps_str = (pad.get_current_caps() or pad.query_caps(None)).to_string()
        if not caps_str.startswith("video"):
            return
        sink_pad = nvvidconv_src.get_static_pad("sink")
        if not sink_pad.is_linked():
            # Guard against double-linking if decodebin emits multiple video pads.
            pad.link(sink_pad)

    source.link(decodebin)
    decodebin.connect("pad-added", _on_decode_pad_added)

    # nvvidconv_src normalizes memory to NVMM NV12 regardless of decode path —
    # nvstreammux only accepts NVMM input.
    nvvidconv_src.link(caps_nvmm)
    caps_nvmm.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    streammux.link(pgie)
    pgie.link(tracker)

    prev = tracker
    for sgie in sgie_elements:
        prev.link(sgie)
        prev = sgie
    prev.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)

    if _IS_QA_ENABLED:
        caps_rgba.link(tiler)
        tiler.link(fakesink)
    else:
        caps_rgba.link(nvosd)
        nvosd.link(nvvidconv2)
        nvvidconv2.link(sink)

    # ── Probe attachment ──────────────────────────────────────────────────────
    if _IS_QA_ENABLED:
        # Probe A: full-res analytics on caps_rgba src-pad (before the tiler)
        caps_rgba.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, pre_tiler_analytics_probe
        )
        # Probe B: overlays + MJPEG feed on tiler src-pad (640×360 composited frame)
        tiler.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe
        )
        logger.info("[QA] Probe A on caps_rgba src-pad; Probe B on tiler src-pad")
    else:
        # Dev mode: single probe on the nvosd sink-pad
        nvosd.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe
        )

    # ── QA services — mirrors the QA setup in app.py ──────────────────────────
    _recording_manager = None
    if _IS_QA_ENABLED:
        # 1 stream → 1×1 grid: cell covers the full tiler frame
        init_qa_cameras([1])
        init_qa_grid(1, 1, _QA_TILER_WIDTH, _QA_TILER_HEIGHT)
        init_pipeline_stats([1])

        # Deferred — RecordingManager and MjpegServer are QA-only dependencies
        # not installed in the production container image.
        from recording_manager import RecordingManager
        _recording_manager = RecordingManager(
            recordings_dir="/nx_tech/recordings",
            redis_client=_redis_qa,
        )
        set_recording_manager(_recording_manager)
        _recording_manager.start()

        from mjpeg_server import MjpegServer
        _mjpeg_srv = MjpegServer(
            tiled_frame_queue=tiled_frame_queue,
            camera_queues=camera_frame_queues,
            port=8080,
            recorder=_recording_manager,
        )
        _mjpeg_srv.start()
        logger.info("[QA] MjpegServer on :8080  /stream/all + /stream/<camera_id>")

        if _redis_qa:
            try:
                _redis_qa.set("nx:qa:playback_info", json.dumps({
                    "video":        video_path,
                    "capabilities": pipeline_caps,
                }))
            except Exception:
                # Informational metadata — Redis failure here is not critical.
                pass

    # ── Main loop ─────────────────────────────────────────────────────────────
    api_client.start()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_bus_message(_bus: Gst.Bus, message: Gst.Message) -> None:
        """Handle GStreamer bus messages: EOS, WARNING, ERROR.

        On EOS with loop_video=True, seeks back to the start.
        On EOS with loop_video=False (QA playback), quits the main loop so
        main() returns and the entrypoint loop can clean up.
        """
        t = message.type
        if t == Gst.MessageType.EOS:
            if loop_video:
                logger.info("EOS — seeking back to start for continuous testing.")
                pipeline.seek_simple(
                    Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
                )
            else:
                logger.info("EOS — playback complete (no-loop mode).")
                loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logger.warning("GStreamer WARNING: %s — %s", err, dbg)
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logger.error("GStreamer ERROR: %s — %s", err, dbg)
            loop.quit()

    bus.connect("message", _on_bus_message)

    logger.info("Starting pipeline — first run will build TensorRT engines.")
    pipeline.set_state(Gst.State.PLAYING)
    start_workers()

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        api_client.stop()
        stop_workers()
        if _recording_manager is not None:
            _recording_manager.stop()
        if _IS_QA_ENABLED and _redis_qa:
            try:
                _redis_qa.delete("nx:qa:playback_info")
            except Exception:
                # Best-effort cleanup — key expires on its own if Redis is unavailable.
                pass


if __name__ == "__main__":
    main()
