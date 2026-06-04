"""
app_video_testing.py — NX Computing AI | Testing Pipeline (MP4 / file source)

Runs the same inference pipeline as app.py but uses a local video file instead
of RTSP. Useful for testing models, verifying analytics, and debugging on-device
before connecting to a DVR.

Usage:
  python pipelines/app_video_testing.py [video.mp4] [--capabilities cap1,cap2,...]
  python pipelines/app_video_testing.py test_videos/clip.mp4
  python pipelines/app_video_testing.py test_videos/office.mp4 --capabilities people_counting,age_gender,face_recognition

Stream mode (NX_STREAM_ENABLED=true):
  Same as app.py stream mode — probe draws bboxes and serves MJPEG on :8080.
  Activate with: NX_STREAM_ENABLED=true python pipelines/app_video_testing.py clip.mp4
"""
import argparse
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
    api_client,
    init_channel_map,
    init_sector,
    init_handlers,
    init_workers,
    start_workers,
    stop_workers,
    init_stream_cameras,
    camera_frame_queues,
    _IS_STREAM_ENABLED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_MODELS_DIR     = Path(__file__).resolve().parent.parent / "models"
TEST_VIDEOS_DIR = "test_videos"

# Fallback dimensions when cv2.VideoCapture cannot probe the clip.
_VIDEO_WIDTH_DEFAULT  = 1920
_VIDEO_HEIGHT_DEFAULT = 1080

# nvtracker input resolution — downscaled to reduce GPU memory pressure.
_TRACKER_INPUT_WIDTH  = 640
_TRACKER_INPUT_HEIGHT = 384


def _find_video(path_arg: str | None) -> str:
    """Return the video path to use. Falls back to first video in TEST_VIDEOS_DIR."""
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
    """Entry point for the file-based testing pipeline."""
    parser = argparse.ArgumentParser(description="NX video testing pipeline")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--input", "-i", default=None,
                        help="Path to video file (alias for positional arg)")
    parser.add_argument("--capabilities", "-c", default="people_counting,age_gender",
                        help="Comma-separated capabilities (default: people_counting,age_gender)")
    parser.add_argument("--client", default=None,
                        help="Client name for face DB (default: /etc/nx_client or 'demo')")
    parser.add_argument("--no-loop", action="store_true",
                        help="Exit when video ends instead of looping")
    args = parser.parse_args()

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
    logger.info("Stream mode : %s", _IS_STREAM_ENABLED)

    # Probe actual clip dimensions so nvstreammux uses the real resolution.
    # Without this, sub-stream or tiled clips get stretched inside the muxer.
    _video_cap = cv2.VideoCapture(video_path)
    video_width  = int(_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or _VIDEO_WIDTH_DEFAULT
    video_height = int(_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or _VIDEO_HEIGHT_DEFAULT
    _video_cap.release()
    logger.info("Video dims  : %dx%d", video_width, video_height)

    try:
        sector = Path("/etc/nx_sector").read_text().strip()
    except FileNotFoundError:
        sector = "comercio"
    init_sector(sector)

    init_channel_map([1])
    init_workers(pipeline_caps, model_dir=str(_MODELS_DIR), face_db_path=face_db_path)
    init_handlers(pipeline_caps)

    # Stream mode: inicializar queues + StreamServer (mismo que app.py)
    if _IS_STREAM_ENABLED:
        init_stream_cameras([1])
        from stream_server import StreamServer
        _stream_srv = StreamServer(camera_queues=camera_frame_queues, port=8080)
        _stream_srv.start()
        logger.info("[Stream] StreamServer en :8080 — /stream/<camera_id> + /viewer/<camera_id>")

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ── Source ────────────────────────────────────────────────────────────────
    # decodebin auto-detects codec (mp4v, h264, h265, etc.).
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

    # ── SGIEs ─────────────────────────────────────────────────────────────────
    sgie_elements = []
    for cap in pipeline_caps:
        if cap == "people_counting":
            continue
        cfg_path = SGIE_CONFIGS.get(cap)
        if cfg_path is None:
            logger.info("Capability '%s' uses a Python worker — skipping SGIE", cap)
            continue
        if not Path(cfg_path).exists():
            logger.warning("SGIE config not found for '%s': %s — skipping", cap, cfg_path)
            continue
        sgie = Gst.ElementFactory.make("nvinfer", f"sgie-{cap}")
        sgie.set_property("config-file-path", cfg_path)
        sgie_elements.append(sgie)
        logger.info("SGIE loaded: %s → %s", cap, cfg_path)

    # ── NV12→RGBA → fakesink ──────────────────────────────────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",     "capsfilter-rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    fakesink   = Gst.ElementFactory.make("fakesink", "fakesink")
    fakesink.set_property("sync", False)

    all_elements = (
        [source, decodebin, nvvidconv_src, caps_nvmm, streammux, pgie, tracker]
        + sgie_elements
        + [nvvidconv1, caps_rgba, fakesink]
    )

    if not all(all_elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)

    for el in all_elements:
        pipeline.add(el)

    # ── Linking ───────────────────────────────────────────────────────────────
    def _on_decode_pad_added(_decodebin: Gst.Element, pad: Gst.Pad) -> None:
        """Connect decodebin video pad to the NVMM converter."""
        caps_str = (pad.get_current_caps() or pad.query_caps(None)).to_string()
        if not caps_str.startswith("video"):
            return
        sink_pad = nvvidconv_src.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    source.link(decodebin)
    decodebin.connect("pad-added", _on_decode_pad_added)
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
    caps_rgba.link(fakesink)

    # ── Probe ─────────────────────────────────────────────────────────────────
    caps_rgba.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    api_client.start()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_bus_message(_bus: Gst.Bus, message: Gst.Message) -> None:
        """EOS with loop_video=True seeks back to start; False quits the loop."""
        t = message.type
        if t == Gst.MessageType.EOS:
            if loop_video:
                logger.info("EOS — seeking back to start for continuous testing.")
                pipeline.seek_simple(
                    Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
                )
            else:
                logger.info("EOS — playback complete.")
                loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logger.warning("GStreamer WARNING: %s — %s", err, dbg)
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logger.error("GStreamer ERROR: %s — %s", err, dbg)
            loop.quit()

    bus.connect("message", _on_bus_message)

    # Pre-roll: wait for PAUSED to ensure TRT engines are compiled before PLAYING.
    logger.info("Pre-rolling pipeline (compilando TRT engines si es la primera vez)...")
    pipeline.set_state(Gst.State.PAUSED)
    _PREROLL_TIMEOUT_NS = 15 * 60 * 1_000_000_000
    ret, _cur, _pend = pipeline.get_state(_PREROLL_TIMEOUT_NS)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Pipeline falló en pre-roll — abortando.")
        sys.exit(1)
    logger.info("Pre-roll completo — iniciando reproducción.")

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


if __name__ == "__main__":
    main()
