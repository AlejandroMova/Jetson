"""
app_video_testing.py — NX Computing AI | Testing Pipeline (MP4 / file source)

Usage:
  python pipelines/app_video_testing.py [video.mp4] [--capabilities cap1,cap2,...]

Examples:
  python pipelines/app_video_testing.py test_videos/clip.mp4
  python pipelines/app_video_testing.py test_videos/fall.mp4 --capabilities people_counting,fall_detection
  python pipelines/app_video_testing.py test_videos/office.mp4 --capabilities people_counting,age_gender,face_recognition --client demo
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import gi
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import pyds
from app import SGIE_CONFIGS
from probes import (
    osd_sink_pad_buffer_probe,
    api_client,
    init_channel_map,
    init_handlers,
    init_workers,
    stop_workers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TEST_VIDEOS_DIR = "test_videos"


def _find_video(path_arg):
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


def main():
    parser = argparse.ArgumentParser(description="NX video testing pipeline")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--input", "-i", default=None,
                        help="Path to video file (alias for positional arg, used by entrypoint)")
    parser.add_argument("--capabilities", "-c", default="people_counting,age_gender",
                        help="Comma-separated capabilities (default: people_counting,age_gender)")
    parser.add_argument("--client", default=None,
                        help="Client name for face DB (default: /etc/nx_client or 'demo')")
    parser.add_argument("--no-loop", action="store_true",
                        help="Exit when video ends instead of looping (usado en modo playback)")
    args = parser.parse_args()

    # --input tiene prioridad sobre el arg posicional
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

    init_channel_map([1])
    init_workers(pipeline_caps, model_dir=str(_MODELS_DIR), face_db_path=face_db_path)
    init_handlers(pipeline_caps)

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ── Source ────────────────────────────────────────────────────────────────
    source     = Gst.ElementFactory.make("filesrc",        "file-source")
    qtdemux    = Gst.ElementFactory.make("qtdemux",        "demuxer")
    h264parser = Gst.ElementFactory.make("h264parse",      "h264-parser")
    decoder    = Gst.ElementFactory.make("nvv4l2decoder",  "nvv4l2-decoder")
    source.set_property("location", video_path)
    decoder.set_property("drop-frame-interval", 0)

    # ── Streammux ─────────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property("width",                1920)
    streammux.set_property("height",               1080)
    streammux.set_property("batch-size",           1)
    streammux.set_property("batched-push-timeout", 33333)

    # ── PGIE — PeopleNet ──────────────────────────────────────────────────────
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path",
                      "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt")

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width",  640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id",         0)
    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file",
        "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    tracker.set_property("display-tracking-id", 1)

    # ── SGIEs (dynamic, based on capabilities) ────────────────────────────────
    sgie_elements = []
    for cap in pipeline_caps:
        if cap == "people_counting":
            continue
        cfg_path = SGIE_CONFIGS.get(cap)
        if cfg_path is None:
            logger.info("Capability '%s' uses Python worker — skipping SGIE", cap)
            continue
        if not Path(cfg_path).exists():
            logger.warning("SGIE config not found for '%s': %s — skipping", cap, cfg_path)
            continue
        sgie = Gst.ElementFactory.make("nvinfer", f"sgie-{cap}")
        sgie.set_property("config-file-path", cfg_path)
        sgie_elements.append(sgie)
        logger.info("SGIE: %s → %s", cap, cfg_path)

    # ── OSD ───────────────────────────────────────────────────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",     "capsfilter-rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd      = Gst.ElementFactory.make("nvdsosd",        "onscreendisplay")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")

    # ── Sink — RTSP output ────────────────────────────────────────────────────
    sink = Gst.ElementFactory.make("nvrtspoutsinkbin", "rtsp-renderer")
    sink.set_property("rtsp-port", 8554)
    sink.set_property("enc-type",  0)
    sink.set_property("codec",     0)
    sink.set_property("bitrate",   4000000)
    sink.set_property("sync",      False)

    all_elements = (
        [source, qtdemux, h264parser, decoder, streammux, pgie, tracker]
        + sgie_elements
        + [nvvidconv1, caps_rgba, nvosd, nvvidconv2, sink]
    )
    if not all(all_elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)

    for el in all_elements:
        pipeline.add(el)

    # ── Linking ───────────────────────────────────────────────────────────────
    def _on_demux_pad_added(_demux, pad):
        caps_str = (pad.get_current_caps() or pad.query_caps(None)).to_string()
        if "video/x-h264" in caps_str:
            sink_pad = h264parser.get_static_pad("sink")
            if not sink_pad.is_linked():
                pad.link(sink_pad)

    source.link(qtdemux)
    qtdemux.connect("pad-added", _on_demux_pad_added)
    h264parser.link(decoder)
    decoder.get_static_pad("src").link(streammux.get_request_pad("sink_0"))
    streammux.link(pgie)
    pgie.link(tracker)

    prev = tracker
    for sgie in sgie_elements:
        prev.link(sgie)
        prev = sgie
    prev.link(nvvidconv1)

    nvvidconv1.link(caps_rgba)
    caps_rgba.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(sink)

    # ── Probe ─────────────────────────────────────────────────────────────────
    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)

    api_client.start()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_bus_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            if loop_video:
                logger.info("EOS — looping video for testing.")
                pipeline.seek_simple(
                    Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
                )
            else:
                logger.info("EOS — video terminado (modo playback, sin loop).")
                loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logger.warning("GStreamer WARNING: %s — %s", err, dbg)
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logger.error("GStreamer ERROR: %s — %s", err, dbg)
            loop.quit()

    bus.connect("message", _on_bus_message)

    logger.info("Starting pipeline… (first run builds TensorRT engines)")
    pipeline.set_state(Gst.State.PLAYING)

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
