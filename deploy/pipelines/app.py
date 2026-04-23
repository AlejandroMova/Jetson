"""
app.py — NX Computing AI | Production Pipeline (Live DVR / RTSP)

Source: RTSP stream(s) from DVR, configured per-client via config_loader.
Sink  : RTSP output on port 8554 — view with VLC:
          vlc rtsp://<jetson-ip>:8554/ds-test

Pipeline:
  rtspsrc → rtph264depay → h264parse → nvv4l2decoder
    → nvstreammux → nvinfer (PeopleNet) → nvtracker
    → nvinfer (Age/Gender) → nvvideoconvert → capsfilter(RGBA)
    → nvdsosd → nvvideoconvert → nvrtspoutsinkbin
"""

import logging
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from config_loader import load_config
from probes import osd_sink_pad_buffer_probe, api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _add_rtsp_source(pipeline, streammux, rtsp_url: str, stream_idx: int):
    """Add one RTSP source branch and link it to streammux sink_{stream_idx}."""
    source  = Gst.ElementFactory.make("rtspsrc",      f"source-{stream_idx}")
    depay   = Gst.ElementFactory.make("rtph264depay", f"depay-{stream_idx}")
    parser  = Gst.ElementFactory.make("h264parse",    f"parser-{stream_idx}")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{stream_idx}")

    if not all([source, depay, parser, decoder]):
        logger.error("Could not create source elements for stream %d", stream_idx)
        sys.exit(1)

    source.set_property("location",        rtsp_url)
    source.set_property("latency",         200)
    source.set_property("drop-on-latency", True)
    source.set_property("protocols",       4)   # TCP only — more reliable through NAT

    decoder.set_property("drop-frame-interval", 0)

    for el in [source, depay, parser, decoder]:
        pipeline.add(el)

    depay.link(parser)
    parser.link(decoder)

    decoder_srcpad    = decoder.get_static_pad("src")
    streammux_sinkpad = streammux.get_request_pad(f"sink_{stream_idx}")
    decoder_srcpad.link(streammux_sinkpad)

    # rtspsrc pads are dynamic — connect when the RTSP server sends the SDP
    def _on_pad_added(_src, pad, _depay=depay):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and "video" in caps.to_string():
            sink = _depay.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)

    source.connect("pad-added", _on_pad_added)
    logger.info("Stream %d → %s", stream_idx, rtsp_url.replace(rtsp_url.split("@")[0].split("//")[1] if "@" in rtsp_url else "", "***:***"))


def main():
    cfg = load_config()
    cfg.log_summary()

    urls = cfg.rtsp_urls()
    n_streams = len(urls)

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ── Streammux ──────────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property("width",               cfg.stream_width)
    streammux.set_property("height",              cfg.stream_height)
    streammux.set_property("batch-size",          n_streams)
    streammux.set_property("batched-push-timeout", 33333)   # ~1 frame @ 30 fps (µs)
    streammux.set_property("live-source",         1)        # RTSP is a live source
    pipeline.add(streammux)

    # ── RTSP sources (one per channel) ────────────────────────────────────────
    for i, url in enumerate(urls):
        _add_rtsp_source(pipeline, streammux, url, i)

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

    # ── SGIE — Age/Gender ─────────────────────────────────────────────────────
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path",
                      "models/resnet_age_gender_FB2/config_infer.txt")

    # ── OSD ───────────────────────────────────────────────────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",    "capsfilter-rgba")
    caps_rgba.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd      = Gst.ElementFactory.make("nvdsosd",        "onscreendisplay")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")

    # ── RTSP output sink ──────────────────────────────────────────────────────
    sink = Gst.ElementFactory.make("nvrtspoutsinkbin", "rtsp-renderer")
    sink.set_property("rtsp-port", 8554)
    sink.set_property("enc-type",  0)       # hardware encoder (nvv4l2h264enc)
    sink.set_property("codec",     0)       # H264
    sink.set_property("bitrate",   4000000)
    sink.set_property("sync",      False)

    elements = [pgie, tracker, sgie, nvvidconv1, caps_rgba, nvosd, nvvidconv2, sink]
    if not all(elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)

    for el in elements:
        pipeline.add(el)

    # ── Linking ───────────────────────────────────────────────────────────────
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)
    caps_rgba.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(sink)

    # ── Probe ─────────────────────────────────────────────────────────────────
    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)

    # ── Run ───────────────────────────────────────────────────────────────────
    api_client.start()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_bus_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("EOS received — stopping.")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logger.warning("GStreamer WARNING: %s — %s", err, dbg)
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logger.error("GStreamer ERROR: %s — %s", err, dbg)
            loop.quit()

    bus.connect("message", _on_bus_message)

    logger.info("Starting pipeline (first run builds TensorRT engines — may take several minutes)…")
    logger.info("RTSP output → rtsp://<jetson-ip>:8554/ds-test")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        pipeline.set_state(Gst.State.NULL)
        api_client.stop()


if __name__ == "__main__":
    main()
