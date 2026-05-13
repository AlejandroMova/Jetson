"""
app.py — NX Computing AI | Production Pipeline (Live DVR / RTSP)

Source: RTSP stream(s) from DVR, configured per-client via config_loader.
Sink  : fakesink — no display output. Visualización disponible con la app QA (NX_MODE=testing).

Pipeline (capabilities driven by config.yaml `pipeline` field or /etc/nx_pipeline):
  rtspsrc → rtph264depay → h264parse → nvv4l2decoder
    → nvstreammux → nvinfer (PeopleNet PGIE) → nvtracker
    → [nvinfer SGIE per active capability]
    → nvmultistreamtiler(640×360) → nvvideoconvert → capsfilter(RGBA)
    → [probe: crops para analytics] → fakesink
"""

import logging
import math
import os
import sys
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from config_loader import load_config
from probes import (
    osd_sink_pad_buffer_probe, api_client,
    init_channel_map, init_sector, init_entry_exit_pads,
    init_handlers, init_workers, start_workers, stop_workers,
)

# Maps each pipeline capability to its nvinfer config file (relative to deploy/).
# None = Python worker (no SGIE element created for that capability).
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SGIE_CONFIGS = {
    "age_gender":      str(_MODELS_DIR / "resnet_age_gender_FB2/config_infer.txt"),
    "epp_detection":   str(_MODELS_DIR / "epp/config_infer.txt"),
    "fire_smoke":      str(_MODELS_DIR / "fire_smoke/config_infer.txt"),
    "license_plate":   str(_MODELS_DIR / "license_plate/config_infer.txt"),
    "fall_detection":  None,  # MoveNet Python worker — no SGIE
    "face_recognition": None,  # PeopleNet class 2 (face) detections fed directly to worker
}


def _validate_pipeline_models(pipeline: list) -> None:
    """Fail fast before GStreamer init if a required model config or ONNX file is missing."""
    missing = []
    for cap in pipeline:
        if cap == "people_counting":
            continue
        cfg_path = SGIE_CONFIGS.get(cap)
        if cfg_path is None:
            continue  # Python worker — no file needed
        p = Path(cfg_path)
        if not p.exists():
            missing.append((cap, cfg_path, "config_infer.txt not found"))
            continue
        # Also verify the model file referenced inside the config exists.
        # Without this check the error only surfaces much later when DeepStream
        # tries to build the TRT engine — after GStreamer is already initialised.
        try:
            for line in p.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("onnx-file=") or stripped.startswith("tlt-encoded-model="):
                    model_name = stripped.split("=", 1)[1].strip()
                    model_path = p.parent / model_name
                    if not model_path.exists():
                        missing.append((cap, str(model_path),
                                        "model file not found — download it first"))
                    break
        except OSError:
            pass
    if missing:
        lines = "\n".join(f"  '{cap}' → {path}  ({reason})" for cap, path, reason in missing)
        raise RuntimeError(
            f"Cannot start: model file(s) missing for requested capabilities:\n{lines}\n"
            "Run:  docker compose run --rm deepstream python3 tools/download_models.py --help"
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _add_rtsp_source(pipeline, streammux, rtsp_url: str, stream_idx: int):
    """Add one RTSP source branch and link it to streammux sink_{stream_idx}.

    Depayloader and parser are created dynamically in pad-added to support both
    H.264 and H.265 cameras on the same DVR without prior knowledge of each codec.
    """
    source  = Gst.ElementFactory.make("rtspsrc",       f"source-{stream_idx}")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{stream_idx}")

    if not all([source, decoder]):
        logger.error("Could not create source/decoder for stream %d", stream_idx)
        sys.exit(1)

    source.set_property("location",        rtsp_url)
    source.set_property("latency",         200)
    source.set_property("drop-on-latency", True)
    source.set_property("protocols",       4)       # TCP only
    source.set_property("tcp-timeout",     5000000) # 5 s keepalive — prevents Dahua 180 s session cut

    decoder.set_property("drop-frame-interval", 0)

    for el in [source, decoder]:
        pipeline.add(el)

    decoder_srcpad    = decoder.get_static_pad("src")
    streammux_sinkpad = streammux.get_request_pad(f"sink_{stream_idx}")
    decoder_srcpad.link(streammux_sinkpad)

    def _on_pad_added(_src, pad, _pipeline=pipeline, _decoder=decoder):
        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_str = caps.to_string() if caps else ""
        if "video" not in caps_str:
            return

        if "H265" in caps_str.upper():
            depay  = Gst.ElementFactory.make("rtph265depay", f"depay-{stream_idx}")
            parser = Gst.ElementFactory.make("h265parse",    f"parser-{stream_idx}")
            logger.info("Stream %d: H.265 codec detected", stream_idx)
        else:
            depay  = Gst.ElementFactory.make("rtph264depay", f"depay-{stream_idx}")
            parser = Gst.ElementFactory.make("h264parse",    f"parser-{stream_idx}")

        if not depay or not parser:
            logger.error("Could not create depay/parser for stream %d", stream_idx)
            return

        _pipeline.add(depay)
        _pipeline.add(parser)
        depay.sync_state_with_parent()
        parser.sync_state_with_parent()

        depay.link(parser)
        parser.link(_decoder)

        sink = depay.get_static_pad("sink")
        if not sink.is_linked():
            pad.link(sink)

    source.connect("pad-added", _on_pad_added)
    logger.info(
        "Stream %d → %s", stream_idx,
        rtsp_url.replace(
            rtsp_url.split("@")[0].split("//")[1] if "@" in rtsp_url else "", "***:***"
        ),
    )


def main():
    cfg = load_config()
    cfg.log_summary()
    _validate_pipeline_models(cfg.pipeline)
    init_channel_map(cfg.channels)
    init_sector(cfg.sector)
    init_entry_exit_pads(cfg.entry_exit_pad_indices())

    client_dir = Path(__file__).resolve().parent.parent / "clients" / cfg.client_name
    face_db_path = str(client_dir / "known_faces.json")
    ws_base_url = os.environ.get("WS_BASE_URL", "").strip()
    api_key = os.environ.get("API_KEY", "")
    init_workers(
        cfg.pipeline,
        model_dir=str(_MODELS_DIR),
        face_db_path=face_db_path,
        ws_base_url=ws_base_url,
        api_key=api_key,
    )
    init_handlers(cfg.pipeline)

    urls = cfg.rtsp_urls()
    n_streams = len(urls)

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ── Streammux ──────────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property("width",                cfg.stream_width)
    streammux.set_property("height",               cfg.stream_height)
    streammux.set_property("batch-size",           n_streams)
    streammux.set_property("batched-push-timeout", 33333)
    streammux.set_property("live-source",          1)
    pipeline.add(streammux)

    for i, url in enumerate(urls):
        _add_rtsp_source(pipeline, streammux, url, i)

    # ── PGIE — PeopleNet (always active) ─────────────────────────────────────
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path",
                      "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt")
    if cfg.pgie_batch_size > 0:
        pgie.set_property("batch-size", cfg.pgie_batch_size)
        logger.info("PGIE batch-size overridden to %d from config.yaml", cfg.pgie_batch_size)
    if cfg.pgie_interval >= 0:
        pgie.set_property("interval", cfg.pgie_interval)
        logger.info("PGIE interval overridden to %d from config.yaml", cfg.pgie_interval)

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width",  320)
    tracker.set_property("tracker-height", 192)
    tracker.set_property("gpu-id",         0)
    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", cfg.tracker_config_path())
    tracker.set_property("display-tracking-id", 1)

    # ── SGIEs — one per active capability beyond people_counting ──────────────
    sgie_elements = []
    for cap in cfg.active_sgies():
        cfg_path = SGIE_CONFIGS.get(cap)
        if cfg_path is None:
            logger.info("Capability '%s' uses Python worker — skipping SGIE", cap)
            continue
        sgie = Gst.ElementFactory.make("nvinfer", f"sgie-{cap}")
        if not sgie:
            logger.error("Could not create nvinfer element for capability '%s'", cap)
            sys.exit(1)
        sgie.set_property("config-file-path", cfg_path)
        sgie_elements.append(sgie)
        logger.info("SGIE loaded: %s → %s", cap, cfg_path)

    if not sgie_elements:
        logger.info("No SGIEs loaded — running people_counting only")

    # ── Tiler — combines N streams into one tiled frame ───────────────────────
    tiler_cols = math.ceil(math.sqrt(n_streams))
    tiler_rows = math.ceil(n_streams / tiler_cols)
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler.set_property("rows",    tiler_rows)
    tiler.set_property("columns", tiler_cols)
    tiler.set_property("width",   640)
    tiler.set_property("height",  360)

    # ── NV12→RGBA (probe needs RGBA for crop extraction) ─────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",     "capsfilter-rgba")
    caps_rgba.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    # ── fakesink — probe extracts all needed crops from RGBA before here ─────
    fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
    fakesink.set_property("sync", False)

    fixed_elements = [pgie, tracker, tiler, nvvidconv1, caps_rgba, fakesink]
    if not all(fixed_elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)

    for el in fixed_elements + sgie_elements:
        pipeline.add(el)

    # ── Linking ───────────────────────────────────────────────────────────────
    streammux.link(pgie)
    pgie.link(tracker)

    # Chain SGIEs in sequence after tracker, then continue to tiler
    prev = tracker
    for sgie in sgie_elements:
        prev.link(sgie)
        prev = sgie
    prev.link(tiler)

    tiler.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)
    caps_rgba.link(fakesink)

    # ── Probe — caps_rgba src-pad sees RGBA (needed for crop extraction) ──────
    caps_rgba_src_pad = caps_rgba.get_static_pad("src")
    caps_rgba_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)

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
            src_name = message.src.get_name() if message.src else ""
            if src_name.startswith("source-"):
                logger.warning("RTSP '%s' failed: %s — pipeline continues.", src_name, err)
            else:
                logger.error("GStreamer ERROR: %s — %s", err, dbg)
                loop.quit()

    bus.connect("message", _on_bus_message)

    logger.info("Starting pipeline…")
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
