"""
app.py — NX Computing AI | Production Pipeline (Live DVR / RTSP)

Source: RTSP stream(s) from DVR, configured per-client via config_loader.
Sink  : fakesink — no display output.

Pipeline (capabilities driven by config.yaml `pipeline` field or /etc/nx_pipeline):
  rtspsrc → rtph264depay → h264parse → nvv4l2decoder
    → nvstreammux → nvinfer (PeopleNet PGIE) → nvtracker
    → [nvinfer SGIE per active capability]
    → nvmultistreamtiler(640×360) → nvvideoconvert → capsfilter(RGBA)
    → [probe: crops para analytics] → fakesink

QA Visual (NX_QA_ENABLED=true):
  El probe dibuja bboxes/labels en el frame tileado y los sirve vía MjpegServer (:8080).
  Metadata se publica a Redis pub/sub para el dashboard Streamlit (:8501).
  Activar con: ./qa.sh  (desde deploy/)
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from config_loader import load_config
from probes import (
    osd_sink_pad_buffer_probe, pre_tiler_analytics_probe, api_client,
    init_channel_map, init_sector, init_entry_exit_pads, init_camera_types,
    init_handlers, init_workers, start_workers, stop_workers,
    init_qa_grid, init_qa_cameras, init_pipeline_stats,
    set_recording_manager,
    tiled_frame_queue, camera_frame_queues,
    _IS_QA_ENABLED, _redis_qa,
)

# Maps each pipeline capability to its nvinfer config file (relative to deploy/).
# None = Python worker (no SGIE element created for that capability).
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# nvinfer config keys whose values are file paths. When the config is copied to a
# temp file in /tmp/, DeepStream resolves relative paths relative to /tmp/ — which
# breaks all model references. These keys must be rewritten as absolute paths.
_NVINFER_PATH_KEYS = frozenset({
    "onnx-file", "model-engine-file", "labelfile-path",
    "int8-calib-file", "tlt-encoded-model", "custom-lib-path",
})
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


def _apply_pgie_overrides(original_path: str, cfg) -> str:
    """Return a path to a (possibly modified) nvinfer config for the PGIE.

    DeepStream reads the nvinfer config from a file — there is no GStreamer property
    to override per-class-attrs at runtime. To support per-client tuning from config.yaml
    without modifying the shared nvinfer_config.txt, this function generates a
    temporary copy with the overridden values substituted in [class-attrs-all].

    If no overrides are set in config.yaml, the original file path is returned as-is
    and no file is written (zero overhead for the common case).

    The temp file at /tmp/ persists for the lifetime of the process — it is NOT cleaned
    up on exit because DeepStream holds an open reference to it while the pipeline runs.

    Overridable parameters (set in config.yaml as pgie_topk / pgie_nms_iou_threshold /
    pgie_pre_cluster_threshold):
      - topk: max detections per inference frame. Raise if >20 people in scene.
      - nms-iou-threshold: two boxes with IoU > this are merged into one.
                           Lower it (e.g. 0.3) so adjacent people aren't collapsed.
      - pre-cluster-threshold: minimum confidence a box must have before NMS.
                               Lower it (e.g. 0.2) to catch occluded / low-confidence people.
    """
    # Determine if there are any overrides to apply
    has_overrides = (
        cfg.pgie_topk > 0
        or cfg.pgie_nms_iou_threshold >= 0.0
        or cfg.pgie_pre_cluster_threshold >= 0.0
    )
    if not has_overrides:
        # No overrides → use the original file directly; nothing to write
        return original_path

    import tempfile

    # ── Parse the original config line by line ────────────────────────────────
    # We can't use configparser because it doesn't preserve comments and nvinfer
    # is picky about the exact format of its config file.
    # Resolve to absolute so relative path keys can be rewritten correctly.
    model_dir = Path(original_path).resolve().parent
    lines = Path(original_path).read_text().splitlines()
    in_class_attrs = False  # flag: are we inside [class-attrs-all] right now?
    result = []

    for line in lines:
        stripped = line.strip()

        # Detect section headers like [property] or [class-attrs-all]
        if stripped.startswith("["):
            in_class_attrs = stripped == "[class-attrs-all]"
            result.append(line)
            continue

        if "=" in stripped and not stripped.startswith("#"):
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip()

            # Rewrite relative file paths as absolute — nvinfer resolves paths relative
            # to the config file location, so a temp file in /tmp/ would break them all.
            if key in _NVINFER_PATH_KEYS and val and not Path(val).is_absolute():
                result.append(f"{key}={model_dir / val}")
                continue

            # Threshold overrides — only inside [class-attrs-all]
            if in_class_attrs:
                if key == "topk" and cfg.pgie_topk > 0:
                    # Limit max detections per frame — raise if crowded scenes miss people
                    result.append(f"topk={cfg.pgie_topk}")
                    logger.info("PGIE override: topk=%d (was %s)", cfg.pgie_topk, stripped)
                    continue

                if key == "nms-iou-threshold" and cfg.pgie_nms_iou_threshold >= 0.0:
                    # Lower this if adjacent people are being merged into one bbox
                    result.append(f"nms-iou-threshold={cfg.pgie_nms_iou_threshold}")
                    logger.info("PGIE override: nms-iou-threshold=%.3f (was %s)",
                                cfg.pgie_nms_iou_threshold, stripped)
                    continue

                if key == "pre-cluster-threshold" and cfg.pgie_pre_cluster_threshold >= 0.0:
                    # Lower this to detect occluded / partially visible people (more noise too)
                    result.append(f"pre-cluster-threshold={cfg.pgie_pre_cluster_threshold}")
                    logger.info("PGIE override: pre-cluster-threshold=%.3f (was %s)",
                                cfg.pgie_pre_cluster_threshold, stripped)
                    continue

        result.append(line)

    # ── Write the modified config to a temp file ──────────────────────────────
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="nx_pgie_runtime_"
    )
    tmp.write("\n".join(result))
    tmp.close()
    logger.info("PGIE runtime config written to %s", tmp.name)
    return tmp.name


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
        """Callback invocado cuando rtspsrc negocia el codec con el DVR.

        Crea el depayloader y parser correctos según el codec detectado (H.264 o H.265).
        Se necesita hacer esto dinámicamente porque el codec se conoce solo tras
        la negociación SDP con el DVR — no antes de conectarse.
        """
        # Leer caps del pad recién creado para detectar el codec
        caps = pad.get_current_caps() or pad.query_caps(None)
        caps_str = caps.to_string() if caps else ""
        if "video" not in caps_str:
            return  # ignorar pads de audio o control

        # Crear depayloader + parser según codec detectado
        if "H265" in caps_str.upper():
            depay  = Gst.ElementFactory.make("rtph265depay", f"depay-{stream_idx}")
            parser = Gst.ElementFactory.make("h265parse",    f"parser-{stream_idx}")
            logger.info("Stream %d: H.265 codec detected", stream_idx)
        else:
            # H.264 es el default — la mayoría de DVRs usan H.264 en canales principales
            depay  = Gst.ElementFactory.make("rtph264depay", f"depay-{stream_idx}")
            parser = Gst.ElementFactory.make("h264parse",    f"parser-{stream_idx}")

        if not depay or not parser:
            logger.error("Could not create depay/parser for stream %d", stream_idx)
            return

        # Agregar elementos al pipeline y sincronizar estado con el padre
        _pipeline.add(depay)
        _pipeline.add(parser)
        depay.sync_state_with_parent()   # evita que el pipeline quede en estado inconsistente
        parser.sync_state_with_parent()

        # Encadenar: depay → parser → decoder
        depay.link(parser)
        parser.link(_decoder)

        # Conectar el pad de rtspsrc al sink del depayloader (solo si no está ya conectado)
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
    """Punto de entrada del pipeline de producción.

    Flujo completo:
      1. Cargar configuración del cliente (config.yaml + env vars + /etc/nx_*)
      2. Validar que todos los modelos requeridos existen antes de iniciar GStreamer
      3. Inicializar mapas de canales, sector, handlers y workers async
      4. Construir el grafo GStreamer: rtspsrc × N → mux → PGIE → tracker → SGIEs
         → nvvidconv → RGBA capsfilter → [tiler si QA] → fakesink
      5. Adjuntar probes (analytics en full-res; overlays en tileado si QA)
      6. Correr el GLib.MainLoop hasta EOS, error, o solicitud de playback (código 42)
      7. Parar workers, cliente API, pipeline y RecordingManager al salir

    En QA mode (NX_QA_ENABLED=true) también:
      - Levanta MjpegServer en :8080
      - Publica nx:qa:status a Redis con toda la configuración del Jetson
      - Inicia polling cada 5 s para detectar solicitud de modo playback desde Streamlit
    """
    cfg = load_config()
    cfg.log_summary()
    _validate_pipeline_models(cfg.pipeline)
    init_channel_map(cfg.channels)
    init_sector(cfg.sector)
    init_entry_exit_pads(cfg.entry_exit_pad_indices())
    init_camera_types(cfg.external_pad_indices(), cfg.count_internal, cfg.count_external)
    # QA: inicializar queues de cámara (se usan en el probe y en MjpegServer)
    if _IS_QA_ENABLED:
        init_qa_cameras(cfg.channels)

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
        reid_gallery_size=cfg.reid_gallery_size,
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
    # _apply_pgie_overrides writes a temp config if config.yaml has threshold overrides;
    # otherwise returns the original path unchanged (no I/O cost).
    _pgie_config_path = _apply_pgie_overrides(
        "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt", cfg
    )
    pgie.set_property("config-file-path", _pgie_config_path)
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
        if cfg.sgie_interval >= 0:
            sgie.set_property("interval", cfg.sgie_interval)
        sgie_elements.append(sgie)
        logger.info("SGIE loaded: %s → %s", cap, cfg_path)

    if not sgie_elements:
        logger.info("No SGIEs loaded — running people_counting only")

    # ── Tiler — solo en QA mode para compositar el video preview ─────────────
    # En producción el probe recibe frames full-res por cámara directamente.
    tiler_cols = math.ceil(math.sqrt(n_streams))
    tiler_rows = math.ceil(n_streams / tiler_cols)
    tiler = None
    if _IS_QA_ENABLED:
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows",    tiler_rows)
        tiler.set_property("columns", tiler_cols)
        tiler.set_property("width",   640)
        tiler.set_property("height",  360)

    # ── RecordingManager — activo cuando recording_enabled=true en config.yaml ─
    # En QA mode siempre está activo (independiente del valor en config.yaml).
    # En producción solo si cfg.recording_enabled=true.
    _recording_manager = None
    if cfg.recording_enabled or _IS_QA_ENABLED:
        from recording_manager import RecordingManager
        _recording_manager = RecordingManager(
            recordings_dir="/nx_tech/recordings",
            redis_client=_redis_qa,   # None en producción sin QA; se ignora elegantemente
        )
        set_recording_manager(_recording_manager)
        _recording_manager.start()
        logger.info("[Recording] RecordingManager iniciado — /nx_tech/recordings/")

    # QA: informar grid al probe + arrancar MjpegServer
    if _IS_QA_ENABLED:
        cell_w = 640 // tiler_cols
        cell_h = 360 // tiler_rows
        init_qa_grid(tiler_cols, tiler_rows, cell_w, cell_h)

        from mjpeg_server import MjpegServer
        _mjpeg_srv = MjpegServer(
            tiled_frame_queue=tiled_frame_queue,
            camera_queues=camera_frame_queues,
            port=8080,
            recorder=_recording_manager,
        )
        _mjpeg_srv.start()
        logger.info("[QA] MjpegServer en :8080  /stream/all + /stream/<camera_id>")
        # Publicar status del Jetson a Redis para que Streamlit lo muestre
        if _redis_qa:
            _redis_qa.set("nx:qa:status", json.dumps({
                "client":              cfg.client_name,
                "package":             cfg.package,
                "capabilities":        cfg.pipeline,
                "channels":            cfg.channels,
                "tracker":             cfg.tracker,
                "sector":              cfg.sector,
                "tiler_cols":          tiler_cols,
                "tiler_rows":          tiler_rows,
                "jetson_id":           os.environ.get("JETSON_ID", ""),
                "entry_exit_channels": cfg.entry_exit_channels,
                "external_channels":   cfg.external_channels,
                "count_internal":      cfg.count_internal,
                "count_external":      cfg.count_external,
                "stream_width":        cfg.stream_width,
                "stream_height":       cfg.stream_height,
                "stream_type":         cfg.stream_type,
                "dvr_port":            cfg.dvr_port,
                "rtsp_url_pattern":    cfg.rtsp_url_pattern,
                "pgie_batch_size":     cfg.pgie_batch_size,
                "pgie_interval":       cfg.pgie_interval,
                "sgie_interval":       cfg.sgie_interval,
                "reid_gallery_size":   cfg.reid_gallery_size,
                "recording_enabled":   cfg.recording_enabled,
                "component_resolutions": {
                    "source":           f"{cfg.stream_width}x{cfg.stream_height}",
                    "probe_a_frame":    f"{cfg.stream_width}x{cfg.stream_height}",
                    "probe_b_frame":    "640x360",
                    "pgie_input":       "960x544",
                    "age_gender_input": "224x224",
                    "facedetect_input": "240x136",
                    "movenet_input":    "192x192",
                    "osnet_input":      "128x256",
                },
            }))
            # Nuevo arranque — borrar overrides viejos del config editor y publicar generación
            # para que el dashboard Streamlit detecte el reinicio y resetee su session_state.
            _redis_qa.delete("nx:qa:config_overrides")
            _redis_qa.set("nx:qa:config_gen", str(time.time()))
            # Inicializar nx:qa:entry_exit solo si no existe (preserva cambios QA en vivo)
            if not _redis_qa.exists("nx:qa:entry_exit"):
                import json as _json
                _redis_qa.set("nx:qa:entry_exit", _json.dumps(cfg.entry_exit_channels))
            # external_channels: igual que entry_exit, no sobreescribir si ya existe
            if not _redis_qa.exists("nx:qa:external_channels"):
                _redis_qa.set("nx:qa:external_channels", json.dumps(cfg.external_channels))
            # count_internal/count_external: igual que entry_exit, no sobreescribir si ya existe
            if not _redis_qa.exists("nx:qa:count_internal"):
                _redis_qa.set("nx:qa:count_internal", "1" if cfg.count_internal else "0")
            if not _redis_qa.exists("nx:qa:count_external"):
                _redis_qa.set("nx:qa:count_external", "1" if cfg.count_external else "0")
            init_pipeline_stats(cfg.channels)

    # ── NV12→RGBA (probe needs RGBA for crop extraction) ─────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",     "capsfilter-rgba")
    caps_rgba.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    # ── fakesink — probe extracts all needed crops from RGBA before here ─────
    fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
    fakesink.set_property("sync", False)

    fixed_elements = [pgie, tracker, nvvidconv1, caps_rgba, fakesink]
    if not all(fixed_elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)
    if _IS_QA_ENABLED and not tiler:
        logger.error("Failed to create nvmultistreamtiler in QA mode.")
        sys.exit(1)

    for el in fixed_elements + sgie_elements:
        pipeline.add(el)
    if tiler is not None:
        pipeline.add(tiler)

    # ── Linking ───────────────────────────────────────────────────────────────
    streammux.link(pgie)
    pgie.link(tracker)

    prev = tracker
    for sgie in sgie_elements:
        prev.link(sgie)
        prev = sgie

    # nvvidconv1 siempre va despues de los SGIEs (produccion y QA)
    prev.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)

    if _IS_QA_ENABLED:
        # QA: caps_rgba -> tiler (compuesto 640x360) -> fakesink
        caps_rgba.link(tiler)
        tiler.link(fakesink)
    else:
        # Produccion: caps_rgba -> fakesink (sin tiler)
        caps_rgba.link(fakesink)

    # ── Probe attachment ──────────────────────────────────────────────────────
    caps_rgba_src_pad = caps_rgba.get_static_pad("src")
    if _IS_QA_ENABLED:
        # Probe A: analytics en frames full-res RGBA (pre-tiler)
        caps_rgba_src_pad.add_probe(Gst.PadProbeType.BUFFER, pre_tiler_analytics_probe)
        # Probe B: overlays en frame tileado RGBA (post-tiler)
        tiler_src_pad = tiler.get_static_pad("src")
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)
        logger.info("[QA] Probe A (analytics) en caps_rgba; Probe B (overlays) en tiler src")
    else:
        # Produccion: probe unico con frames full-res
        caps_rgba_src_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)
        logger.info("Probe unico (produccion) en caps_rgba src-pad")

    # ── Run ───────────────────────────────────────────────────────────────────
    api_client.start()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    def _on_bus_message(_bus, message):
        """Maneja mensajes del bus GStreamer: EOS, WARNING y ERROR.

        Los errores en fuentes RTSP individuales (source-N) no detienen el pipeline —
        el stream simplemente se pierde pero los demás continúan. Solo los errores
        en elementos core (mux, PGIE, tracker) detienen el pipeline completo.
        """
        t = message.type
        if t == Gst.MessageType.EOS:
            # Fin de stream — ocurre en modo playback o cuando todas las fuentes RTSP se cierran
            logger.info("EOS received — stopping.")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            # Advertencias no fatales — loguear para diagnóstico sin detener el pipeline
            err, dbg = message.parse_warning()
            logger.warning("GStreamer WARNING: %s — %s", err, dbg)
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            src_name = message.src.get_name() if message.src else ""
            if src_name.startswith("source-"):
                # Error en una cámara RTSP — el resto del pipeline continúa
                logger.warning("RTSP '%s' failed: %s — pipeline continues.", src_name, err)
            else:
                # Error en elemento core — detener todo
                logger.error("GStreamer ERROR: %s — %s", err, dbg)
                loop.quit()

    bus.connect("message", _on_bus_message)

    # QA: detectar solicitud de modo playback desde Streamlit cada 5 s
    _exit_for_playback = [False]
    if _IS_QA_ENABLED and _redis_qa:
        def _check_playback_mode():
            """Polling de Redis cada 5 s para detectar solicitud de playback desde Streamlit.

            Cuando el usuario hace clic en "▶ Correr Inferencia" en el dashboard,
            Streamlit escribe la clave nx:qa:playback_video en Redis con el path del video.
            Este callback la detecta y llama loop.quit() directamente para detener el pipeline.

            NOTA: No usamos pipeline.send_event(Gst.Event.new_eos()) porque rtspsrc es una
            live source y puede ignorar o no propagar el evento EOS downstream. El resultado
            sería que loop.quit() nunca se llama y app.py corre indefinidamente.
            loop.quit() es seguro de llamar desde un callback de GLib (mismo thread que el loop).
            El cleanup del pipeline ocurre en el finally block de loop.run() via
            pipeline.set_state(Gst.State.NULL).
            """
            try:
                v = _redis_qa.get("nx:qa:playback_video")
                if v and not _exit_for_playback[0]:
                    _exit_for_playback[0] = True  # evitar múltiples quit()
                    logger.info("[QA] Modo playback solicitado: %s — deteniendo pipeline", v)
                    loop.quit()
            except Exception:
                pass
            return True  # retornar True mantiene el timeout de GLib activo
        GLib.timeout_add(5000, _check_playback_mode)

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
        if _recording_manager is not None:
            _recording_manager.stop()

    # Salir con código 42 para que docker-entrypoint.sh arranque modo playback
    if _exit_for_playback[0]:
        sys.exit(42)


if __name__ == "__main__":
    main()
