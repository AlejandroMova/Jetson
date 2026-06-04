"""
app.py — NX Computing AI | Production Pipeline (Live DVR / RTSP)

Source: RTSP stream(s) from DVR, configured per-client via config_loader.
Sink  : fakesink — no display output en producción.

Pipeline (capabilities driven by config.yaml `pipeline` field or /etc/nx_pipeline):
  rtspsrc → rtph264depay → h264parse → nvv4l2decoder
    → nvstreammux → nvinfer (PeopleNet PGIE) → nvtracker
    → [nvinfer SGIE per active capability]
    → nvvideoconvert → capsfilter(RGBA)
    → [Probe A: analytics full-res por cámara] → fakesink

Stream mode (NX_STREAM_ENABLED=true):
  → nvvideoconvert → capsfilter(RGBA)
  → [Probe A: analytics full-res, guarda labels en _track_labels]
  → nvmultistreamtiler(640×360)
  → [Probe B: tiled_overlay_probe — dibuja bboxes → tiled_frame_queue]
  → fakesink
  MjpegServer(:8080) sirve tiled_frame_queue en /stream/all y /viewer/all.
  Activar con: ./stream.sh  (desde deploy/)
"""

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
    osd_sink_pad_buffer_probe, tiled_overlay_probe, api_client,
    init_channel_map, init_sector, init_entry_exit_pads, init_camera_types,
    init_handlers, init_workers, start_workers, stop_workers,
    init_stream_grid, tiled_frame_queue,
    _IS_STREAM_ENABLED,
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
         → nvvidconv → RGBA capsfilter → fakesink
      5. Adjuntar Probe A en caps_rgba src-pad (analytics full-res)
      6. Si NX_STREAM_ENABLED: insertar nvmultistreamtiler + Probe B + MjpegServer en :8080
      7. Correr el GLib.MainLoop hasta EOS o error
      8. Parar workers, cliente API y pipeline al salir
    """
    cfg = load_config()
    cfg.log_summary()
    _validate_pipeline_models(cfg.pipeline)
    init_channel_map(cfg.channels)
    init_sector(cfg.sector)
    init_entry_exit_pads(cfg.entry_exit_pad_indices())
    init_camera_types(cfg.external_pad_indices(), cfg.count_internal, cfg.count_external)

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

    # ── NV12→RGBA (probe needs RGBA for crop extraction) ─────────────────────
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba  = Gst.ElementFactory.make("capsfilter",     "capsfilter-rgba")
    caps_rgba.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
    fakesink.set_property("sync", False)

    # ── Stream mode: tiler (compone todas las cámaras en 640×360) ─────────────
    tiler = None
    if _IS_STREAM_ENABLED:
        tiler_cols = math.ceil(math.sqrt(n_streams))
        tiler_rows = math.ceil(n_streams / tiler_cols)
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            logger.error("[Stream] No se pudo crear nvmultistreamtiler — abortando.")
            sys.exit(1)
        tiler.set_property("rows",    tiler_rows)
        tiler.set_property("columns", tiler_cols)
        tiler.set_property("width",   640)
        tiler.set_property("height",  360)
        pipeline.add(tiler)
        init_stream_grid(tiler_cols, tiler_rows, 640 // tiler_cols, 360 // tiler_rows)
        from mjpeg_server import MjpegServer
        _mjpeg_srv = MjpegServer(tiled_queue=tiled_frame_queue, port=8080)
        _mjpeg_srv.start()
        logger.info("[Stream] MjpegServer en :8080 — /stream/all  /viewer/all")

    fixed_elements = [pgie, tracker, nvvidconv1, caps_rgba, fakesink]
    if not all(fixed_elements):
        logger.error("Failed to create one or more pipeline elements.")
        sys.exit(1)

    for el in fixed_elements + sgie_elements:
        pipeline.add(el)

    # ── Linking ───────────────────────────────────────────────────────────────
    streammux.link(pgie)
    pgie.link(tracker)

    prev = tracker
    for sgie in sgie_elements:
        prev.link(sgie)
        prev = sgie

    prev.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)

    if _IS_STREAM_ENABLED and tiler is not None:
        # Stream: caps_rgba → tiler → fakesink
        caps_rgba.link(tiler)
        tiler.link(fakesink)
    else:
        # Producción: caps_rgba → fakesink
        caps_rgba.link(fakesink)

    # ── Probe A — analytics full-res por cámara (siempre activo) ─────────────
    caps_rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)
    logger.info("Probe A en caps_rgba src-pad (frames full-res por cámara)")

    # ── Probe B — overlay tileado (solo stream mode) ──────────────────────────
    if _IS_STREAM_ENABLED and tiler is not None:
        tiler.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, tiled_overlay_probe)
        logger.info("Probe B en tiler src-pad (frame tileado 640×360)")

    # ── Run ───────────────────────────────────────────────────────────────────
    api_client.start()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()

    # ── DVR IP auto-rediscovery ───────────────────────────────────────────────
    # Si TODOS los streams RTSP fallan en los primeros 60 s de arranque, es señal
    # de que el DVR cambió de IP por DHCP. En ese caso se corre nmap para encontrar
    # la nueva IP, se actualiza /etc/nx_dvr_ip y se reinicia el pipeline (exit 0).
    _pipeline_start_time  = time.monotonic()
    _failed_sources: set  = set()
    _exit_for_rediscover  = [False]
    _STARTUP_WINDOW_S     = 60   # solo actuar si todos fallan dentro de los primeros 60 s

    def _try_rediscover_dvr() -> None:
        """Corre en background thread cuando todas las fuentes RTSP fallan al arrancar.

        Ejecuta nmap en la subred /24 del DVR actual buscando un host con el puerto
        554 abierto. Si encuentra una IP distinta a la actual:
          1. Escribe la nueva IP en /etc/nx_dvr_ip
          2. Pide al main loop que termine (exit code 0)
          3. docker-entrypoint.sh detecta el exit 0 y reinicia app.py con la nueva IP

        Solo se lanza una vez por ciclo de vida del pipeline (guardado por _exit_for_rediscover).
        """
        import subprocess

        current_ip = cfg.dvr_ip
        try:
            import ipaddress
            subnet = str(ipaddress.ip_network(f"{current_ip}/24", strict=False))
        except Exception as e:
            logger.error("[DVR] No se pudo derivar subred de %s: %s", current_ip, e)
            return

        logger.info("[DVR] Escaneando %s en puerto 554 (nmap -T4, ~15 s)...", subnet)
        try:
            result = subprocess.run(
                ["nmap", "-p", "554", subnet, "--open", "-T4", "-oG", "-"],
                capture_output=True, text=True, timeout=90,
            )
            new_ip = None
            for line in result.stdout.splitlines():
                if line.startswith("Host:") and "554/open" in line:
                    candidate = line.split()[1]
                    if candidate != current_ip:
                        new_ip = candidate
                        break
        except Exception as e:
            logger.error("[DVR] nmap falló: %s", e)
            return

        if new_ip:
            logger.info("[DVR] DVR encontrado en nueva IP: %s → %s — actualizando y reiniciando", current_ip, new_ip)
            try:
                Path("/etc/nx_dvr_ip").write_text(new_ip)
            except Exception as e:
                logger.error("[DVR] No se pudo escribir /etc/nx_dvr_ip: %s", e)
                return
            _exit_for_rediscover[0] = True
            GLib.idle_add(loop.quit)
        else:
            logger.warning("[DVR] DVR no encontrado en %s. ¿Está apagado o cambió de subred?", subnet)

    def _on_bus_message(_bus, message):
        """Maneja mensajes del bus GStreamer: EOS, WARNING y ERROR.

        Los errores en fuentes RTSP individuales (source-N) no detienen el pipeline —
        el stream simplemente se pierde pero los demás continúan. Solo los errores
        en elementos core (mux, PGIE, tracker) detienen el pipeline completo.

        Si TODAS las fuentes configuradas fallan dentro del primer minuto de arranque,
        se asume cambio de IP del DVR y se lanza _try_rediscover_dvr en background.
        """
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
                _failed_sources.add(src_name)
                # Trigger rediscovery if ALL configured sources failed in the startup window
                elapsed = time.monotonic() - _pipeline_start_time
                if (not _exit_for_rediscover[0]
                        and elapsed < _STARTUP_WINDOW_S
                        and len(_failed_sources) >= len(cfg.channels)):
                    logger.warning(
                        "[DVR] Todos los %d streams fallaron en %.0f s — probablemente el DVR cambió de IP",
                        len(cfg.channels), elapsed,
                    )
                    import threading as _th
                    _th.Thread(target=_try_rediscover_dvr, name="dvr-rediscover", daemon=True).start()
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

    if _exit_for_rediscover[0]:
        logger.info("[DVR] Reiniciando pipeline con nueva IP del DVR.")
        sys.exit(0)


if __name__ == "__main__":
    main()
