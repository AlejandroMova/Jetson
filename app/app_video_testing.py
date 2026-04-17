import sys
import os
import logging
import gi

# Mitigación X11 shared memory en Docker/x86 — debe ir antes de cualquier init de GStreamer/EGL
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
from probes import osd_sink_pad_buffer_probe, api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEST_VIDEOS_DIR = "test_videos"


def get_video_path() -> str:
    """Devuelve el video a procesar: argumento CLI o el primero en test_videos/."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        videos = sorted(
            os.path.join(TEST_VIDEOS_DIR, f)
            for f in os.listdir(TEST_VIDEOS_DIR)
            if f.lower().endswith((".mp4", ".h264", ".h265", ".mkv"))
        )
    except FileNotFoundError:
        logger.error("Directorio '%s' no encontrado.", TEST_VIDEOS_DIR)
        sys.exit(1)
    if not videos:
        logger.error("No se encontraron videos en %s", TEST_VIDEOS_DIR)
        sys.exit(1)
    return videos[0]


def main():
    Gst.init(None)

    video_path = get_video_path()
    logger.info("Video: %s", video_path)

    pipeline = Gst.Pipeline()

    # 1. Fuente
    source = Gst.ElementFactory.make("filesrc", "file-source")
    source.set_property("location", video_path)

    # 2. Demuxer MP4 (pads dinámicos; se enlazan en on_demux_pad_added)
    qtdemux = Gst.ElementFactory.make("qtdemux", "demuxer")

    # 3. Parser + Decodificador GPU
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    decoder.set_property("drop-frame-interval", 0)

    # 4. Streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 33333)   # ~1 frame @ 30 fps (µs)

    # 5. PGIE — PeopleNet
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt")

    # 5b. Tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id", 0)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    tracker.set_property("display-tracking-id", 1)

    # 6. SGIE — Pedestrian Attributes (cuerpo completo)
    secondary_inference = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    secondary_inference.set_property("config-file-path", "models/resnet_age_gender_FB2/config_infer.txt")

    # 7. OSD
    nvvidconv1  = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    caps_rgba   = Gst.ElementFactory.make("capsfilter", "capsfilter-rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd       = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv2  = Gst.ElementFactory.make("nvvideoconvert", "convertor2")

    # 8. Sink
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    # sync=1 para test con archivo: el video avanza a velocidad natural del clip.
    # En producción (RTSP/webcam) usar sync=0 — la cámara ya actúa como reloj.
    sink.set_property("sync", 1)
    sink.set_property("qos", 0)

    elements = [source, qtdemux, h264parser, decoder, streammux,
                pgie, tracker, secondary_inference, nvvidconv1, caps_rgba, nvosd, nvvidconv2, sink]

    if not all(elements):
        sys.stderr.write("Error: No se pudieron crear todos los elementos.\n")
        sys.exit(1)

    for el in elements:
        pipeline.add(el)

    # qtdemux tiene pads dinámicos — conectar cuando aparezca el pad de video
    def on_demux_pad_added(demux, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        name = caps.to_string() if caps else ""
        if "video/x-h264" in name:
            sinkpad = h264parser.get_static_pad("sink")
            if not sinkpad.is_linked():
                if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
                    logger.error("No se pudo enlazar qtdemux → h264parse")
        elif "video/x-h265" in name:
            logger.error("Video H265 no soportado — reemplaza h264parse por h265parse")

    source.link(qtdemux)
    qtdemux.connect("pad-added", on_demux_pad_added)

    h264parser.link(decoder)

    decoder_srcpad    = decoder.get_static_pad("src")
    streammux_sinkpad = streammux.request_pad_simple("sink_0")
    decoder_srcpad.link(streammux_sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(secondary_inference)
    secondary_inference.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)
    caps_rgba.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(sink)

    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Error: No se pudo obtener el sink pad del OSD.\n")
        sys.exit(1)
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)

    api_client.start()

    def on_bus_message(_bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("EOS — reiniciando video (loop para testing).")
            pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                0,
            )
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            logger.warning("GStreamer WARNING: %s — %s", err, debug)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error("GStreamer ERROR: %s — %s", err, debug)
            loop.quit()

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus_message)

    logger.info("Iniciando pipeline... (primera ejecución construye motores TensorRT)")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Cancelado por el usuario.")
    except Exception as e:
        logger.error("Error inesperado: %s", e)
    finally:
        logger.info("Limpiando pipeline...")
        pipeline.set_state(Gst.State.NULL)
        api_client.stop()


if __name__ == "__main__":
    main()
