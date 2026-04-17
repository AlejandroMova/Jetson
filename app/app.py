import sys
import logging
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds
from probes import bus_call, osd_sink_pad_buffer_probe, api_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# funcion principal
def main():
    # Inicializar GStreamer
    Gst.init(None)

    # Crear el pipeline principal
    pipeline = Gst.Pipeline()
    logger.info("Pipeline creado.")

    # creacion de elementos
    # 1. fuente de video
    source = Gst.ElementFactory.make("filesrc", "file-source")
    # source.set_property('location', "sample_30.h264")
    source.set_property('location', "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264")

    # 2. Parseador y Decodificador (Prepara el video para la GPU)
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

    # TODO quitar esta linea
    decoder.set_property("drop-frame-interval", 0)

    # 3. Streammux (Agrupa los frames en lotes para la inferencia)
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 960) # Coincide con tu config infer-dims=3;544;960
    streammux.set_property('height', 544)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 33333)   # ~1 frame @ 30 fps (µs)

    # 4. Motor de Inferencia Primario (peoplenet)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt")

    # 4b. Tracker — asigna IDs persistentes entre frames.
    #     Necesario para que el SGIE async encuentre el mismo objeto en frames posteriores
    #     y para deduplicar personas ya clasificadas en probes.py.
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('gpu-id', 0)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
    tracker.set_property('display-tracking-id', 1)

    # 5. age_gender
    secondary_inference = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    secondary_inference.set_property('config-file-path', "models/resnet_age_gender_FB2/config_infer.txt")

    # 6. Conversores y OSD (On-Screen Display para dibujar las cajas)
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    # capsfilter fuerza RGBA en el sink-pad del OSD para que el probe pueda
    # extraer frames numpy (cv2.COLOR_RGBA2BGR) para crops y frame de referencia
    caps_rgba  = Gst.ElementFactory.make("capsfilter", "capsfilter-rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    nvosd      = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    # nvvidconv2 reconvierte la salida del OSD al formato preferido por el sink
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")

    # 7. Sumidero: Pantalla en vivo (Live Window)
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    # sync=0: pipeline no se throttlea al reloj del video.
    # Para file-source, webcam y RTSP siempre debe ser 0; de lo contrario
    # el sink impone un techo de FPS igual a la velocidad de la fuente.
    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    elements = [source, h264parser, decoder, streammux, pgie, tracker,
                secondary_inference, nvvidconv1, caps_rgba, nvosd, nvvidconv2, sink]

    if not all(elements):
        sys.stderr.write("Error: No se pudieron crear todos los elementos.\n")
        sys.exit(1)

    logger.info("Elementos creados exitosamente.")

    for el in elements:
        pipeline.add(el)

    # linking de elementos
    source.link(h264parser)
    h264parser.link(decoder)

    decoder_srcpad = decoder.get_static_pad("src")
    streammux_sinkpad = streammux.get_request_pad("sink_0")
    decoder_srcpad.link(streammux_sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(secondary_inference)
    secondary_inference.link(nvvidconv1)
    nvvidconv1.link(caps_rgba)
    caps_rgba.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(sink)

    # Conectar el probe al sink-pad del nvdsosd
    # El probe se ejecuta en cada frame ANTES de que el OSD lo dibuje,
    # lo que permite modificar los textos y colores de las cajas.
    osd_sink_pad = nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Error: No se pudo obtener el sink pad del OSD.\n")
        sys.exit(1)
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)
    logger.info("Probe conectado al sink pad del OSD.")

    # Iniciar el cliente de API en su hilo de fondo
    api_client.start()

    # EJECUCION
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    logger.info("Iniciando el pipeline... (Puede tardar unos minutos si está construyendo el motor TensorRT)")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Ejecución cancelada por el usuario.")
    except Exception as e:
        logger.error("Error inesperado: %s", e)
    finally:
        logger.info("Limpiando y apagando el pipeline...")
        pipeline.set_state(Gst.State.NULL)
        api_client.stop()

if __name__ == '__main__':
    main()
