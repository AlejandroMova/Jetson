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

def main():
    Gst.init(None)
    pipeline = Gst.Pipeline()
    logger.info("Pipeline de Webcam (MJPG) creado.")

    # 1. Fuente: Webcam configurada para MJPG
    source = Gst.ElementFactory.make("v4l2src", "camera-source")
    source.set_property("device", "/dev/video0")

    # 2. Decodificación de MJPG y conversión
    # Añadimos caps específicos para forzar a la cámara a usar su modo MJPG 720p
    caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2_caps")
    caps_v4l2src.set_property("caps", Gst.Caps.from_string("image/jpeg, width=1280, height=720, framerate=30/1"))
    
    jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
    vidconvsrc = Gst.ElementFactory.make("videoconvert", "vidconvsrc")
    nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "nvvidconvsrc")
    
    # Filtro para entrar a memoria NVMM (GPU)
    caps_nvmm = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    caps_nvmm.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    # 3. Streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    # 4. Inferencia Primaria (PeopleNet)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "models/peoplenet_vpruned_quantized_decrypted_v2.3.4/nvinfer_config.txt")
    pgie.set_property('batch-size', 1)

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

    # 5. Inferencia Secundaria (Age/Gender)
    secondary_inference = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    secondary_inference.set_property('config-file-path', "models/resnet_age_gender_FB/config_infer.txt")

    # 6. OSD y Conversores
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # 7. Sink (Pantalla)
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    sink.set_property("sync", 0) # Latencia cero
    sink.set_property("qos", 0)

    if not all([source, caps_v4l2src, jpegdec, vidconvsrc, nvvidconvsrc, caps_nvmm, streammux, pgie, tracker, secondary_inference, nvvidconv1, nvosd, sink]):
        logger.error("No se pudieron crear los elementos.")
        sys.exit(1)

    # Añadir al pipeline
    for el in [source, caps_v4l2src, jpegdec, vidconvsrc, nvvidconvsrc, caps_nvmm, streammux, pgie, tracker, secondary_inference, nvvidconv1, nvosd, sink]:
        pipeline.add(el)

    # LINKING (Piezas de LEGO)
    source.link(caps_v4l2src)
    caps_v4l2src.link(jpegdec)
    jpegdec.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_nvmm)

    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_nvmm.get_static_pad("src")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(secondary_inference)
    secondary_inference.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(sink)

    # Probe para metadatos
    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe)

    api_client.start()

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    logger.info("Lanzando NX Computing AI con Webcam 720p...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        api_client.stop()

if __name__ == '__main__':
    main()