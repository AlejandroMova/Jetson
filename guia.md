📘 DeepStream SDK: Guía Definitiva de Construcción de Pipelines
Resumen arquitectónico basado en los tutoriales oficiales de NVIDIA DLI.

Un pipeline de DeepStream es como una línea de ensamblaje industrial para videos. Los datos entran por un lado, se decodifican, pasan por la Inteligencia Artificial (Inferencia), se les dibujan cajas (OSD), y salen por el otro lado.

1. 🎬 Inicialización y Creación del Pipeline
Antes de crear elementos, debes encender el motor de GStreamer y crear el contenedor principal ("Pipeline").

Python

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
import pyds # <-- Bindings de Python para DeepStream

# 1. Inicializar el motor interno de GStreamer
Gst.init(None)

# 2. Crear el contenedor principal
pipeline = Gst.Pipeline()
2. 🧩 Creación de Elementos (Nodos del Grafo)
Cada tarea requiere un "Elemento" específico. Se crean usando Gst.ElementFactory.make("tipo_de_plugin", "nombre_personalizado").

Python

# A. LECTURA Y DECODIFICACIÓN
source = Gst.ElementFactory.make("filesrc", "file-source")
source.set_property('location', "video.h264") # Ruta del video

h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

# B. AGRUPADOR (Muxer) - Obligatorio antes de la IA
streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
streammux.set_property('width', 960) 
streammux.set_property('height', 544) 
streammux.set_property('batch-size', 1)

# C. CEREBRO DE INTELIGENCIA ARTIFICIAL (TensorRT)
pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
pgie.set_property('config-file-path', "nvinfer_config.txt") # Conecta tu modelo aquí

# D. GRÁFICOS Y PANTALLA (OSD)
nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay") # Dibuja las cajas rojas

# E. SALIDA (Sink)
sink = Gst.ElementFactory.make('nveglglessink', 'pantalla-en-vivo')
sink.set_property("sync", 0) # 0 = Máxima velocidad GPU, 1 = Velocidad de video normal

# ¡IMPORTANTE! Añadir todo al pipeline:
elementos = [source, h264parser, decoder, streammux, pgie, nvvidconv1, nvosd, sink]
for el in elementos:
    pipeline.add(el)
3. 🔗 Enlace de los Elementos (Linking)
Los elementos deben conectarse en el orden exacto en el que fluye el video. El puente entre el Decoder y el Muxer requiere un trato especial ("Request Pad").

Python

# Enlace secuencial normal
source.link(h264parser)
h264parser.link(decoder)

# ENLACE ESPECIAL: Decoder -> Streammux
decoder_srcpad = decoder.get_static_pad("src")
streammux_sinkpad = streammux.get_request_pad("sink_0")
decoder_srcpad.link(streammux_sinkpad)

# Continuación del enlace normal
streammux.link(pgie)
pgie.link(nvvidconv1)
nvvidconv1.link(nvosd)
nvosd.link(sink)
4. 🧠 Intercepción de Datos: El "Probe" (Magia de Metadatos)
El probe es un espía que colocas entre dos elementos para leer los resultados de la IA (Metadatos) en cada frame, sin detener el video.

Python

def extraer_metadatos_probe(pad, info):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
            
        print(f"Frame: {frame_meta.frame_num} | Objetos: {frame_meta.num_obj_meta}")
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return Gst.PadProbeReturn.OK

# Conectar el espía (Probe) a la salida del OSD
osd_sinkpad = nvosd.get_static_pad("sink")
osd_sinkpad.add_probe(Gst.PadProbeType.BUFFER, extraer_metadatos_probe)
5. 🚀 Arranque y Control (Bus & MainLoop)
El MainLoop es el corazón que mantiene el programa vivo, y el Bus es el sistema de notificaciones (errores, fin del video).

Python

def bus_call(bus, message, loop):
    """ Escucha los mensajes de error o cuando el video termina """
    t = message.type
    if t == Gst.MessageType.EOS: # End Of Stream
        print("Fin del video.")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    return True

# 1. Crear Loop y conectar el Bus
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

# 2. ¡Encender la máquina!
print("Iniciando Pipeline de IA...")
pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run() # El programa se queda girando aquí
except KeyboardInterrupt:
    pass

# 3. Apagar y limpiar memoria
pipeline.set_state(Gst.State.NULL)