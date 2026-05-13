# Error History — NX Computing AI

Historial de errores diagnosticados y resueltos en el proyecto. Antes de intentar resolver cualquier error, revisar este archivo para ver si ya existe una solución documentada.

Ver regla 10 de CLAUDE.md para el formato de entradas y el protocolo completo.

---

## 2026-05-12 — cudaErrorIllegalAddress (700) en nvbufsurftransform con 16 cámaras

**Contexto:** Pipeline de producción (`app.py`) con 16 cámaras sub-stream (960×544). Crash a los 20-80 segundos de inicio.

**Error en consola:**
```
NvBufSurfTransformImpl: NvBufSurfTransform failed with error -1
Error in NvBufSurfTransformAsync:3
GST_RETURN_ON_ERR failed with error: -1
cudaErrorIllegalAddress (700) in nvbufsurftransform
NvMapMemAllocInternalTagged: error 12 (ENOMEM)
```

**Causa raíz:** `batch-size=16` en `nvinfer_config.txt` hacía que TensorRT reservara workspace para 16 frames simultáneos. Combinado con 16 decoders NVDEC, nvstreammux, tracker y OSD, la memoria NVMM (unificada en Orin Nano) se agotaba. El crash se aceleraba en reinicios sucesivos por memory leaks entre ciclos.

**Solución:** Cambiar `batch-size=16 → 4` e `interval=2 → 4` en `nvinfer_config.txt`. Actualizar `model-engine-file` de `_b16_` a `_b4_`. Exponer ambos valores como opcionales en `config.yaml` vía `pgie_batch_size` y `pgie_interval` (leídos en `config_loader.py`, overrides en runtime en `app.py`). Borrar el engine `_b16_` del Jetson para forzar recompilación.

---

## 2026-05-13 — Pipeline crashea en loop por RTSP timeout de una sola cámara

**Contexto:** Pipeline de producción (`app.py`), cualquier número de cámaras. Al fallar una fuente RTSP, todo el pipeline moría y reiniciaba indefinidamente.

**Error en consola:**
```
GStreamer ERROR: gst-resource-error-quark: Could not open resource for reading and writing. (7) — ../gst/rtsp/gstrtspsrc.c(8130)
Failed to connect. (Timeout while waiting for server response)
```

**Causa raíz:** El handler de errores del bus GStreamer en `app.py` llamaba `loop.quit()` ante cualquier `GST_MESSAGE_ERROR`, incluyendo timeouts individuales de `rtspsrc`. Con `restart: unless-stopped` en docker-compose, el container reiniciaba inmediatamente, generando un loop infinito de crash/restart.

**Solución:** En `_on_bus_message` de `app.py`, verificar si el error proviene de un elemento `source-N` (rtspsrc). Si es así, loguear como WARNING y continuar. Solo llamar `loop.quit()` para errores de otros elementos (nvinfer, tracker, etc.).

---

## 2026-05-13 — DVR inalcanzable (Destination Host Unreachable) — IP dinámica

**Contexto:** Jetson Orin Nano en red local, DVR Dahua con DHCP. Pipeline falla al conectar con todas las cámaras.

**Error en consola:**
```
From 192.168.10.183 icmp_seq=1 Destination Host Unreachable
RTSP 'source-N' failed: gst-resource-error-quark: Could not open resource for reading and writing. (7)
```

**Causa raíz:** El DVR usa DHCP y cambió de IP (de .14 a .159) tras un reinicio. `/etc/nx_dvr_ip` en el Jetson tenía la IP antigua. El ping devuelve "Destination Host Unreachable" porque la respuesta viene del propio Jetson (ARP falla al no encontrar .14 en la red).

**Solución:** Escanear la red con `nmap -p 554 192.168.10.0/24 --open` para encontrar el dispositivo con RTSP activo. Actualizar `/etc/nx_dvr_ip` con la nueva IP y reiniciar deepstream. **Solución permanente:** asignar IP estática al DVR via DHCP reservation en el router (por MAC address).

---

## 2026-05-13 — nvdsosd crash por agotamiento NVMM con 16 cámaras

**Contexto:** Pipeline de producción (`app.py`) con 16 cámaras sub-stream (960×544) después de reducir batch-size a 4. El pipeline arrancaba y corría ~1-3 minutos antes de crashear.

**Error en consola:**
```
nvbufsurftransform_copy.cpp:438: Failed in mem copy
nvbufsurftransform_copy.cpp:452: NvBufSurfTransformImpl: NvBufSurfTransform failed with error -1
Error in NvBufSurfTransformAsync:3
GST_RETURN_ON_ERR failed with error: -1
cudaErrorIllegalAddress (700) in nvbufsurftransform
```

**Causa raíz:** `nvdsosd` requiere que nvvideoconvert convierta el frame a RGBA en NVMM antes de dibujar bounding boxes. Con 16 streams simultáneos, la asignación de la superficie RGBA adicional para OSD agotaba los buffers NVMM restantes después de que nvinfer, nvtracker y el tiler ya estaban usando la mayor parte del pool.

**Solución:** Eliminar solo `nvdsosd` del pipeline en `app.py` — conservar `nvvidconv1` y `caps_rgba` para que el probe siga recibiendo buffers RGBA (necesario para extracción de crops). El probe se mueve del sink-pad de nvosd al src-pad de caps_rgba, que entrega el mismo formato RGBA. El pipeline nuevo es: `tiler → nvvidconv1(NV12→RGBA) → caps_rgba → nvvidconv2(RGBA→NV12) → caps_nv12 → appsink`. El video MJPEG muestra video limpio sin bounding boxes dibujados; los crops para appearance worker, face recognition y pose worker siguen funcionando.

---

<!-- Agregar entradas aquí siguiendo el formato:

## [Fecha] — Título breve del error

**Contexto:** dónde ocurrió (archivo, componente, etapa del pipeline)

**Error en consola:**
```
<output exacto del error, traceback, o mensaje de log>
```

**Causa raíz:** explicación concisa de por qué ocurría

**Solución:** qué se cambió y en qué archivo(s)

**Fuente externa:** [título](url) — si se consultó documentación, issue, foro o artículo externo

-->
