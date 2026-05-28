# Error History — NX Computing AI

Historial de errores diagnosticados y resueltos en el proyecto. Antes de intentar resolver cualquier error, revisar este archivo para ver si ya existe una solución documentada.

Ver regla 10 de CLAUDE.md para el formato de entradas y el protocolo completo.

---

## 2026-05-27 — QA app inunda logs con warnings de deprecación `use_container_width`

**Contexto:** `deploy/qa_app/streamlit_app.py` — visible al correr `./qa.sh` con Streamlit ≥ 1.46

**Error en consola:**
```
`use_container_width` will be removed after 2025-12-31.
For `use_container_width=True`, use `width='stretch'`.
For `use_container_width=False`, use `width='content'`.
```
(se repite decenas de veces por segundo porque el auto-refresh de 500 ms re-renderiza la página en cada ciclo)

**Causa raíz:** Streamlit deprecó el parámetro `use_container_width` en favor de `width`. El auto-refresh de `st_autorefresh` a 500 ms multiplica el warning por cada render.

**Solución:** Reemplazar `use_container_width=True` → `width="stretch"` en los 3 usos del archivo:
- Línea ~527: `st.button("💾 Guardar en config.yaml", ...)`
- Línea ~531: `st.button("↺ Recargar del pipeline", ...)`
- Línea ~701: `st.image(str(thumb_path), ...)`

---

## 2026-05-18 — ReID cross-cámara no matcheaba: clave de AppearanceWorker sin pad_index

**Contexto:** `deploy/pipelines/appearance_worker.py` + `deploy/pipelines/probes.py` — subsistema de Re-ID cross-cámara.

**Síntoma:** El ReID nunca reconocía que una persona vista en cámara A era la misma en cámara B. Cada aparición en cualquier cámara siempre producía `EVENT_NEW_PERSON` aunque el threshold ya estaba en 0.55.

**Causa raíz:** `AppearanceWorker._results` usaba solo `track_id` como clave (`Dict[int, np.ndarray]`), pero en DeepStream el tracker asigna track IDs **localmente por stream**. Dos cámaras pueden tener el mismo `track_id` simultáneamente. El embedding de la segunda cámara sobreescribía el de la primera en `_results[track_id]`, causando que `match_or_create()` recibiera el embedding incorrecto.

**Solución:**
- `appearance_worker.py`: cambiar clave a `(pad_index, track_id)` en `_results`, actualizar firmas de `enqueue()`, `get_result()`, agregar `clear_result()`.
- `probes.py` → `_handle_appearance_reid()`: pasar `pad_index` a `get_result()` y `enqueue()`.
- `probes.py` → `_expire_lost_tracks()`: llamar `_appearance_worker.clear_result(track_id, pad_index)` al expirar un track.
- `reid_manager.py`: `SIMILARITY_THRESHOLD` fijado en 0.60 con guía comentada (0.45 causa falsos positivos, 0.65 no matchea); cambiar reemplazo de embedding por EMA (alpha=0.7).
- `config.yaml` (demo): activar `pgie_interval: 2` (era -1=4) para reducir el delay de primer bbox.
- `probes.py` → `_needs_pixel`: condicional inteligente para ambos probes — solo copia GPU→CPU cuando hay tracks nuevos o con embedding pendiente; evita bloquear GStreamer en frames donde todos los tracks ya están asentados.

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

## 2026-05-13 — TRT scopedCudaResources crash a los ~9 min con 16 cámaras (NVMM agotado por tiler 1280×720)

**Contexto:** Pipeline de producción (`app.py`) con 16 cámaras sub-stream (960×544), después de quitar `nvdsosd`. El pipeline corría ~9 minutos antes de crashear con exit code 0 y reiniciar.

**Error en consola:**
```
[E] [TRT]  1: [scopedCudaResources.cpp::~ScopedCudaResources::...] Error Code 1: Cuda (an illegal memory access was encountered)
[E] [TRT]  1: [scopedCudaResources.cpp::~ScopedCudaResources::...] Error Code 1: Cuda (an illegal memory access was encountered)
... (decenas de líneas similares) ...
GStreamer ERROR: ... — pipeline forzado a salir
```

**Causa raíz:** Aunque se quitó `nvdsosd`, el tiler seguía corriendo a 1280×720 y `nvvidconv1` seguía allocando una superficie RGBA de 1280×720×4 ≈ 3.5 MB en NVMM por batch. Con 16 streams activos, el pool NVMM se agotaba gradualmente hasta que TRT no podía alocar workspace durante inferencia, causando el crash en `scopedCudaResources.cpp` (cleanup de recursos CUDA al fallar). El pipeline salía con código 0 → Docker reiniciaba → loop infinito.

**Solución:** Eliminar el servidor MJPEG completo (`_MjpegServer`, `appsink`, `nvvidconv2`, `caps_nv12`) y reducir el tiler a 640×360. El sink final pasa a ser `fakesink`. La superficie RGBA baja a 640×360×4 ≈ 0.9 MB (4× menos). Los crops siguen funcionando igual porque el probe sigue en el src-pad de `caps_rgba`. Archivos modificados: `app.py` (eliminar clase `_MjpegServer` y elementos de display), `CLAUDE.md` (actualizar diagrama del pipeline).

---

## 2026-05-17 — MjpegServer: cero bytes en el stream (single-threaded blocking)

**Contexto:** `deploy/pipelines/mjpeg_server.py`, `MjpegServer.run()`. La app Streamlit intentaba leer el stream MJPEG vía un thread Python (`_MjpegReader` con `requests`) y no recibía ningún frame.

**Error en consola:**
```
# Sin error explícito — el stream simplemente no entregaba frames.
# Los logs del servidor no mostraban conexiones entrantes adicionales.
```

**Causa raíz:** `HTTPServer` de Python es single-threaded. La primera conexión de `_MjpegReader` ocupaba el único slot de atención; ninguna otra conexión (ni el browser del usuario) podía conectarse. `_MjpegReader` recibía el primer boundary vacío porque el `_encode_loop` aún no había procesado frames, y luego se bloqueaba.

**Solución:** Convertir a `_ThreadingHTTPServer(ThreadingMixIn, HTTPServer)` con `daemon_threads = True`. Cada conexión de cliente corre en su propio thread. También se eliminó `_MjpegReader` completamente — el browser sirve el stream directamente vía `<img src="/stream/key">` en una página HTML que el propio MjpegServer sirve desde `/viewer/<key>`.

---

## 2026-05-17 — st.components.v1.html recrea el iframe en cada rerender (flickering)

**Contexto:** `deploy/qa_app/streamlit_app.py`, panel de video. El stream MJPEG parpadeaba cada 150-500 ms.

**Error en consola:**
```
DeprecationWarning: st.components.v1.html is deprecated and will be removed on 2026-06-01.
Use st.html instead.
```

**Causa raíz:** `st.components.v1.html` (y `st.html`) trata el bloque como contenido dinámico reemplazable en cada rerender. El autorefresh de 500 ms forzaba un rerender que destruía y recreaba el iframe, interrumpiendo la conexión MJPEG del browser.

**Solución:** Usar `st.iframe(viewer_url, height=560)` — Streamlit preserva el nodo iframe en React cuando el `src` no cambia entre rerenders. El autorefresh ya no interrumpe el stream. La URL del viewer es `http://<host>:<port>/viewer/<key>` y el HTML que sirve MjpegServer usa `<img src="/stream/<key>">` (mismo origen → sin CORS).

---

## 2026-05-17 — st.iframe() rechaza el argumento `scrolling`

**Contexto:** `deploy/qa_app/streamlit_app.py`, llamada a `st.iframe()`.

**Error en consola:**
```
TypeError: IframeMixin.iframe() got an unexpected keyword argument 'scrolling'
```

**Causa raíz:** `st.iframe()` (nuevo en Streamlit 1.44+) no acepta el parámetro `scrolling` que sí aceptaba `st.components.v1.iframe()`.

**Solución:** Eliminar el argumento `scrolling=False` de la llamada. El scroll en el iframe se controla desde el HTML interno (el viewer de MjpegServer no tiene scroll porque `<img>` ocupa el 100% del ancho con `display:block` y no hay overflow).

---

## 2026-05-17 — Detecciones y API calls no aparecen en Streamlit (ScriptRunContext ausente)

**Contexto:** `deploy/qa_app/streamlit_app.py`, subscriber daemon de Redis. Los paneles de detecciones y API calls siempre mostraban "Sin detecciones aún" / "Sin API calls aún" aunque el pipeline estaba publicando a Redis.

**Error en consola:**
```
Exception in thread qa-subscriber:
...
missing ScriptRunContext! This warning can be ignored when running in bare mode.
```

**Causa raíz:** El thread daemon del subscriber escribía en `st.session_state` desde fuera del ScriptRunContext de Streamlit. En Streamlit moderno (≥ 1.32), estas escrituras son silenciosamente descartadas cuando no hay ScriptRunContext activo — el warning aparece pero los datos nunca llegan a la UI.

**Solución:** Reemplazar `st.session_state.detections/apicalls` con deques a nivel de proceso usando `@st.cache_resource`. El subscriber daemon escribe en los deques directamente (sin necesitar ScriptRunContext); cada rerender los lee mediante `_bufs = _get_buffers()`. También se agregó auto-reconexión al subscriber (`while True: ... except Exception: time.sleep(2)`).

---

## 2026-05-17 — Ctrl+C no detiene qa.sh (doble trap por EXIT + comportamiento de `wait`)

**Contexto:** `deploy/qa.sh`. Al presionar Ctrl+C, el script no ejecutaba el cleanup o lo ejecutaba dos veces.

**Error en consola:**
```
# Primera iteración: trap en EXIT + INT → cleanup se ejecutaba dos veces
# Segunda iteración: `wait $LOGS_PID` bloqueaba el trap en algunas versiones de bash
```

**Causa raíz (primera iteración):** `trap _cleanup EXIT INT TERM` hacía que bash disparara el handler dos veces: una por INT y otra al salir vía `set -e` (el exit code no-cero de `docker compose logs` terminado por Ctrl+C disparaba EXIT).

**Causa raíz (segunda iteración):** `wait $LOGS_PID` puede bloquearse indefinidamente en algunas versiones de bash/Compose, impidiendo que el trap INT sea atendido antes de que el process group completo sea eliminado.

**Solución:** (1) Registrar solo `trap _cleanup INT TERM` — sin EXIT. (2) Correr `docker compose logs -f` en background y sustituir `wait` por un loop `while kill -0 "$LOGS_PID"; do sleep 1; done`. El comando `sleep` siempre sale inmediatamente ante SIGINT, lo que garantiza que el trap sea disparado con la siguiente iteración del loop.

---

## 2026-05-20 — ReID no reconoce misma persona al reentrar o cambiar cámara

**Contexto:** `deploy/pipelines/probes.py` → `_handle_appearance_reid()` + `deploy/pipelines/reid_manager.py`.

**Síntoma:** El ReID funcionaba correctamente dentro de la misma sesión de tracking (mismo `track_id` activo), pero al salir una persona del frame y volver a entrar (nuevo `track_id`) o aparecer en otra cámara, siempre producía `EVENT_NEW_PERSON` en lugar de reconocer el `global_id` existente.

**Causa raíz:** `_handle_appearance_reid()` tenía un guard `if not state.appearance_sent:` que envolvía todo el bloque de procesamiento. Una vez que el primer embedding era generado y consumido, `appearance_sent = True` permanecía para siempre — nunca más se encolaban crops ni se consumían resultados del `AppearanceWorker`. El embedding almacenado en `ReIdManager` era del primer crop (posiblemente parcial, desde el borde del frame) y nunca se actualizaba. Cuando la persona reaparecía con un nuevo `track_id`, su nuevo embedding se comparaba contra ese embedding inicial estale; si la similitud era < 0.60 (o 0.55), se creaba un nuevo `global_id`.

**Solución:**
- `probes.py` → `_handle_appearance_reid()`: eliminar el guard exterior. Llamar `get_result()` y `clear_result()` incondicionalmente. Después del primer match (`appearance_sent=False`), seguir encolando crops: primero frame, luego cada 15 frames hasta tener resultado, luego cada 90 frames para refresh periódico. Cuando ya hay `global_id`, el nuevo vector se pasa a `_reid_manager.update_embedding()` (EMA) en lugar de `match_or_create()`.
- `reid_manager.py`: agregar método `update_embedding(global_id, embedding)` que aplica EMA (alpha=0.7) al embedding existente sin pasar por el matching.
- `reid_manager.py`: bajar `SIMILARITY_THRESHOLD` de 0.60 a 0.55 para mejorar recall cross-cámara.

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
