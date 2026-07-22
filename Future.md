# Future Improvements — NX Computing AI

Registro de mejoras técnicas identificadas durante el desarrollo. Cada entrada documenta una posible implementación futura con suficiente contexto para evaluarla e implementarla sin tener que reconstruir la conversación original.

Ver regla 11 de CLAUDE.md para el formato de entradas y el protocolo completo.

---

## `update.sh` debe reinstalar servicios systemd de host (ej. `nx-dvr-watchdog`) automáticamente

**Descripción:** Servicios de host como `nx-dvr-watchdog` solo se instalan en la sección 6d de `setup.sh`, que corre una única vez durante la instalación inicial. `update.sh` únicamente hace `git pull` + rebuild condicional del container — nunca reinstala ni actualiza los servicios systemd del host. Si se agrega una feature de este tipo después de que un Jetson ya fue instalado en campo, o si se corrige un bug en un script como `dvr_watchdog.sh` (ver `ErrorHistory.md` 2026-07-01), los Jetsons ya desplegados se quedan con el servicio desactualizado o sin instalar, sin que nadie lo note hasta que falla.

**Por qué sería mejor:** Hoy se descubrió que un Jetson en campo nunca tuvo el watchdog de auto-recuperación de IP del DVR instalado — justo cuando se necesitaba (DVR con IP dinámica, 11 cámaras caídas). Un `update.sh` normal en ese Jetson no lo habría solucionado, porque no toca servicios de host. El fix quedó commiteado en el repo (para Jetsons nuevos vía `setup.sh`), pero los Jetsons ya desplegados requieren reinstalación manual.

**Reemplazaría:**
- Archivo: `deploy/tools/update.sh`
- Sección: lógica de actualización (git pull + rebuild condicional)
- Descripción de lo que se agrega: detectar si `tools/dvr_watchdog.sh` cambió respecto al `/usr/local/bin/nx_dvr_watchdog.sh` instalado (diff o hash) y, de ser así, re-copiar con sustitución de `@@WORK_DIR@@` y `systemctl restart nx-dvr-watchdog`. Idealmente generalizar a cualquier script de `tools/` que se instale como servicio de host.

**Tech stack propuesto:** Bash puro (mismo patrón que la sección 6d de `setup.sh`), sin dependencias nuevas.

**Consideraciones:** Requiere `sudo` dentro de `update.sh` (ya lo requiere para otras cosas). Esfuerzo estimado: 30 min.

---

## SSIM para detección de cambio visual en reference frames

**Descripción:** Reemplazar la métrica actual de diferencia media normalizada en `_scene_changed()` por SSIM (Structural Similarity Index) de scikit-image o OpenCV. SSIM es más robusto ante ruido de sensor y pequeñas variaciones locales que no reflejan un cambio estructural real de la escena.

**Por qué sería mejor:** La métrica actual (diferencia de píxeles normalizada por iluminación media) puede disparar falsos positivos si hay objetos en movimiento en el borde del frame (sombras, reflejos). SSIM mide similitud estructural y es menos sensible a estas perturbaciones locales, reduciendo reenvíos innecesarios.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Función: `_scene_changed()` (la comparación de la miniatura 64×36 con `np.abs(a/mean_a - b/mean_b).mean()`)
- Descripción: calcular `ssim(small_a, prev_np)` y disparar cuando `1 - ssim < REFERENCE_FRAME_CHANGE_THRESHOLD`.

**Tech stack propuesto:**
- Librería: `scikit-image >= 0.21` (función `skimage.metrics.structural_similarity`) o `cv2.quality.QualitySSIM_compute` (OpenCV contrib)
- Integración: reemplazo directo de la comparación numpy en `_scene_changed()`, sin cambios en el resto del pipeline.

**Consideraciones:** scikit-image agrega ~80 MB a la imagen Docker. OpenCV contrib está disponible en la imagen base de DeepStream. Esfuerzo estimado: 1 h.

---

## Detección de Caídas (`fall_detection`) — removido del MVP, pendiente reintegración

**Descripción:** Detecta cuando una persona cae al suelo mediante estimación de pose. Aplica 3 reglas geométricas: ángulo del torso > 45°, bbox más ancho que alto, caderas al nivel de los tobillos. Emite alerta si ≥ 2/3 reglas se cumplen. Cooldown de 4 segundos por persona.

**Por qué sería mejor:** Feature de alto valor para el sector hogar (residencias, condominios). Fue removido del MVP para simplificar el pipeline inicial.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py` — agregar `_FallDetectionHandler` y wiring en `init_handlers()`
- Archivo: `deploy/pipelines/app.py` — agregar `"fall_detection": None` en `SGIE_CONFIGS`
- Archivo: `deploy/pipelines/config_loader.py` — agregar a `VALID_CAPABILITIES` y paquetes `hogar_*`
- Nuevo archivo: `deploy/pipelines/pose_worker.py` — worker thread MoveNet ONNX
- Archivo: `deploy/pipelines/probes.py` — `NxApiClient.post_fall_detected()`

**Tech stack propuesto:**
- Modelo: MoveNet SinglePose Lightning ONNX (192×192, 17 keypoints COCO) — MIT
- Runtime: onnxruntime-gpu (ya instalado para OSNet/ArcFace)
- Descarga: `tools/download_models.py --movenet` (URL pública GitHub)

**Consideraciones:** El modelo es ligero (~5 MB). Workers ONNX corren en CPU y no compiten con TensorRT. Fue implementado y funcionaba — ver ErrorHistory.md para problemas previos resueltos.

---

## EPP Detection (`epp_detection`) — pendiente modelo

**Descripción:** Detecta cumplimiento de equipos de protección personal (cascos, chalecos, guantes) en entornos industriales. Emite `epp_violation` con lista de items faltantes y presentes.

**Por qué sería mejor:** Feature de alto valor para sector industrial. Solo falta definir y entrenar/adaptar el modelo SGIE.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py` — implementar `_EppHandler.process()` (stub listo, retorna None)
- Archivo: `deploy/pipelines/app.py` — `"epp_detection": str(_MODELS_DIR / "epp/config_infer.txt")`
- Archivo: `deploy/pipelines/config_loader.py` — agregar a `VALID_CAPABILITIES` y paquetes `industrial_*`

**Tech stack propuesto:**
- Modelo: SGIE DeepStream custom (YOLOv8-nano o similar), exportado a ONNX → TRT FP16
- Clases: helmet/no_helmet, vest/no_vest, gloves/no_gloves

**Consideraciones:** La arquitectura del pipeline ya soporta SGIEs. El stub `_EppHandler` está listo para recibir la implementación real. El trabajo principal es entrenar/adaptar el modelo y compilar el config nvinfer.

---

## Detección de Fuego y Humo (`fire_smoke`) — pendiente modelo

**Descripción:** Clasificador a nivel de frame que detecta presencia de fuego o humo. Emite `fire_smoke_alert` con severidad crítica.

**Por qué sería mejor:** Feature de seguridad crítica para sectores industrial y hogar. Solo falta el modelo.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py` — implementar `_FireSmokeHandler.process()` (stub listo)
- Archivo: `deploy/pipelines/app.py` — `"fire_smoke": str(_MODELS_DIR / "fire_smoke/config_infer.txt")`
- Archivo: `deploy/pipelines/config_loader.py` — agregar a `VALID_CAPABILITIES` y paquetes relevantes

**Tech stack propuesto:**
- Modelo: clasificador frame-level (EfficientNet-Lite o MobileNetV3), exportado a ONNX → TRT FP16
- Alternativa: modelo pre-entrenado público (Kaggle fire detection dataset)

---

## Lectura de Placas Vehiculares (`license_plate`) — pendiente modelos

**Descripción:** Detecta vehículos y lee sus placas (two-stage: LPD + LPR). Emite `vehicle_detected` con la placa leída.

**Por qué sería mejor:** Feature diferenciador para sector industrial (control de acceso vehicular).

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py` — implementar `_LicensePlateHandler.process()` (stub listo)
- Archivo: `deploy/pipelines/app.py` — agregar SGIEs para LPD y LPR en `SGIE_CONFIGS`

**Tech stack propuesto:**
- Modelos: NVIDIA LPD + LPR de NGC (disponibles públicamente, ARM64)
- Dos SGIEs en cadena: LPD (detecta ROI de placa) → LPR (lee caracteres)

---

## Grabación automática de clips (`recording_enabled`) — removida del MVP

**Descripción:** `RecordingManager` grababa automáticamente clips MP4 cuando se detectaban personas. Guardaba `tiled.mp4` (640×360) y `<camera_id>.mp4` (full-res) con thumbnail, metadata y auto-limpieza cuando el total superaba 10 GB.

**Por qué sería mejor:** Útil para auditoría y revisión post-evento. Fue removido porque el caso de uso principal era la QA app (que también fue removida).

**Reemplazaría:**
- Nuevo archivo: `deploy/pipelines/recording_manager.py` — restaurar desde git (`git show main:deploy/pipelines/recording_manager.py`)
- Archivo: `deploy/pipelines/probes.py` — wiring de `notify_detection()` en el probe
- Archivo: `deploy/pipelines/app.py` — instanciar y arrancar `RecordingManager` si `recording_enabled=true`
- Archivo: `deploy/pipelines/config_loader.py` — restaurar campo `recording_enabled: bool = False`

**Consideraciones:** La implementación completa está en el branch `main` (antes del MVP). Restaurar es trivial con `git show`. El valor sin QA app es limitado a menos que se agregue otro mecanismo de acceso a los clips (API endpoint o sftp).

---

## ~~Galería de embeddings por global_id en ReIdManager (reemplazar EMA único)~~ ✅ IMPLEMENTADO (2026-05-20)

**Descripción:** En lugar de mantener un solo vector EMA por `global_id`, almacenar una galería de hasta K embeddings (propuesta: K=5) que representen distintos ángulos y poses de la persona. Al matchear, la similitud se calcula como `max(query @ emb_i for emb_i in gallery)` — si algún ángulo coincide, el match ocurre aunque el ángulo actual difiera del resto.

**Por qué sería mejor:** El EMA mezcla embeddings de diferentes poses en un único vector que puede no representar bien ninguna de ellas — un embedding de espaldas promediado con uno de frente queda en un punto del espacio que no corresponde a ninguna pose real. La galería captura la variedad de apariencias real de la persona, igual que hace `FaceRecognizer` con las fotos de enrolamiento. Esto mejora directamente el recall cross-cámara (la persona puede llegar por otro ángulo y aún matchear).

**Reemplazaría:**
- Archivo: `deploy/pipelines/reid_manager.py`
- Sección / función: clase `_Entry` (campo `embedding: np.ndarray`) + `match_or_create()` líneas ~93-132 + `update_embedding()` líneas ~134-150
- Descripción: `_Entry.embedding` pasa de un array `(512,)` a una lista de arrays `List[np.ndarray]`; `_find_best_match` pasa de un único dot product a `max` sobre la galería; `update_embedding` añade el nuevo vector a la galería solo si es suficientemente distinto a los existentes

**Lógica de adición a la galería:**
- Si la galería tiene < K embeddings: añadir siempre
- Si la galería está llena: añadir solo si `max(new @ emb_i) < 0.85` para todos los embeddings existentes (el nuevo vector es suficientemente distinto = ángulo nuevo)
- Si el nuevo vector es muy similar a uno existente (`sim > 0.85`): ignorar (duplicado del mismo ángulo)
- Esto garantiza que la galería cubre distintas poses sin almacenar duplicados

**Tech stack propuesto:**
- Solo numpy — sin dependencias nuevas
- Matching: `np.stack(gallery) @ query` → `max` — sigue siendo O(N×K) pero K≤5, negligible
- Persistencia: `_save()` guarda lista de embeddings por `global_id` en JSON (lista de listas)

**Consideraciones:** Cambio de esquema en `reid_db.json` — requiere migración o reset del archivo al desplegar. El matching sigue siendo vectorizable por numpy. Esfuerzo estimado: 2-3 horas. Relacionado con [[EMA adaptativo con pesos por calidad de crop en ReIdManager]] — si se implementa la galería, el EMA adaptativo pierde relevancia.

---

## ~~EMA adaptativo con pesos por calidad de crop en ReIdManager~~ ❌ DESCARTADO (2026-05-20 — reemplazado por galería)

**Descripción:** El embedding de referencia en `ReIdManager` se actualiza con EMA fija (alpha=0.7). Una mejora sería ponderar el update según la calidad del crop: crops grandes y bien iluminados deberían tener más peso que crops pequeños u ocluidos.

**Por qué sería mejor:** El EMA fijo mezcla embeddings buenos y malos por igual. Con pesos por calidad (basado en tamaño del bbox y confianza del PGIE), el embedding de referencia converge más rápido hacia representaciones estables y los matches cross-cámara mejorarían, especialmente en sub-streams con personas lejanas.

**Reemplazaría:**
- Archivo: `deploy/pipelines/reid_manager.py`
- Sección / función: `match_or_create()` líneas ~107-110
- Descripción: EMA fija `0.7 * old + 0.3 * new`

**Tech stack propuesto:**
- Solo numpy — sin deps nuevas
- Métricas de calidad: `bbox_area / frame_area` y `pgie_confidence` ya disponibles en el probe
- Requeriría pasar `quality_score: float` a `match_or_create()` y ajustar la firma

**Consideraciones:** Cambio de API en `reid_manager.match_or_create()` — hay 2 call sites en `probes.py`. Esfuerzo estimado: 1-2 horas.

---

## Resolución del tiler MJPEG configurable por cliente

**Descripción:** La resolución del preview MJPEG (nvmultistreamtiler) está hardcodeada a 1280×720. Sería útil exponerla en `config.yaml` como `tiler_width` / `tiler_height` para que instalaciones con menos cámaras puedan usar 1920×1080 y deployments con más restricciones de memoria puedan bajar más.

**Por qué sería mejor:** Flexibilidad sin cambiar código. Actualmente 1280×720 es un compromiso conservador; clientes con 4-6 cámaras podrían preferir preview en HD.

**Reemplazaría:**
- Archivo: `deploy/pipelines/app.py`
- Sección / función: construcción del tiler (líneas ~323-327)
- Descripción: valores hardcodeados `1280` y `720`

**Tech stack propuesto:**
- Leer `tiler_width` / `tiler_height` desde `config.yaml` vía `config_loader.py`, con defaults 1280/720

**Consideraciones:** Cambio menor. Cuidar que valores muy grandes no causen NVMM overflow en deployments de 16 cámaras (razón por la que se bajó de 1920×1080 a 1280×720).

---

## ~~App de QA Visual — Streamlit con pipeline DeepStream en modo testing~~ ✅ IMPLEMENTADO (2026-05-16)

**Descripción:** Herramienta de QA visual que corre el mismo pipeline de producción (`app.py` + `probes.py`) activado con `NX_MODE=testing`, exponiendo en una interfaz Streamlit:

1. **Video en vivo con overlays**: MJPEG stream con bounding boxes y labels por feature activa (persona detectada, edad/género encima del bbox, bbox de rostro reconocido con nombre, etc.). En producción estos overlays no se dibujan para ahorrar NVMM — en testing mode se activan explícitamente.

2. **Panel de metadatos en tiempo real**: log scrolleable de lo que el pipeline detecta frame a frame — track_id, clase, confianza, coordenadas, clasificación edad/género, identidad facial reconocida. Permite verificar que la inferencia produce los valores correctos sin interpretar logs del terminal.

3. **Preview de payloads al API**: muestra en tiempo real los JSON que `NxApiClient` está enviando al backend — `person_entry`, `person_exit`, `analytics_snapshot`, etc. Útil para verificar que el formato y los campos son correctos antes de conectar al backend real.

4. **Toggles por capacidad**: botones on/off para activar/desactivar features individualmente (`age_gender`, `face_recognition`, `fall_detection`, etc.) sin reiniciar el pipeline. Permite aislar qué modelo causa un problema o cuánto afecta el rendimiento cada feature.

5. **Fuente de video intercambiable**: botón para cambiar entre las cámaras RTSP del lugar y un archivo de video de prueba (MP4 local). Permite reproducir escenas controladas (personas caminando, caídas, EPP) para validar detecciones de forma reproducible.

**Por qué sería mejor:** Hoy para verificar que PeopleNet detecta bien hay que interpretar logs de texto. No hay forma visual rápida de confirmar que un bbox está bien posicionado, que el payload tiene los campos correctos, o que la clasificación de edad/género funciona. Esta herramienta elimina ese friction loop de desarrollo y QA.

**Diseño propuesto:**
- Usar el mismo `app.py` con variable de entorno `NX_MODE=testing` que active OSD (bounding boxes), exponga los metadatos a Streamlit, y permita fuente de video dinámica. Sin duplicar archivos — mantiene el código sincronizado con producción.
- `NX_MODE=testing` activa: OSD rendering, metadata stream vía Redis pub/sub o queue Python, y acepta `TEST_VIDEO_PATH` como fuente alternativa al RTSP.
- Streamlit se suscribe al mismo Redis del stack para leer metadatos y payloads en tiempo real.
- Corre como un Docker container adicional con `docker compose --profile testing up`, sin tocar el stack de producción.

**Reemplazaría:**
- Archivo: `deploy/pipelines/app_video_testing.py` (archivo actual de testing limitado, solo archivos MP4, sin UI)
- Descripción: reemplaza el testing manual por terminal con una UI interactiva completa

**Tech stack propuesto:**
- UI: Streamlit ≥1.32 (MIT)
- Video: MJPEG stream embebido en Streamlit (`st.image` con streaming) o iframe al servidor MJPEG existente
- Metadatos: Redis pub/sub (ya existe en el stack) — `probes.py` publica, Streamlit consume
- Archivo de prueba: `st.file_uploader` o selector de archivos en `test_videos/`

**Consideraciones:**
- Streamlit debe correr en la misma red Docker que deepstream y Redis
- El OSD activado en testing consume NVMM extra — no usar con 16 cámaras simultáneas en Orin Nano; limitar a 4-8 streams en modo testing
- Los payloads al API en testing deben usar un `API_BASE_URL` de staging, no producción — documentar en `.env.example`
- Esfuerzo estimado: 2-3 días de desarrollo

---

## Fuente MP4 dinámica en QA Visual App

**Descripción:** La QA Visual app actual solo funciona con fuentes RTSP (pipeline de producción). Sería útil poder seleccionar un archivo MP4 desde el sidebar de Streamlit para reproducirlo en el pipeline y observar las detecciones en condiciones controladas (escenas de caídas, EPP, etc.) sin necesidad de que las cámaras del cliente tengan actividad en ese momento.

**Por qué sería mejor:** Permite validar detecciones de forma reproducible — las mismas escenas producen exactamente las mismas detecciones, facilitando comparación antes/después de ajustar parámetros o modelos.

**Reemplazaría:**
- Archivo: `deploy/qa.sh` + `deploy/docker-compose.qa.yml`
- Descripción: actualmente el pipeline arranca siempre contra las fuentes RTSP del cliente. Habría que soportar `TEST_VIDEO_PATH` como fuente alternativa, lo que requeriría reiniciar el pipeline deepstream con `app_video_testing.py` en lugar de `app.py`.

**Tech stack propuesto:**
- Selector de archivos en Streamlit (`st.selectbox` sobre la carpeta `test_videos/`) o `st.file_uploader`
- Variable de entorno `TEST_VIDEO_PATH` ya soportada por `app_video_testing.py`
- El cambio de fuente requiere un `docker restart deepstream` con la nueva variable — se puede hacer desde Streamlit via `docker SDK` o simplemente documentando el comando

**Consideraciones:** Cambiar de RTSP a MP4 requiere reiniciar el container deepstream, lo que interrumpe el stream MJPEG ~10 segundos. Si se quiere hacer sin reinicio habría que soportar fuente dinámica dentro de GStreamer (más complejo). Esfuerzo estimado: 1 día.

---

## Detección de EPP (`epp_detection`)

**Descripción:** Detectar cumplimiento de equipos de protección personal (cascos, chalecos reflectivos, guantes) en entornos industriales. Emitir alerta cuando una persona entra a una zona sin el EPP requerido.

**Por qué sería mejor:** Actualmente no existe ningún modelo EPP en el pipeline. Es la capacidad industrial de mayor valor para fábricas y bodegas.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: stub `_EppHandler` (búscar `_EppHandler` en probes.py)
- Descripción: actualmente el stub no hace nada; el SGIE tampoco existe

**Tech stack propuesto:**
- Modelo: SGIE custom (ONNX → TRT FP16) entrenado sobre personas con/sin EPP. Alternativa: adaptar YOLOv8-nano exportado a ONNX.
- `gie-unique-id=4` (los IDs 1–3 ya están ocupados)
- Agregar entrada en `SGIE_CONFIGS` en `app.py` y activar en paquetes `industrial_*`

**Consideraciones:** Requiere dataset de entrenamiento con EPP industrial (cascos amarillo/blanco, chalecos naranja/amarillo). Tamaño esperado <50MB. Esfuerzo estimado: 3-5 días (dataset + entrenamiento + integración).

---

## Detección de Fuego y Humo (`fire_smoke`)

**Descripción:** Clasificador a nivel de frame que detecta la presencia de fuego o humo en la escena. Emite alerta inmediata al backend.

**Por qué sería mejor:** Actualmente el stub no hace nada. Es una capacidad de alto valor para sectores industrial y hogar.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: stub `_FireSmokeHandler`
- Descripción: actualmente vacío; el SGIE no existe

**Tech stack propuesto:**
- Modelo: clasificador de imagen ONNX → TRT FP16 (entrada 224×224, salida: [no_fire, smoke, fire])
- Frame-level (no requiere bbox de persona — opera sobre el frame completo del tiler)
- Alternativa: FireNet o modelo Kaggle Fire Detection (Apache 2.0)
- `gie-unique-id=5`

**Consideraciones:** Falsos positivos con luz solar directa o reflejos. Requiere ajustar umbral de confianza por instalación. Esfuerzo estimado: 2-3 días.

---

## Lectura de Placas Vehiculares (`license_plate`)

**Descripción:** Detectar vehículos y leer sus placas usando dos SGIEs en cadena: LPD (License Plate Detector) y LPR (License Plate Reader/OCR).

**Por qué sería mejor:** Actualmente el stub no hace nada. Capacidad de alto valor para accesos vehiculares en industria y condominios.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: stub `_LicensePlateHandler`
- Descripción: actualmente vacío; los SGIEs no existen

**Tech stack propuesto:**
- LPD: NVIDIA TAO LPD (ONNX → TRT FP16, `gie-unique-id=6`)
- LPR: NVIDIA TAO LPR (ONNX → TRT FP16, `gie-unique-id=7`) — OCR carácter a carácter
- Ambos disponibles en NVIDIA NGC con licencia NVIDIA Developer

**Consideraciones:** LPR requiere resolución mínima de placa ~80×20px — subcámaras a 960×544 pueden ser insuficientes para placas lejanas. Esfuerzo estimado: 3-4 días (descargar modelos TAO, integrar SGIEs, parsear output de caracteres).

---

## ~~Auto-redescubrimiento de IP del DVR cuando cambia por DHCP~~ ✅ IMPLEMENTADO (2026-05-26)

**Descripción:** El DVR usa DHCP y cambia de IP cada vez que se reinicia o el router renueva el lease. Actualmente el técnico debe correr `nmap -p 554 192.168.10.0/24 --open` manualmente, actualizar `/etc/nx_dvr_ip` y reiniciar deepstream. Esto ocurrió el 2026-05-13 y el 2026-05-19. La solución automática haría que el pipeline detecte el fallo de todas las fuentes RTSP y reintente con la IP actualizada sin intervención humana.

**Por qué sería mejor:** El pipeline hoy simplemente continúa sin fuentes activas — no hay video, no hay detecciones, y el problema pasa desapercibido hasta que alguien lo nota. Un mecanismo automático garantizaría uptime sin necesidad de monitoreo manual.

**Reemplazaría:**
- Archivo: `deploy/pipelines/app.py`
- Sección / función: `_on_bus_message()` — handler de errores RTSP del bus GStreamer
- Descripción: actualmente logguea WARNING y continúa sin hacer nada más cuando todas las fuentes RTSP fallan

**Soluciones posibles (de menor a mayor complejidad):**

1. **DHCP reservation en el router (solución de infraestructura — recomendada):** Asignar IP fija al DVR por MAC address en la configuración del router. Costo: 5 minutos de configuración una sola vez. No requiere cambios de código. Ver ErrorHistory.md 2026-05-13.

2. **Re-scan automático en `_on_bus_message` cuando fallan todas las fuentes:**
   - Llevar un contador de fuentes RTSP fallidas. Si el conteo llega a N (total de cámaras configuradas), lanzar un thread que corra `nmap -p 554 <subnet> --open` (subprocess), compare la IP encontrada con `/etc/nx_dvr_ip`, y si difiere: actualice el archivo y haga `pipeline.set_state(Gst.State.NULL)` + reconstruya las fuentes con la nueva IP.
   - Limitación: `nmap` tarda ~3-5 segundos. Durante ese tiempo el pipeline está sin fuentes.

3. **Watchdog en `setup.sh` / systemd:** Un script separado que corre cada 5 min y verifica `ping $DVR_IP`. Si no responde, corre el nmap, actualiza `/etc/nx_dvr_ip` y hace `docker restart deepstream`. Independiente del código Python.

**Tech stack propuesto:**
- `subprocess` + `nmap` (ya instalado en el Jetson por `setup.sh`)
- Alternativa más rápida: `python-nmap` (Apache 2.0) — wraps nmap con API Python
- La opción 3 (watchdog shell) no requiere dependencias nuevas

**Consideraciones:** La solución de infraestructura (opción 1) es la correcta a largo plazo y debe hacerse en cada instalación. Las opciones 2 y 3 son fallbacks para instalaciones donde no se tiene acceso al router. El nmap requiere que el Jetson esté en la misma subred que el DVR (siempre cumplido). Esfuerzo: opción 1 = 0 código; opción 2 = ~4 horas; opción 3 = ~2 horas.

---

## PAR (Pedestrian Attribute Recognition) para Age/Gender + Augmentación de ReID

**Descripción:** Reemplazar el SGIE ResNet-18 de age/gender (6 clases via DeepStream nvinfer) con un modelo PAR Python worker que produce 26 atributos PA-100K: gender, 3 grupos de edad, pose, accesorios, tipo y color de ropa. Al mismo tiempo, usar los atributos PAR como validador del ReID: cuando OSNet encuentra un match por similitud de apariencia, PAR verifica que gender y age_group sean compatibles antes de confirmar el match — reduciendo falsos positivos cross-cámara entre personas de distinto género o rango de edad.

**Por qué sería mejor:** El SGIE actual clasifica en 6 categorías fijas (female/male × young/adult/senior). PAR con PA-100K da 26 atributos discriminativos. Para el ReID, el OSNet puro puede confundir personas de apariencia visual similar (mismo color de ropa) — PAR agrega una capa semántica que es complementaria al embedding de apariencia.

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: `_AgeGenderHandler` (líneas aprox. 785–886) — eliminar lectura de `classifier_meta_list` del SGIE; reemplazar con `_par_worker.get_result(track_id, pad_index)`
- Archivo: `deploy/pipelines/app.py`
- Sección / función: `SGIE_CONFIGS["age_gender"]` (línea ~48) — pasar de path a config_infer.txt a `None` (Python worker, no SGIE)
- Archivo: `deploy/pipelines/reid_manager.py`
- Sección / función: `_Entry` dataclass y `match_or_create()` — agregar `par_vec: Optional[np.ndarray]` y filtro de compatibilidad PAR

**Archivos nuevos a crear:**
- `deploy/pipelines/par_worker.py` — Python thread worker, mismo patrón que `appearance_worker.py`; queue de crops BGR, ONNX Runtime GPU, output vector 26-dim sigmoid float32
- `deploy/tools/export_par_onnx.py` — script para ejecutar en máquina dev: carga checkpoint PA-100K, exporta via `torch.onnx.export()` a `par_resnet18_pa100k.onnx` (opset=11)
- `deploy/models/par/par_resnet18_pa100k.onnx` — modelo exportado (no en git, se descarga via download_models.py)

**Estrategia de ReID augmentada (Filtro post-matching):**
```
OSNet match(query, gallery) → best_match si sim >= 0.55
↓ Si ambas personas tienen PAR result disponible:
  gender_ok = |female_prob_query - female_prob_match| < 0.3
  age_ok    = argmax(age_probs_query) == argmax(age_probs_match)
  Si NOT (gender_ok AND age_ok) → rechazar match → NEW_PERSON
```
El threshold OSNet 0.55 no cambia. El filtro PAR se puede desactivar con `use_par_reid_filter: false` en config.yaml.

**Mapeo de atributos PA-100K → age/gender actual:**
| Índice | Atributo | Uso |
|--------|----------|-----|
| 0 | female | gender display + ReID filter |
| 1 | age < 18 | → "Joven" + ReID filter |
| 2 | 18 ≤ age < 60 | → "Adulto/a" + ReID filter |
| 3 | age ≥ 60 | → "Mayor" + ReID filter |
| 4–6 | pose (front/side/back) | info solamente |
| 7–25 | accesorios + ropa | guardados en `_TrackState.par_vec`, reservados para futuro |

**Tech stack propuesto:**
- Modelo: Strong Baseline ResNet-18 (aajinjin/Strong_Baseline_of_Pedestrian_Attribute_Recognition) — MIT-compatible, puro PyTorch, sin extensiones C++
- Dataset fine-tuning: PA-100K (26 atributos, disponible públicamente)
- Input: 256×192 RGB, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- Output: sigmoid por atributo, rango [0,1]
- **Por qué NO OpenPAR (PromptPAR):** CLIP ViT-Base tiene ~87-100M params y 20-40ms de latencia vs ~12M params y 3-8ms del ResNet-18. Para los mismos 26 atributos PA-100K, el costo no justifica la diferencia de accuracy.

**Integración preferida: SGIE (no Python worker)**
Lo más eficiente es integrarlo como SGIE de DeepStream (igual que el age/gender actual), no como Python worker. El ResNet-18 exporta a ONNX y DeepStream lo convierte a TensorRT automáticamente — batching de crops en GPU, sin overhead de colas Python, integrado en el grafo GStreamer.

El único requisito adicional es un **custom C++ parser** (`custom_sigmoid_parser.so`) que reemplace al `custom_softmax_parser.so` actual. La diferencia: en lugar de leer 6 scores softmax y devolver 1 clase ganadora, lee 26 scores sigmoid independientes y crea 26 `NvDsLabelInfo` entries en `NvDsClassifierMeta` — una por atributo. `_AgeGenderHandler` los leería todos de `classifier_meta_list`.

```
config_infer.txt del SGIE PAR:
  gie-unique-id=2          # mismo que el actual
  num-detected-classes=26  # 26 atributos PA-100K
  operate-on-gie-id=1      # sobre detecciones del PGIE (personas)
  operate-on-class-ids=0   # solo class=0 (person)
  custom-lib-path=libcustom_sigmoid_parser.so
  parse-classifier-func-name=CustomPARParseFunction
```

Como alternativa más simple para el experimento inicial: usar Python worker (ONNX Runtime GPU) para evitar el C++, y migrar a SGIE TensorRT una vez validado el modelo.

**Consideraciones:**
- **Pesos PA-100K:** el repo no los incluye — requiere entrenar en una máquina dev con GPU (~2-4h en RTX 3060+) o buscar checkpoint publicado en HuggingFace. El training es offline, el Jetson solo hace inferencia.
- **Frecuencia de llamadas:** PAR se llama 1 vez al inicio del track + cada 90 frames (igual que OSNet en modo refresh) — latencia por frame promedio < 0.5ms.
- **RAM impacto:** +~75MB al footprint actual (~550MB) — bien dentro de 8GB unificados del Orin Nano.
- **docker-entrypoint.sh:** la compilación de `custom_softmax_parser.so` ya no es necesaria si age_gender usa PAR worker en lugar de SGIE — hacerla condicional.
- **Rollback:** `use_par_reid_filter=false` en config.yaml desactiva el filtro PAR sin tocar el OSNet ReID. El branch `feat/par-reid` no afecta `main`.
- Esfuerzo estimado: 1 día de training/export en máquina dev + 2-3 días de integración en el pipeline.

---

## ~~Auto-recuperación cuando el DVR cambia de IP~~ ✅ IMPLEMENTADO (2026-06-02)

**Descripción:** Cuando todos los streams RTSP fallan dentro de los primeros 60 s de arranque, `app.py` lanza automáticamente nmap en la subred /24 del DVR actual buscando un host con puerto 554 abierto. Si encuentra una IP distinta, actualiza `/etc/nx_dvr_ip` y sale con código 0 para que el entrypoint reinicie `app.py` con la nueva dirección — sin intervención manual.

**Implementado en:** `deploy/pipelines/app.py` — función `_try_rediscover_dvr()` + contador `_failed_sources` en `_on_bus_message`. Exit code 0 (no 1) para que el entrypoint loop continúe sin matar el container. Ventana de startup: 60 s. Timeout de nmap: 90 s.

---

## [Bug] `person_channel_change` rechazado por el backend con HTTP 422

**Descripción:** El Jetson emite `type: "person_channel_change"` a `POST /api/events` cada vez que el ReID local confirma que la misma persona cambió de cámara. El backend usa un discriminated union de Pydantic (`JetsonEvent`) que no incluye este tipo — Pydantic lanza `ValidationError` y el backend retorna 422. El Jetson loguea el error pero no reintenta (el diseño actual solo reintenta en fallos de red). Todos los eventos de cambio de cámara se pierden silenciosamente.

**Por qué sería mejor:** Corregirlo permite que el backend reciba el resultado del ReID del Jetson directamente (además del que calcula él mismo con `appearance_vector`), elimina los 422 en los logs del Jetson, y habilita la trazabilidad cross-cámara en tiempo real sin esperar a que llegue `person_appearance`.

**Reemplazaría:**
- Archivo: `NX-Platform/Backend/app/schemas.py`
- Sección / función: `JetsonEvent` union (líneas 178–197) — agregar `PersonChannelChangeEvent` con campos `track_id: int`, `bbox: BBoxSchema`, `confidence: float`, `global_id: str`, `prev_camera_id: Optional[str]`, `is_entry_exit_camera: bool`
- Archivo: `NX-Platform/Backend/app/routes/events.py`
- Sección / función: dispatcher `if t == "person_entry" / elif t == ...` (líneas 129–178) — agregar rama `elif t == "person_channel_change"` que llame a `tracker.on_person_channel_change(...)` (método nuevo en `person_tracker.py`) para reusar el `global_person_id` ya existente en lugar de crear uno nuevo

**Tech stack propuesto:**
- Sin dependencias nuevas — solo Pydantic + SQLAlchemy ya presentes

**Verificado en código:**
- `probes.py:552` — `post_person_channel_change()` envía `type: "person_channel_change"`
- `schemas.py:178–197` — `JetsonEvent` union no contiene este tipo → Pydantic ValidationError → HTTP 422
- `events.py:128–178` — ningún `elif` maneja `"person_channel_change"`

**Consideraciones:** El backend ya tiene su propia lógica de ReID por `appearance_vector` en `PersonTracker.on_person_appearance()`. El `person_channel_change` del Jetson es complementario — llega más rápido (no espera OSNet) pero es menos robusto. Se puede implementar como: si llega `person_channel_change`, buscar en `active_persons` el `global_person_id` por el `global_id` del Jetson (guardado en `payload`) y moverlo a la nueva cámara. Esfuerzo estimado: 2–3 horas.

---

## ~~[Bug] Página de Asistencia siempre vacía — backend roto y frontend bloqueado con "Pronto"~~ ✅ IMPLEMENTADO (2026-06-07)

**Descripción:** La página de Asistencia (`/activity`) tiene dos problemas independientes que hacen que nunca muestre datos reales:

1. **Backend:** `attendance.py` filtra por `event_type == "employee_identified"` en cuatro lugares (líneas 89, 186, 275, 345). El Jetson **nunca** genera ese tipo de evento — emite `employee_seen` (comercio/industrial) y `known_person_seen` (hogar). La columna `Event.employee_id` (UUID FK hacia `employees`) solo se llena cuando `event.type == "employee_identified"` — para `employee_seen` siempre queda `NULL`. Resultado: todas las queries de asistencia retornan 0 filas y la página muestra a todos los empleados como "ausente".

2. **Frontend:** la ruta `/activity` está enrutada a `<ComingSoon title="Actividad de empleados" />` en `App.jsx:121` en lugar del componente `Attendance` ya existente. La ruta también aparece en `COMING_SOON_ROUTES` en `lib/plans.js:47`, lo que agrega el badge "Pronto" en el sidebar. La página real (`Attendance.jsx`) existe y tiene la estructura UI, pero el usuario nunca llega a verla.

**Por qué sería mejor:** Con ambas correcciones, la página de Asistencia funcionaría de punta a punta con los datos reales que genera el pipeline, sin ningún cambio en el Jetson.

**Reemplazaría:**

*Backend:*
- Archivo: `NX-Platform/Backend/app/routes/attendance.py`
- Sección / función: los cuatro `Event.event_type == "employee_identified"` (líneas 89, 186, 275, 345) — reemplazar por `Event.event_type.in_(["employee_seen", "known_person_seen"])`. Eliminar el filtro `Event.employee_id.is_not(None)` (ese campo es siempre NULL para estos eventos). Cambiar el SELECT para extraer el nombre del empleado desde `Event.payload["employee_id"]` (string con el nombre, no UUID) o, mejor, usar la tabla `EmployeeZoneInterval` que ya está correctamente poblada con name strings desde `events.py:173`
- Archivo: `NX-Platform/Backend/app/routes/events.py`
- Sección / función: línea 105 — `employee_id_col = ... if event.type == "employee_identified" else None` — para `employee_seen` el campo `employee_id` contiene el nombre del empleado (str), no un UUID FK. Si se quiere mantener el lookup por UUID, hay que hacer `SELECT id FROM employees WHERE name = event.employee_id AND tenant_id = jetson.tenant_id` en el handler de `employee_seen`

*Frontend:*
- Archivo: `NX-Platform/frontend/src/App.jsx`
- Sección / función: línea 121 — `<Route path="/activity" element={<PlanRoute path="/activity"><ComingSoon title="Actividad de empleados" /></PlanRoute>} />` — reemplazar `<ComingSoon ...>` por `<Attendance />` (el import ya existe en línea 11)
- Archivo: `NX-Platform/frontend/src/lib/plans.js`
- Sección / función: línea 47 — eliminar `'/activity'` del set `COMING_SOON_ROUTES` para quitar el badge "Pronto" del sidebar

**Tech stack propuesto:**
- Sin dependencias nuevas. La solución más simple para el backend: reescribir las queries de `attendance.py` sobre `EmployeeZoneInterval` (ya correctamente poblada) en lugar de sobre `Event` directamente. Para el frontend: cambio de dos líneas.

**Verificado en código:**
- `probes.py:608–616` — Jetson emite `"employee_seen"`, nunca `"employee_identified"`
- `attendance.py:89,186,275,345` — filtra `event_type == "employee_identified"` → 0 resultados
- `events.py:105` — `employee_id_col` es `None` para `employee_seen` → columna `Event.employee_id` siempre NULL
- `models.py:146` — `Event.employee_id` es `UUID | None`, FK a `employees.id` — incompatible con el string que llega en `employee_seen`
- `App.jsx:121` — ruta `/activity` apunta a `<ComingSoon>` en lugar de `<Attendance>`
- `plans.js:47` — `'/activity'` en `COMING_SOON_ROUTES` → badge "Pronto" en sidebar

**Consideraciones:** La corrección del backend más robusta es cambiar las queries de `attendance.py` para usar `EmployeeZoneInterval` (que usa `employee_id: str` con el nombre) y hacer el JOIN con `employees` por nombre. Alternativa más correcta a largo plazo: resolver el nombre a UUID en el handler de `employee_seen` en `events.py` y guardar el UUID en `Event.employee_id` — requiere una query extra por evento pero mantiene integridad referencial. El cambio de frontend es trivial (2 líneas) y debe hacerse junto al fix del backend, no antes — de lo contrario el usuario verá una página con datos vacíos sin explicación. Esfuerzo estimado: 3–5 horas (backend) + 15 min (frontend).

---

## [Bug] Campo `entry_type` de `person_entry` se descarta silenciosamente en el backend

**Descripción:** El Jetson envía `entry_type: "new" | "return"` en cada `person_entry` (probes.py:541) para indicar si la persona es un visitante nuevo o alguien que el ReID local ya reconoció como regresante. El schema `PersonEntryEvent` en `schemas.py` no tiene este campo y usa `extra="ignore"`, por lo que Pydantic lo descarta sin error. El backend no puede distinguir una visita nueva de una reentrada y trata todo como nuevo, potencialmente creando `PersonSession` duplicadas para la misma persona.

**Por qué sería mejor:** Agregar el campo permite que el `PersonTracker` del backend salte la creación de una nueva `PersonSession` cuando `entry_type == "return"` y en cambio busque la sesión existente para el mismo `global_id` — reduciendo conteos duplicados de visitantes.

**Reemplazaría:**
- Archivo: `NX-Platform/Backend/app/schemas.py`
- Sección / función: `PersonEntryEvent` (líneas 54–59) — agregar `entry_type: Literal["new", "return"] = "new"`
- Archivo: `NX-Platform/Backend/app/routes/events.py`
- Sección / función: handler `t == "person_entry"` (líneas 129–135) — pasar `entry_type=event.entry_type` a `tracker.on_person_entry()`
- Archivo: `NX-Platform/Backend/app/services/person_tracker.py`
- Sección / función: `on_person_entry()` — si `entry_type == "return"` y existe sesión abierta para ese `global_person_id`, reutilizarla en lugar de crear una nueva

**Tech stack propuesto:**
- Sin dependencias nuevas

**Verificado en código:**
- `probes.py:541` — `"entry_type": "return" if is_return else "new"` siempre presente en el payload
- `schemas.py:54–59` — `PersonEntryEvent` no tiene `entry_type` → campo ignorado

**Consideraciones:** Cambio de bajo riesgo — el campo ya existe en el payload del Jetson. Solo requiere agregar el campo al schema, pasarlo al tracker, y agregar la lógica de reutilización de sesión. Esfuerzo estimado: 1–2 horas.

---

## [Bug] Campo `global_id` del Jetson se descarta en todos los eventos de persona

**Descripción:** El Jetson incluye `global_id` (su UUID de ReID local, asignado por `ReIdManager`) en `person_entry`, `person_exit` y `person_channel_change` (probes.py:544, 575). Ninguno de los schemas del backend tiene este campo — `PersonEntryEvent` y `PersonExitEvent` usan `extra="ignore"`, por lo que el campo se descarta silenciosamente. El backend calcula su propio ReID por separado usando `appearance_vector`, sin poder correlacionarlo con el ID que ya calculó el Jetson localmente.

**Por qué sería mejor:** Exponer el `global_id` del Jetson al backend permitiría: (1) vincular `person_channel_change` con la persona existente sin esperar `person_appearance`; (2) correlacionar el ReID local del Jetson con el del backend para detectar inconsistencias; (3) usar el `global_id` como clave de deduplicación adicional cuando llegan eventos fuera de orden.

**Reemplazaría:**
- Archivo: `NX-Platform/Backend/app/schemas.py`
- Sección / función: `PersonEntryEvent` (líneas 54–59) — agregar `global_id: Optional[str] = None`; mismo campo en `PersonExitEvent` (líneas 68–71)
- Archivo: `NX-Platform/Backend/app/routes/events.py`
- Sección / función: handler `t == "person_entry"` (líneas 129–135) — pasar `jetson_global_id=event.global_id` al tracker para que lo guarde en `ActivePerson`

**Tech stack propuesto:**
- Sin dependencias nuevas

**Verificado en código:**
- `probes.py:543–544` — `if global_id: payload["global_id"] = global_id` en `post_person_entry`
- `probes.py:574–575` — mismo patrón en `post_person_exit`
- `schemas.py:54–71` — ninguno de los dos schemas tiene `global_id`

**Consideraciones:** Este bug está relacionado con el Bug de `person_channel_change` — ambos se resuelven más fácilmente juntos, ya que `PersonChannelChangeEvent` también necesita `global_id`. Esfuerzo estimado: 1 hora (solo agregar el campo; decidir qué hacer con él en el tracker es el trabajo real).

---

## ReID cross-Jetson vía Redis local compartido en el mismo sitio (en vez de depender solo del backend)

**Descripción:** Hoy el único mecanismo para reconciliar la identidad de una persona vista por dos Jetsons físicos distintos es el backend: `PersonTracker.find_match()` compara el `appearance_vector` crudo que llega en `person_appearance` contra `recent_exits` (`REID_THRESHOLD=0.65`), con alcance "mismo `tenant_id`" — sin ningún concepto de "sitio físico". La idea es que, cuando un mismo sitio tiene **más de un Jetson** (necesario cuando hay más cámaras de las que un solo Jetson soporta — ver Notas de Rendimiento: máx. 6 streams main / 16 sub — o múltiples DVRs no unificados), esos Jetsons compartan un **Redis local en su propia LAN** donde escriban/lean directamente embeddings y `global_id` resueltos, igual que hoy hace `ReIdManager` con `deploy/reid_db.json`, pero compartido entre procesos/dispositivos en vez de un archivo local a un solo proceso.

**Por qué sería mejor:**

1. **Alcance correctamente delimitado a "mismo sitio físico".** Verificado en código: el modelo `Jetson` en el backend (`NX-Platform/Backend/app/models.py:72-92`) solo tiene un campo `location: str | None` de texto libre — no existe ninguna agrupación estructurada de "estos Jetsons son del mismo sitio". Esto significa que si un tenant tiene dos tiendas separadas, cada una con su propio Jetson, el `find_match()` actual del backend podría (incorrectamente) fusionar la visita de la misma persona en la Tienda A por la mañana y la Tienda B por la tarde en un solo `global_person_id`/sesión — algo que probablemente no tiene sentido de negocio (ver Consideraciones). Un Redis compartido solo alcanzable por LAN local resuelve esto gratis: es físicamente imposible que un Jetson de otro sitio (otra red) le pegue al Redis de este sitio, así que el alcance de "misma persona" queda naturalmente limitado a "mismo sitio físico".

2. **Latencia y resiliencia offline.** Resolver el ReID cross-cámara localmente evita la ida y vuelta a la nube, y sigue funcionando si el sitio pierde internet — consistente con la filosofía del proyecto ("todo el procesamiento de video ocurre on-device... funcionamiento sin dependencia de internet para la inferencia", ver introducción de este CLAUDE.md).

3. **Simplifica el backend.** Si el ReID cross-cámara del caso multi-Jetson-mismo-sitio ya llega resuelto (mismo `global_id`) desde el Jetson, el comparador de embeddings del backend (`PersonTracker.find_match`/`recent_exits`, `NX-Platform/Backend/app/services/person_tracker.py:131-156`) deja de ser necesario para ese caso — quedaría solo como fallback para el caso genuinamente cross-sitio, si es que ese caso llega a tener sentido de negocio.

**Reemplazaría / tocaría:**
- Archivo: `deploy/pipelines/reid_manager.py` — hoy `ReIdManager` persiste su galería en `deploy/reid_db.json`, un archivo local al proceso/dispositivo. Habría que abstraer el storage detrás de una interfaz (archivo local vs Redis compartido), activable solo cuando el sitio tiene >1 Jetson.
- Archivo: `deploy/pipelines/config_loader.py` — no existe hoy ningún concepto de "sitio" que agrupe Jetsons; habría que agregar algo como `site_id` en `config.yaml` para que los Jetsons del mismo sitio sepan que deben compartir Redis.
- Archivo: `deploy/setup.sh` — el flujo de instalación asume un dispositivo autónomo; instalar un segundo Jetson en el mismo sitio necesitaría un flag nuevo ("¿es el primer Jetson del sitio, o se une a uno existente?") para configurar el host del Redis compartido.
- Archivo: `NX-Platform/Backend/app/services/person_tracker.py` — `find_match()`/`recent_exits` podrían simplificarse o re-scopearse una vez que el caso same-site quede resuelto localmente en el Jetson.

**Preguntas abiertas para la próxima sesión (sin resolver todavía):**
1. **¿Quién hostea el Redis compartido?** — ¿uno de los Jetsons del sitio actúa como "coordinador" y corre el Redis, o se agrega una cajita/contenedor dedicado en la LAN?
2. **¿Qué pasa si el nodo que hostea el Redis se cae o se reinicia?** — ¿los demás Jetsons degradan a su galería local (`reid_db.json`) mientras tanto, o pierden ReID cross-cámara temporalmente? Hay que decidir la política de fallback explícitamente, no dejarlo implícito.
3. **¿Cómo migra `ReIdManager` de gallery local a gallery compartida?** — hoy persiste en `deploy/reid_db.json` por proceso; habría que hacer condicional la lectura/escritura a Redis cuando el modo "sitio multi-Jetson" esté activo, sin romper ni complicar el modo single-Jetson (que debe seguir funcionando igual, sin Redis, para el caso común).
4. **¿Cómo se descubren los Jetsons entre sí en la LAN?** — ¿mDNS, IP fija documentada en el `config.yaml` del sitio, o reusar Tailscale (ya presente en el stack para VPN mesh de acceso remoto)?

**Consideraciones:**
- Este cambio es **independiente** de los bugs [[Campo `global_id` del Jetson se descarta en todos los eventos de persona]] y [[Campo `entry_type` de `person_entry` se descarta silenciosamente en el backend]] — esos son bugs del caso same-Jetson (un solo dispositivo) y deben arreglarse primero, por separado. Esta mejora es específicamente para el caso multi-Jetson-mismo-sitio.
- Vale la pena cuestionar si el matching cross-sitio (tiendas distintas del mismo tenant) tiene sentido de negocio en absoluto — fusionar la visita de una persona a la Tienda A con su visita a la Tienda B como "la misma sesión" probablemente NO es el comportamiento deseado para dwell-time/conteo de visitas por tienda. Si nunca se quiere ese comportamiento, el backend debería dejar de intentarlo (hoy lo hace sin querer, por falta de scoping), y el Redis local sería la única forma de ReID cross-cámara que tendría sentido real en el producto.
- Multi-Jetson-por-sitio no es el caso común hoy — un solo Jetson soporta hasta 16 sub-streams o 6 main streams desde un DVR. Sería relevante solo para sitios grandes (fábricas, centros comerciales) con más cámaras de las que un Jetson soporta, o con múltiples DVRs no unificados.
- Esfuerzo estimado: mayor que un fix puntual — toca `setup.sh`, `config_loader.py`, `reid_manager.py`, y potencialmente el backend. Requiere diseño explícito (resolver las 4 preguntas de arriba) antes de implementar — no es un cambio de una sola sesión.

---

## [Bug] `EmployeeIdentifiedEvent` es código muerto — nunca lo genera el Jetson

**Descripción:** El backend define `EmployeeIdentifiedEvent` con `type: "employee_identified"` y `employee_id: UUID` (FK a `employees`) en `schemas.py:167–175`. Este tipo está incluido en el `JetsonEvent` union y tiene lógica especial en `events.py:105`. El Jetson nunca genera este evento — fue probablemente el tipo original de reconocimiento facial antes de ser renombrado a `employee_seen`. El código muerto confunde porque `attendance.py` lo busca directamente (ver Bug de asistencia), y cualquier desarrollador que lea el schema asume que el Jetson lo envía.

**Por qué sería mejor:** Eliminar o marcar claramente el tipo muerto reduce la confusión, evita que bugs futuros dependan de él, y simplifica la lista de tipos del `JetsonEvent` union.

**Reemplazaría:**
- Archivo: `NX-Platform/Backend/app/schemas.py`
- Sección / función: `EmployeeIdentifiedEvent` (líneas 167–175) y su entrada en `JetsonEvent` union (línea 193) — eliminar la clase y la entrada del union. Antes de eliminar, confirmar que no hay ningún producer en el Jetson o en scripts de testing que lo use
- Archivo: `NX-Platform/Backend/app/routes/events.py`
- Sección / función: línea 105 — eliminar `employee_id_col = ... if event.type == "employee_identified" else None` ya que nunca entra por esa rama

**Tech stack propuesto:**
- Sin dependencias nuevas — solo eliminación de código

**Verificado en código:**
- `probes.py` — búsqueda completa de `"employee_identified"` retorna 0 resultados; el Jetson emite `"employee_seen"` (línea 608)
- `schemas.py:167–175` — `EmployeeIdentifiedEvent` con `employee_id: UUID` (UUID FK, no nombre string)
- `events.py:105` — rama especial que nunca se ejecuta
- `attendance.py:89` — usa este tipo como referencia, causando el Bug de asistencia

**Consideraciones:** Antes de eliminar, correr `grep -rn "employee_identified"` en toda la plataforma (incluidos tests, scripts de CI, y cualquier integración externa) para confirmar que no hay otro producer. Si existe un plan futuro de migrar la identificación de empleados a UUID (en lugar de nombre string), `EmployeeIdentifiedEvent` es el diseño correcto — en ese caso documentar el intent y marcar como `# pendiente conexión con el Jetson` en lugar de eliminar. Esfuerzo estimado: 30 minutos (verificación) + 1 hora (eliminación + ajuste de attendance).

---

## ~~Heatmap de recorrido por empleado~~ ✅ IMPLEMENTADO (2026-06-07)

**Descripción:** Vista de heatmap individual por empleado que muestra, sobre el reference frame de cada cámara, las zonas donde el empleado estuvo y cuánto tiempo pasó en ellas. Accesible desde la página de empleados al hacer click en un empleado específico.

**Por qué sería mejor:** Actualmente el dashboard solo muestra el tiempo total de permanencia por zona (`EmployeeZoneInterval.duration_seconds`) sin visualización espacial. Esta feature combinaría la información de tiempo por zona con las posiciones reales (WebSocket de posiciones) para mostrar un mapa de calor personalizado por empleado, permitiendo entender patrones de movimiento individuales: qué áreas recorre más, en qué zonas se detiene, comparativas entre turnos o empleados.

**Reemplazaría:**
- No reemplaza nada existente — es una página nueva en el frontend
- Se apoya en datos ya almacenados: `EmployeeZoneInterval` (tiempo por cámara), `reference_frames` (fondo de imagen por cámara), y posiciones del WebSocket (coordenadas normalizadas x_norm, y_norm)

**Tech stack propuesto:**
- Backend: nuevo endpoint `GET /api/employees/{employee_id}/heatmap?camera_id=&start=&end=` que agrega posiciones históricas filtradas por `employee_id` desde una nueva tabla `employee_position_logs`
- Frontend: reusar el componente canvas de `Heatmap.jsx` con el reference frame de la cámara como fondo, renderizando los puntos de calor del empleado seleccionado sobre la imagen real de la cámara
- Datos de posición: los snapshots del WebSocket (`/ws/positions`, cada 10s) actualmente son efímeros — habría que persistirlos en una nueva tabla `employee_position_logs (employee_id, camera_id, x_norm, y_norm, timestamp)` con retención configurable (ej. 30 días)

**Consideraciones:** El WebSocket de posiciones actualmente no se persiste en base de datos — es la principal barrera para esta feature. Hay dos opciones de diseño: (a) guardar todas las posiciones históricas en `employee_position_logs` cuando el track corresponde a un empleado identificado por face recognition — esto requiere cruzar el `track_id` del WebSocket con el de `employee_seen` en tiempo real dentro del backend; (b) calcular el heatmap solo en tiempo real para sesiones activas, sin historia. La opción (a) es más valiosa pero requiere más trabajo de infraestructura. La tabla de posiciones puede crecer rápido — considerar TimescaleDB hypertable con política de retención automática. Esfuerzo estimado: 3–5 días (tabla + persistencia en WS handler + endpoint + página frontend).

---

## ✅ Output verbose en modo stream (`NX_STREAM_ENABLED=true`) — implementado 2026-06-07

**Descripción:** Cuando el pipeline corre en modo stream (`./stream.sh`), imprimir por stdout — visible en `docker logs` — una línea legible por cada detección y por cada llamada a la API. Actualmente las llamadas exitosas solo se loguean en nivel DEBUG (invisible en producción) y las detecciones no generan ningún output de texto; solo se ven errores. En modo stream el operador quiere ver qué está pasando en tiempo real sin necesidad de un debugger.

**Por qué sería mejor:** Permite confirmar de un vistazo que el pipeline está detectando personas, asignando global IDs, clasificando demografía, reconociendo empleados y enviando eventos al backend — sin tener que consultar el dashboard ni esperar a que aparezcan datos.

**Reemplazaría / extendería:**

- Archivo: `deploy/pipelines/probes.py`
- Sección: inicio del archivo (después de `_IS_STREAM_ENABLED`) — agregar función helper:
  ```python
  _C = {  # ANSI colors, solo en stream mode
      "reset": "\033[0m", "bold": "\033[1m",
      "green": "\033[92m", "yellow": "\033[93m",
      "cyan": "\033[96m", "magenta": "\033[95m", "red": "\033[91m",
  }

  def _slog(*parts: str) -> None:
      """Print una línea de stream-log a stdout (visible en docker logs)."""
      if _IS_STREAM_ENABLED:
          print("".join(parts) + _C["reset"], flush=True)
  ```

- Archivo: `deploy/pipelines/probes.py`
- Sección / función: `_handle_appearance_reid()` línea ~1150, después de que `_reid_manager.match_or_create()` devuelve el resultado — imprimir detección con global ID:
  ```python
  # Línea existente: logger.info("ReID track=%d cam=%s → %s gid=%s prev=%s", ...)
  _slog(
      f"{_C['cyan']}[{camera_id}]{_C['reset']} ",
      f"{_C['bold']}DETECCIÓN{_C['reset']} ",
      f"track={p_track_id:<4} ",
      f"gid={_C['green']}{global_id[:8]}{_C['reset']}  ",
      f"tipo={_C['yellow']}{event_type}{_C['reset']}  ",
      f"conf={confidence:.2f}",
      f"  prev={prev_camera}" if prev_camera else "",
  )
  ```

- Archivo: `deploy/pipelines/probes.py`
- Sección / función: `_AgeGenderHandler.process()` línea ~738, cuando se consolida el `winner` con suficientes votos — imprimir clasificación demográfica:
  ```python
  _slog(
      f"{_C['cyan']}[{camera_id}]{_C['reset']} ",
      f"{_C['magenta']}DEMOGRAFÍA{_C['reset']} ",
      f"track={p_track_id:<4} ",
      f"{winner_gd} | {winner_ad}  conf={winner_prob:.0%}",
  )
  ```
  `camera_id` y `p_track_id` ya están en scope en ese método.

- Archivo: `deploy/pipelines/probes.py`
- Sección / función: `_FaceRecognitionHandler.process()` línea ~830, cuando se reconoce un empleado (`name != "Desconocido"`) — imprimir reconocimiento:
  ```python
  _slog(
      f"{_C['cyan']}[{camera_id}]{_C['reset']} ",
      f"{_C['green']}{_C['bold']}EMPLEADO{_C['reset']} ",
      f"track={parent_track_id:<4} ",
      f"nombre={_C['bold']}{name}{_C['reset']}  sim={conf:.2f}",
  )
  ```

- Archivo: `deploy/pipelines/probes.py`
- Sección / función: `NxApiClient._send()` líneas 383–396 — cambiar el `logger.debug` de éxito a `_slog` y mostrar el tipo de evento cuando está disponible. La firma de `_send()` actualmente no recibe el payload, pero `post_*` métodos sí lo tienen — la forma más simple es loguear en cada `post_*` individualmente (ya conocen el tipo de evento) en lugar de en `_send()`:
  ```python
  # En cada método post_* de NxApiClient (post_person_entry, post_person_exit,
  # post_employee_seen, post_reference_frame, post_analytics_snapshot, etc.)
  # agregar al inicio, antes de llamar a self._send():
  _slog(
      f"{_C['yellow']}[API]{_C['reset']} ",
      f"POST {endpoint}  ",
      f"tipo={_C['bold']}{event_type}{_C['reset']}  ",
      f"cam={camera_id}  track={track_id}",
  )
  ```
  Alternativamente, pasar el `event_type` como argumento opcional a `_send()` y loguearlo ahí junto con el status code de respuesta.

**Formato de output esperado en `docker logs`:**
```
[DEMOT-01-ch05] DETECCIÓN  track=42    gid=a3f9c1b2  tipo=new          conf=0.87
[DEMOT-01-ch05] DEMOGRAFÍA track=42    Masculino | Adulto  conf=84%
[DEMOT-01-ch05] EMPLEADO   track=42    nombre=Juan García  sim=0.91
[API] POST /api/events  tipo=person_entry    cam=DEMOT-01-ch05  track=42
[API] POST /api/events  tipo=person_classified  cam=DEMOT-01-ch05  track=42
[API] POST /api/events  tipo=employee_seen   cam=DEMOT-01-ch05  track=42
```

**Tech stack propuesto:**
- Sin dependencias nuevas — solo `print()` con códigos ANSI estándar (soportados en terminales Linux y en `docker logs`)
- Gateado en `_IS_STREAM_ENABLED` (línea 156) — cero overhead en producción normal

**Consideraciones:**
- `print(..., flush=True)` es necesario para que el output aparezca sin buffer en `docker logs -f`
- Los colores ANSI pueden quitarse si el operador prefiere logs limpios para grep — hacer los códigos configurables via `NO_COLOR=1` env var
- En instalaciones con 16 cámaras y alta actividad, el volumen de líneas puede ser alto — considerar un rate-limit por cámara (ej. max 1 línea/segundo por `camera_id`) para no saturar la terminal
- `_slog` debe ser una función de módulo (no método) para ser accesible desde handlers, probe callbacks y `NxApiClient` sin pasar referencias
- Esfuerzo estimado: 2–3 horas

---

## CHANGE TO OSNET1 — OSNet como SGIE de DeepStream (GPU nativo, sin Python TRT)

**Descripción:** Reemplazar el `AppearanceWorker` (Python thread con onnxruntime) por un SGIE de DeepStream que corre OSNet-x1.0 directamente en el pipeline TRT de GPU. DeepStream gestiona el contexto CUDA, la construcción del engine y el batching — igual que hace con PeopleNet y AgeGender. La embedding 512-dim se lee en el probe vía `NvDsInferTensorMeta`. El `ReIdManager` no cambia.

**Por qué es mejor que el approach actual (Python TRT):**
- Sin conflicto de contextos CUDA: DeepStream ya es dueño del GPU — el SGIE se integra al mismo contexto, sin kernel Cask errors.
- Sin `onnxruntime-gpu`, sin `pycuda`, sin gestión manual de memoria CUDA.
- Batching nativo: DeepStream agrupa crops de múltiples personas en un batch eficiente.
- Los embeddings llegan en el mismo frame (síncronos), no diferidos — simplifica la lógica de `person_entry` deferido.

**Validado contra:** https://github.com/ml6team/deepstream-python (repo de referencia con OSNet + DeepStream, DS 6.1, patrón idéntico)

---

### Paso 1 — Traer a `main` los cambios útiles de `UpgradedOSNETGPU`

Hacer cherry-pick de estos commits (o mergear solo estos archivos):
- `deploy/.gitignore` — excluye `models/osnet/` del repo
- `deploy/tools/download_models.py` — soporte de token GitHub (`--github-token`)
- `deploy/setup.sh` — sección de descarga de OSNet via `download_models.py`

**NO traer a main:**
- `deploy/tools/test_trt.py` — experimento de Python TRT, ya no necesario (eliminar del branch también)
- Los cambios de `appearance_worker.py` (GPU providers) — el archivo se elimina en el paso 3

---

### Paso 2 — Crear `deploy/models/osnet/config_infer_sgie_osnet.txt`

```ini
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/nx_tech/models/osnet/osnet_x1_0_market1501.onnx
model-engine-file=/nx_tech/models/osnet/osnet_x1_0_market1501.trt
force-implicit-batch-dim=0
batch-size=8
process-mode=2
network-mode=0
network-type=1
output-blob-names=output
output-tensor-meta=1
operate-on-gie-id=1
operate-on-class-ids=0
gie-unique-id=3
interval=0
input-object-min-width=32
input-object-min-height=32
```

Notas:
- `network-mode=0` = FP32. Probar `network-mode=2` (FP16) después — DeepStream lo maneja distinto a Python TRT, puede funcionar.
- `operate-on-class-ids=0` = personas en PeopleNet (no 2 como en el repo de referencia que usa YOLOv4).
- `gie-unique-id=3` (1=PeopleNet, 2=AgeGender, 3=OSNet).
- `output-blob-names=output` — nombre del tensor de salida del ONNX de OSNet.
- `model-engine-file` se genera automáticamente en el primer arranque de DeepStream (~2 min adicionales la primera vez, igual que PeopleNet).

---

### Paso 3 — Modificar `deploy/pipelines/app.py`

En `SGIE_CONFIGS`, agregar OSNet junto a AgeGender:

```python
SGIE_CONFIGS = {
    "age_gender": str(_MODELS_DIR / "resnet_age_gender_FB2" / "config_infer.txt"),
    "appearance":  str(_MODELS_DIR / "osnet" / "config_infer_sgie_osnet.txt"),
}
```

El key `"appearance"` no es una capacidad en `VALID_CAPABILITIES` — se activa si el ONNX existe (igual que hoy lo hace `AppearanceWorker`). Agregar la lógica condicional igual al patrón ya existente.

---

### Paso 4 — Modificar `deploy/pipelines/probes.py`

**Eliminar:**
- Import de `AppearanceWorker`
- Instanciación de `_appearance_worker` en `init_workers()`
- Llamadas a `_appearance_worker.submit()` y `_appearance_worker.get_result()`
- Lógica de embedding deferido (deadline 30 frames)

**Agregar** en el probe, dentro del loop de objetos (donde se tiene `obj_meta`):

```python
import ctypes  # ya importado

OSNET_GIE_ID = 3

def _extract_osnet_embedding(obj_meta) -> np.ndarray | None:
    """Lee el vector 512-dim del SGIE OSNet desde los metadatos del objeto."""
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if tensor_meta.unique_id == OSNET_GIE_ID:
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer),
                                  ctypes.POINTER(ctypes.c_float))
                emb = np.ctypeslib.as_array(ptr, shape=(512,)).copy()
                norm = np.linalg.norm(emb)
                return emb / norm if norm > 0 else emb
        try:
            l_user = l_user.next
        except StopIteration:
            break
    return None
```

El embedding se pasa directamente a `_reid_manager.match_or_create()` en el mismo frame — sin cola, sin deferido.

---

### Paso 5 — Eliminar `deploy/pipelines/appearance_worker.py`

Ya no se necesita. El SGIE reemplaza su función por completo.

---

### Paso 6 — `setup.sh` y descarga del ONNX

`setup.sh` ya extrae el token del remote de git y lo pasa a `download_models.py --reid --github-token`. El token se manda en el header de autorización — eso ya está implementado en `UpgradedOSNETGPU`.

**⚠️ Pendiente:** `download_models.py` usa la URL directa de GitHub Releases (`github.com/releases/download/...`) que devuelve **HTTP 404 en repos privados aunque el token sea válido**. Esa URL no funciona para repos privados — solo funciona la URL de la API de GitHub. Implementar el fix de la entrada **"Descarga automática de OSNet desde GitHub Releases privado"** de este mismo archivo antes de desplegar en un Jetson nuevo.

Una vez aplicado ese fix, el flujo completo es automático:
1. `setup.sh` extrae token → llama `download_models.py --reid --github-token $TOKEN`
2. `download_models.py` descarga el ONNX via API de GitHub → `models/osnet/osnet_x1_0_market1501.onnx`
3. Primer `docker compose up`: DeepStream construye el engine TRT (~2 min extra, igual que PeopleNet)
4. Arranques siguientes: carga `osnet_x1_0_market1501.trt` directamente, sin recompilar

---

### Paso 7 — Opcional: limpiar `Dockerfile.jetson`

Remover la línea que instala el wheel de `onnxruntime-gpu` de nschloe (el que no funcionaba en GPU). `onnxruntime` puede quedar en `requirements.txt` para uso de InsightFace en CPU si lo necesita, pero el wheel de nschloe ya no tiene utilidad.

---

### Archivos modificados en total

| Archivo | Acción |
|---|---|
| `deploy/models/osnet/config_infer_sgie_osnet.txt` | **Crear** |
| `deploy/pipelines/app.py` | Agregar `"appearance"` en `SGIE_CONFIGS` |
| `deploy/pipelines/probes.py` | Reemplazar AppearanceWorker por lectura de `NvDsInferTensorMeta` |
| `deploy/pipelines/appearance_worker.py` | **Eliminar** |
| `deploy/tools/test_trt.py` | **Eliminar** (era solo experimental) |
| `deploy/Dockerfile.jetson` | Remover wheel onnxruntime-gpu nschloe (opcional) |

**Sin cambios:** `setup.sh`, `download_models.py`, `reid_manager.py`, `docker-compose.yml`, `config_loader.py`

---

## Descarga automática de OSNet desde GitHub Releases privado

**Descripción:** `download_models.py --reid` actualmente falla con HTTP 404 porque la URL directa de GitHub (`releases/download/<tag>/<file>`) no funciona para repos privados aunque se pase token. La solución confirmada es usar la API de GitHub para obtener la URL real del asset y descargar desde ahí.

**Flujo confirmado que funciona (verificado con curl):**
1. `GET https://api.github.com/repos/AlejandroMova/NX-JETSON/releases/tags/models-v1` con `Authorization: token <token>` → obtiene `asset["url"]` = `https://api.github.com/repos/.../releases/assets/456165693`
2. `GET <asset_url>` con `Authorization: token <token>` y `Accept: application/octet-stream` → GitHub redirige a S3 pre-signed URL → descarga el archivo (~8.4 MB en 1s)

**El token** se extrae automáticamente del remote de git en `setup.sh`:
```bash
GITHUB_TOKEN=$(git -C "${WORK_DIR}" remote get-url origin | sed -n 's|https://\([^@]*\)@.*|\1|p')
python3 "${WORK_DIR}/tools/download_models.py" --reid --github-token "$GITHUB_TOKEN"
```

**Reemplazaría:**
- Archivo: `deploy/tools/download_models.py`
- Función: `_download()` (actualmente usa `urllib.request.urlretrieve` directo con la URL de GitHub que da 404)
- Cambio: agregar `import json`; nueva función `_find_github_asset_url(owner, repo, tag, filename, token)`; nueva función `_download_github_private(asset_api_url, dest, label, token)`; actualizar `download_osnet()` para usarlas

**Consideraciones:** El token debe tener scope `repo` (lectura de releases privados). Si el repo se hace público en el futuro, se puede volver a la URL directa y eliminar la dependencia del token. Esfuerzo estimado: 30 min.

---

## Fine-tuning de OSNet-x1.0 con datos propios del cliente

**Descripción:** Fine-tunear el checkpoint OSNet-x1.0 (Market-1501) con crops reales extraídos de las cámaras del cliente. El modelo genérico fue entrenado en datasets de benchmarks de investigación; las cámaras fijas de un comercio tienen condiciones mucho más acotadas (iluminación constante, ángulos fijos, ropa cotidiana), lo que hace que un modelo fine-tuneado en el dominio específico supere al genérico incluso con pocos datos.

**Por qué sería mejor:** OSNet-x1.0 pre-entrenado alcanza ~94% Rank-1 en Market-1501 (benchmark de investigación), pero en producción con cámaras de DVR y condiciones reales el accuracy efectivo puede ser menor. Con 500–1000 pares etiquetados del propio cliente se puede obtener un modelo que supere ese número en el dominio real, reduciendo falsas identidades cross-cámara.

**Reemplazaría:**
- Archivo: `deploy/models/osnet/osnet_x1_0_market1501.onnx`
- Descripción: el ONNX actual (pre-entrenado en Market-1501) se reemplaza por uno fine-tuneado en datos propios; la integración en `AppearanceWorker` no cambia.

**Tech stack propuesto:**
- Framework: FastReID (Meta AI Research, Apache 2.0) — zoo de modelos + pipeline de entrenamiento con triplet loss y batch hard mining
- Datos: crops extraídos del tracker DeepStream + etiquetado manual de identidades entre cámaras (~500–1000 pares es suficiente para fine-tuning)
- Export: `torch.onnx.export()` desde el checkpoint fine-tuneado, mismo formato que el actual (opset 11, input NCHW 3×256×128, output (batch, 512))
- Entrenamiento: en máquina dev con GPU (RTX 3060+), no en el Jetson

**Consideraciones:**
- El etiquetado de pares es el cuello de botella: necesita que una persona aparezca en al menos dos cámaras y que un operador confirme que es la misma. Herramienta sugerida: script de CLI que muestre crops lado a lado y pida confirmación (s/n).
- El fine-tuning no requiere datos de mil personas distintas — con 50–100 identidades únicas vistas en múltiples cámaras es suficiente para ajustar el espacio de embeddings al dominio.
- El ONNX exportado es drop-in: reemplaza `osnet_x1_0_market1501.onnx` sin ningún cambio de código.
- FastReID soporta exportar directamente desde su pipeline de entrenamiento con `--export-onnx`.
- Esfuerzo estimado: 1 día de recolección de datos + etiquetado + 2–4 horas de entrenamiento.

## Umbral mínimo del SGIE OSNet como fracción de resolución, no píxeles absolutos

**Descripción:** `input-object-min-width/height` en `config_infer_sgie_osnet.txt` (96×192 px) es un valor absoluto de píxeles del streammux. Pero la resolución del streammux cambia según `stream_type` en `config.yaml` — `sub` es 960×544, `main` es 1920×1080 (ver `RESOLUTION_BY_STREAM_TYPE` en `config_loader.py`). La misma persona física, a la misma distancia real de la cámara, ocupa el doble de píxeles en main-stream que en sub-stream. Un umbral fijo en píxeles es entonces ~2x más laxo en clientes con `stream_type: main` que en los de `sub`, sin que nadie lo haya decidido así.

**Por qué sería mejor:** El umbral dejaría de depender silenciosamente de qué `stream_type` eligió cada cliente — hoy dos instalaciones con la misma distancia/ángulo de cámara real pueden tener comportamiento de ReID distinto solo por ese flag.

**Reemplazaría:**
- Archivo: `deploy/models/osnet/config_infer_sgie_osnet.txt`
- Sección: `input-object-min-width=96` / `input-object-min-height=192` (valores fijos)
- Descripción de lo que se reemplaza: umbral absoluto en píxeles, igual para `main` y `sub`

**Tech stack propuesto:**
- No requiere librería nueva. `input-object-min-*` de nvinfer es siempre absoluto, así que esto implicaría generar el config del SGIE dinámicamente (como ya se hace con `_apply_pgie_overrides()` en `app.py` para el PGIE) calculando el valor en píxeles a partir de `cfg.stream_width/height` y un porcentaje configurable.

**Consideraciones:** Solo vale la pena si en la práctica hay clientes mezclando `main` y `sub` con la misma expectativa de tamaño mínimo de persona — evaluar primero si esto es un problema real antes de agregar la generación dinámica de config (mismo patrón de riesgo que cualquier config generada en `/tmp/`, ver ErrorHistory.md 2026-05-28). Esfuerzo estimado: 1-2h.

---

## Evitar recomputar el embedding de OSNet en cada frame vía `secondary-reinfer-interval`

**Descripción:** El SGIE de OSNet corre con `interval=0` ([config_infer_sgie_osnet.txt](deploy/models/osnet/config_infer_sgie_osnet.txt)) — infiere en GPU en cada frame para cada persona sobre el umbral de tamaño, aunque `_handle_appearance_reid` en `probes.py` solo use ese embedding una vez para resolver el `global_id` (línea ~1488) y luego solo cada 90 frames para refrescar la galería (línea ~1541). El resto de las inferencias se calculan y se descartan sin usarse.

DeepStream tiene un mecanismo nativo para esto: cuando un SGIE corre detrás de un tracker, puede cachear el resultado por ID de objeto y solo reinferir cuando el objeto se ve por primera vez, su bbox crece ≥20%, o pasan `secondary-reinfer-interval` frames desde la última inferencia — sin quedar claro en la documentación si este cacheo aplica igual a un SGIE `network-type=1` (tensor crudo, como OSNet) o solo a "classifiers" tradicionales.

**Por qué sería mejor:** Ahorraría cómputo GPU real (no solo descartar resultado en Python) si el cacheo nativo no está aplicando ya solo. Alinearía además el ritmo de refresco de la galería con el motor de inferencia en vez de un contador manual en Python.

**Reemplazaría:**
- Archivo: `deploy/models/osnet/config_infer_sgie_osnet.txt` — agregar `secondary-reinfer-interval=N`
- Posible simplificación en `deploy/pipelines/probes.py` — el chequeo manual `frame_num % 90 == 0` en `_handle_appearance_reid` podría volverse redundante si DeepStream ya throttlea la inferencia real

**Consideraciones:** No hay evidencia hoy de que esto sea un cuello de botella real (sin frames caídos ni degradación de FPS reportada) — antes de tocar el config, verificar empíricamente si el cacheo nativo ya aplica a `network-type=1` (loggear si el tensor de dos frames consecutivos del mismo track es bit-idéntico). Sin ese dato, cualquier valor de `secondary-reinfer-interval` sería una suposición. Esfuerzo estimado: 1h de verificación + 30 min de cambio si aplica.

---

## AdaFace como reemplazo de ArcFace para reconocimiento facial en baja calidad

**Descripción:** Reemplazar `w600k_r50.onnx` (cabeza de reconocimiento ArcFace de `buffalo_l`) por un modelo entrenado con AdaFace (CVPR 2022, "Quality Adaptive Margin for Face Recognition"). AdaFace ajusta el margen de la función de pérdida según un indicador de calidad de imagen estimado durante el entrenamiento — para imágenes de baja calidad prioriza muestras "fáciles" en vez de forzar difíciles, lo que lo hace más robusto a las condiciones típicas de CCTV (crops chicos, ángulo oblicuo, poca luz) que ArcFace.

**Por qué sería mejor:** En IJB-B e IJB-C (benchmarks de calidad mixta, los más parecidos a condiciones de CCTV real — a diferencia de LFW, que es fotos curadas de alta calidad), AdaFace reduce el error 11% y 9% respectivamente contra el segundo mejor método evaluado en el paper original. En datos de alta calidad rinde aproximadamente igual que ArcFace — la ganancia real está específicamente en el régimen de baja calidad, que es el problema diagnosticado con el cliente Mova (similitudes de 0.10-0.31 contra un threshold de 0.50, cara casi siempre en mal ángulo).

**Reemplazaría:**
- Archivo: `deploy/models/insightface/` (modelo `w600k_r50.onnx` dentro de `buffalo_l`)
- Sección / función: `face_recognizer.py::_load_model()` y el uso de `FaceAnalysis.get()` en `_process()` — la alineación (`norm_crop` con landmarks de `det_10g.onnx`) y la detección no cambian, solo el modelo final de embedding
- Descripción de lo que se reemplaza: la cabeza de reconocimiento ArcFace por una cabeza AdaFace, manteniendo el mismo pipeline de detección+alineación de `buffalo_l`

**Tech stack propuesto:**
- Modelo: AdaFace, repo oficial `mk-minchul/AdaFace` — licencia MIT (permisiva, confirmado en el LICENSE del repo)
- Input/output: 112×112×3 BGR (normalizado mean=0.5/std=0.5), embedding 512-dim — mismo shape que ArcFace, compatible con el resto del pipeline (`known_faces.json`, columna de embeddings en el backend) sin cambios de schema
- Forma de integración: exportar el checkpoint de PyTorch a ONNX (no hay un ONNX oficial pre-exportado disponible — hay que hacerlo manualmente) y cargarlo en el lugar de `w600k_r50.onnx`

**Consideraciones:**
- `FACE_SIMILARITY_THRESHOLD=0.50` deja de ser válido — es un modelo distinto con otra distribución de similitud, hay que recalibrar con datos reales del cliente.
- **Todos los embeddings existentes quedan obsoletos** — son vectores de ArcFace, no comparables con AdaFace. Requiere re-enrollment de todos los empleados de todos los clientes. El backend ya soporta `model_version` en `EmployeeEmbedding` para esto, así que no es un rediseño de schema, pero sí una migración operativa real.
- No hay ONNX pre-exportado listo — exportarlo y verificar que el preprocesamiento coincide exacto con lo que espera `norm_crop()` es trabajo real de ingeniería, no un cambio de config.
- Evaluar solo después de probar el filtro de pose por track y CLAHE (ver entradas relacionadas) — si esos dos ya mejoran lo suficiente, esta migración (la más cara de las opciones evaluadas) puede no ser necesaria. Esfuerzo estimado: alto (exportación + recalibración de threshold + re-enrollment), no cuantificado en horas por la incertidumbre del paso de exportación.

---

## ReID robusto a vistas parciales/oclusión — cómo lo resuelven otras empresas/investigación

**Descripción:** El 2026-07-08 se eliminó por completo la rama de "vista parcial" de `ReIdManager` (`PARTIAL_BODY_MAX_RATIO`/`PARTIAL_BODY_REID_THRESHOLD`, ver `CLAUDE.md` sección "Re-ID entre Cámaras") — bajaba el umbral de match a 0.64 para compensar que OSNet da similitudes más bajas cuando solo se ve torso/hombros, y calibrando con crops reales del cliente DEMOONE se encontró que esa puerta fusionaba personas distintas con la misma frecuencia que el 0.68 original. Se decidió simplificar: un solo `SIMILARITY_THRESHOLD` para todo, sin manejo especial de vistas parciales — el costo aceptado es que una persona vista solo parcialmente puede tardar más en confirmar su identidad (o nunca confirmarla si la cámara nunca la ve de cuerpo completo), en vez del riesgo de fusionar personas distintas. Esta entrada registra qué opciones más sofisticadas existen si ese costo (menos recall en cámaras con vistas parciales frecuentes) resulta ser un problema real en producción.

**Por qué sería mejor:** recuperar la capacidad de confirmar identidad en vistas parciales sin reintroducir el riesgo de falsos positivos que motivó eliminar la rama actual.

**Opciones investigadas (de más simple a más compleja):**

1. **Umbral parcial recalibrado con datos reales, en vez de 0.64 a ciegas** — la opción más barata: mantener una rama de vista parcial pero con `create=False`/`add_to_gallery=False` (nunca crea identidad ni contamina galería, solo confirma) y un umbral calibrado con crops reales del rango de similitud real observado, no un número heredado sin verificar. Es la que se propuso y el usuario decidió no hacer por ahora (prioriza simplicidad).

2. **Matching por partes basado en pose (occlusion-aware ReID)** — el enfoque que usan sistemas de investigación robustos a oclusión (ej. "Dual-Branch Occlusion-Aware Semantic Part-Features Extraction Network", MDPI 2025): usan un modelo de estimación de pose para identificar qué partes del cuerpo son visibles en cada crop, y comparan similitud solo sobre las partes visibles en AMBAS imágenes (query y galería), en vez de comparar el embedding completo cuando falta información. Mucho más preciso en teoría, pero requiere un modelo adicional de pose (nuevo SGIE o worker) — no es un cambio de threshold, es una arquitectura nueva.

3. **Fusión ponderada por calidad ("quality-aware pooling")** — en vez de un umbral fijo, ponderar cada frame por un score de calidad (qué tan completo/nítido es el crop) al promediar o comparar embeddings — usado en ReID de video multi-frame. Requeriría calcular y propagar un score de calidad por detección, que hoy no existe en el pipeline.

4. **EMA (exponential moving average) en vez de galería de snapshots** — el enfoque de StrongSORT/BoT-SORT (tracking multi-objeto): un solo vector por identidad que se actualiza con cada nueva lectura en vez de guardar una galería de ángulos distintos. Ya se descartó en una conversación anterior (ver notas de la sesión 2026-07-07) porque "olvida" ángulos de cámaras distintas — no aplica bien al caso de re-id cross-cámara con ángulos muy diferentes por cámara, que es justo el problema principal de este proyecto.

**Reemplazaría:**
- Archivo: `deploy/pipelines/reid_manager.py` (`match_or_create`, `_find_best_match`) y `deploy/pipelines/probes.py` (`_handle_appearance_reid`)
- Descripción de lo que se reemplaza: el umbral único `SIMILARITY_THRESHOLD` actual, sin distinción por calidad/completitud de la vista

**Tech stack propuesto (solo para opción 2, la más completa):**
- Modelo / librería: un modelo de pose estimation ligero compatible con TensorRT (ej. MoveNet, ya usado como referencia en otras partes del proyecto) para generar keypoints por detección
- Forma de integración: nuevo SGIE o worker Python que anote qué región del bbox es visible antes de comparar embeddings

**Consideraciones:** las opciones 2-4 son inversión real de ingeniería (modelo nuevo, o cambio de arquitectura de galería) — solo valdría la pena evaluarlas si, con el sistema simplificado actual, se observa en producción que las cámaras con ángulos que rara vez muestran cuerpo completo (ej. muy altas o muy cercanas) tienen `person_count` inflado por falta de re-id, no antes. La opción 1 es la más barata de las cuatro si se necesita una solución intermedia sin ingeniería nueva.

---

## ~~Incluir `global_id` (o un manifest) en el export de crops de `download_crops_zip()`~~ ✅ IMPLEMENTADO (2026-07-14)

**Descripción:** El export de crops para calibración (`/superadmin/dataset` → `admin_crops.py::download_crops_zip()`) nombra los archivos `track_<track_id>_frame_<frame_num>.jpg`, sin `global_id`. La tabla `Crop` en el backend sí tiene `global_id` por fila (`models.py`), pero se pierde al exportar. Esto obliga a cruzar el CSV `osnet_reid.csv` contra los crops de forma heurística (proximidad de tiempo + cámara) en vez de un join exacto — ver el playbook agregado en `CLAUDE.md` (sección "Re-ID entre Cámaras", ronda 2026-07-14).

**Por qué sería mejor:** Con `global_id` en el nombre (o un `manifest.csv` adicional dentro del zip con `track_id,global_id,frame_num,camera_id,timestamp`), una ronda de calibración futura podría agrupar crops por `global_id` directamente y comparar galerías completas entre sí, sin depender de que dos tracks distintos hayan aparecido a pocos segundos de diferencia en la misma cámara para poder verificarlos visualmente. También eliminaría el riesgo de cruce heurístico incorrecto por reuso de `track_id` (ver caso `track_id=57` descartado en la ronda 2026-07-14).

**Reemplazaría:**
- Archivo: `AV-Platform/Backend/app/routes/admin_crops.py`
- Función: `download_crops_zip()` (línea ~267, construcción de `filename`)
- Descripción de lo que se agrega: incluir `global_id` en el nombre de archivo o escribir un `manifest.csv` junto a las carpetas de cámara dentro del zip.

**Tech stack propuesto:** ninguno nuevo — es un cambio de formato en una ruta ya existente (Python/FastAPI, `zipfile` de stdlib).

**Consideraciones:** cambio pequeño y de bajo riesgo (no toca producción en el Jetson, solo la herramienta de export del admin). Esfuerzo estimado: 15-20 min.

---

## Señal adicional a similitud coseno para el rango ambiguo de OSNet (0.65-0.83)

**Descripción:** La ronda de calibración 2026-07-14 (ver `CLAUDE.md`) encontró que, incluso restringiendo la comparación a la misma cámara y un gap temporal corto (<60s), la similitud coseno de OSNet-x1.0 **no separa** pares confirmados como la misma persona de pares confirmados como personas distintas — ambos grupos caen en el mismo rango (~0.64-0.81), con solapamiento total. El único patrón que sí se sostuvo en la muestra fue el gap temporal (0-60s → 75% misma persona; ≥60s → 0% misma persona, 6/6 personas distintas), pero no es lo bastante confiable como para automatizar una decisión de match/no-match por sí solo.

**Por qué sería mejor:** Ahora mismo, `SIMILARITY_THRESHOLD=0.85` es la única señal para decidir si dos apariciones son la misma persona, y estamos limitados por el techo de lo que esa señal puede resolver de forma segura en este local (ropa de tono genérico, iluminación de tienda, cámaras de resolución media). Una señal adicional y barata podría reducir el rango ambiguo sin arriesgar más fusiones incorrectas.

**Reemplazaría:** nada todavía — es exploratorio. Candidato de integración: `reid_manager.py::_find_best_match()`, como una señal secundaria que solo se consulta cuando `best_sim` cae en la zona ambigua (ej. 0.65-0.83), no como reemplazo del umbral.

**Tech stack propuesto (a evaluar, no decidido):**
- Histograma de color de la ropa (HSV, unas pocas líneas con OpenCV) como corroboración barata — descarta candidatos con colores de prenda muy distintos incluso si el embedding OSNet los acerca.
- Continuidad de posición/trayectoria (bbox del track saliente vs. bbox del track entrante, ya disponible en `probes.py` vía `rect_params`) — un salto de posición físicamente imposible en el gap de tiempo descarta el match sin tocar apariencia.

**Consideraciones:** requiere una muestra etiquetada más grande (idealmente con ground truth real, no verificación visual manual) antes de comprometerse a una de las dos opciones — esto es una nota para no repetir el análisis desde cero, no una decisión tomada.

**Actualización 2026-07-16 (calibración ronda 3):** se implementó una versión de "señal adicional" distinta a las dos candidatas de arriba — tiempo+cámara en vez de color/posición. `SIMILARITY_THRESHOLD_QUICK_REMATCH=0.75` en `reid_manager.py` solo se activa si el mejor candidato fue visto por última vez en la misma cámara hace menos de `QUICK_REMATCH_WINDOW_S=45s`. Motivada por evidencia visual directa en el piloto de bodega (bata azul, polo con placa, sin uniforme de por medio, fragmentados en varias identidades en cuestión de minutos — ver CLAUDE.md). Color de ropa y continuidad de posición **siguen como candidatas** si esta mitigación no es suficiente — no se implementaron todavía.

---

## Submódulo de ReID propio de NvDCF (`tracker: nvdcf_reid`) — ✅ IMPLEMENTADO (2026-07-22), pendiente de validar en Mova

**Descripción:** DeepStream trae, además del filtro de correlación normal de NvDCF (`_perf`, el que corre hoy en producción), un submódulo de re-identificación por apariencia *dentro del propio tracker* (`TrajectoryManagement.enableReAssoc` + sección `ReID:`) — una red neuronal separada que corre por objeto/por frame, por cámara, para recuperar un track_id tras una oclusión o pérdida más larga de lo que un filtro de correlación puro puede seguir. Es un mecanismo distinto y complementario a nuestro propio OSNet + `ReIdManager` (que sí cruza cámaras); el de NvDCF solo ayuda a que el tracker no fragmente a la misma persona en varios `track_id` *dentro de una misma cámara*, reduciendo cuántas veces le toca a OSNet decidir desde cero.

**Por qué antes estaba bloqueado:** se intentó activar el 2026-07-16 en el cliente Mova vía `tracker: nvdcf_accuracy` (el perfil stock completo de NVIDIA) y **truena el pipeline** — su submódulo de ReID necesita su propio engine TRT/TAO que nunca se descargó en ningún Jetson de este proyecto (`"TAO model file does not exist"`, ver `ErrorHistory.md` 2026-07-16). Se revirtió a `nvdcf` de inmediato. La Opción B (`nvdcf_extended_shadow`, ventana de búsqueda más ancha sin modelo nuevo) quedó como mitigación de menor riesgo mientras tanto.

**Qué se hizo (2026-07-22):**
- Se leyó el YAML stock `config_tracker_NvDCF_accuracy.yml` directamente del Jetson real (DeepStream 7.1) para confirmar, sin adivinar, el formato exacto: modelo `.etlt` (TAO-encoded, key `nvidia_tao`), `inferDims: [3, 256, 128]`, `networkMode: 1` (fp16), `offsets`/`netScaleFactor` de preprocesamiento, y todo el bloque `TrajectoryManagement.enableReAssoc` (pesos y umbrales de Re-Assoc). Esa es exactamente la información que faltaba el 07-16.
- Modelo: `ReIdentificationNet` (NVIDIA TAO, `resnet50_market1501.etlt`, deployable_v1.0) — confirmado público en NGC, sin API key (`https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt`, ~92 MB).
- `download_models.py::download_tracker_reid()` + flag `--tracker-reid` — mismo patrón que `download_osnet()`.
- `setup.sh::_download_models()` descarga el modelo automáticamente solo si el cliente ya tiene `tracker: nvdcf_reid` en su `config.yaml` (opt-in, sin flag de CLI — mismo patrón manual que `nvdcf_extended_shadow`).
- `deploy/models/tracker/config_tracker_NvDCF_reid.yml` — parte de `config_tracker_NvDCF_extended_shadow.yml` (BaseConfig/DataAssociator/StateEstimator/VisualTracker ya calibrados, `maxShadowTrackingAge=100` intacto) y le agrega **solo** el bloque ReID/Re-Assoc, con los valores exactos del stock (sin recalibrar — son específicos del modelo, no ajustables a ojo). Deliberadamente NO se partió de `_accuracy.yml` completo para no cambiar el perfil de asociación frame-a-frame ya calibrado en producción al mismo tiempo que se agrega ReID.
- `TRACKER_CONFIGS["nvdcf_reid"]` en `config_loader.py`.

**Pendiente (no hecho todavía, ver `systemrefactor.md` Capa 1 / plan de validación):**
- Correr en Mova (cliente de prueba) con `tracker: nvdcf_reid`, confirmar arranque sin error y que el engine se cachea (nombre `model-engine-file` ya coincide con la convención `<archivo>_b100_gpu0_fp16.engine`, misma trampa que OSNet en `ErrorHistory.md` 2026-07-04, ya evitada).
- Medir costo de GPU real — es una red completa corriendo por objeto/por frame (`reidExtractionInterval: 8` en frames), encima de PeopleNet + OSNet + AgeGender ya corriendo. Ya se confirmó headroom de GPU amplio con `pgie_interval=0` (`systemrefactor.md` §10-ter), pero eso no mide el costo específico de esta red.
- Medir fragmentación (span mediano de track, % tracks de 1 crop, tasa de match) contra la línea base main-15fps de `systemrefactor.md` §10-bis.

**Consideraciones:** sigue siendo intra-cámara, no reemplaza a OSNet/`ReIdManager` ni cambia `SIMILARITY_THRESHOLD` — alcance deliberadamente aditivo. Si la medición no muestra mejora clara o la GPU se satura sin ganancia, revertir a `nvdcf_extended_shadow` y escalar directo a la Capa 3 (clustering nocturno).

---

<!-- Agregar entradas aquí siguiendo el formato:

## [Título de la mejora]

**Descripción:** qué es esta implementación futura y qué resuelve o mejora

**Por qué sería mejor:** ventaja concreta sobre la solución actual (precisión, velocidad, escalabilidad, etc.)

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: nombre de la función o clase (líneas aprox. XXX–XXX)
- Descripción de lo que se reemplaza

**Tech stack propuesto:**
- Modelo / librería: nombre + versión + licencia
- Forma de integración: SGIE / worker Python / reemplazo de config / etc.

**Consideraciones:** dependencias, tamaño del modelo, compatibilidad con Jetson Orin Nano, esfuerzo estimado

-->
