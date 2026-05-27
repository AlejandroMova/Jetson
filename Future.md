# Future Improvements — NX Computing AI

Registro de mejoras técnicas identificadas durante el desarrollo. Cada entrada documenta una posible implementación futura con suficiente contexto para evaluarla e implementarla sin tener que reconstruir la conversación original.

Ver regla 11 de CLAUDE.md para el formato de entradas y el protocolo completo.

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

## ~~Auto-recuperación cuando el DVR cambia de IP~~ ✅ IMPLEMENTADO (2026-05-26)

**Descripción:** Cuando todos los streams RTSP fallan al arrancar (o todos mueren dentro de los primeros 30 s), `app.py` lanza automáticamente un ping sweep del subnet configurado para encontrar la nueva IP del DVR. Si la encuentra, actualiza `/etc/nx_dvr_ip` y reinicia el pipeline con la nueva dirección — sin intervención manual.

**Por qué sería mejor:** En instalaciones de campo la IP del DVR puede cambiar por DHCP, cambio de router, o re-configuración del cliente. Hoy el pipeline queda corriendo vacío hasta que alguien lo detecta y actualiza `/etc/nx_dvr_ip` manualmente.

**Reemplazaría:**
- Archivo: `deploy/pipelines/app.py`
- Sección / función: bus handler `_on_bus_message` + nueva función `_try_rediscover_dvr()`
- Descripción: agregar lógica que cuente streams fallidos en los primeros 30 s; si todos fallan, llamar `identify_dvr.py` con el subnet derivado del IP actual del Jetson (via `ip route get 1`). Si se encuentra una nueva IP, escribir `/etc/nx_dvr_ip` y hacer `sys.exit(1)` para que el entrypoint reinicie el pipeline.

**Tech stack propuesto:**
- `identify_dvr.py` ya existente en `deploy/tools/`
- Alternativa más rápida: `subprocess.run(["nmap", "-p", "554", subnet, "--open"])` — solo busca el host sin probar patrones RTSP (asume que el patrón del DVR no cambia, solo la IP)

**Consideraciones:** `identify_dvr.py` tarda 2-5 min; la versión solo-nmap tarda ~15 s. Para runtime, preferir la versión rápida. Activar solo si el 100% de los streams fallan en los primeros 30 s (no activar por fallo de cámaras individuales). El subnet se puede derivar del `ip route` del Jetson sin configuración adicional.

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
