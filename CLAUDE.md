# NX Computing AI — CLAUDE.md

## Resumen del Proyecto

**NX Computing AI** convierte las cámaras CCTV existentes de un negocio en un sistema de inteligencia sobre el espacio físico. Es un producto **llave en mano**: se instala un dispositivo **NVIDIA Jetson Orin Nano** en las instalaciones del cliente, se conecta al DVR de las cámaras que ya tiene, y a partir de ese momento el cliente recibe analytics en tiempo real a través de un dashboard en la nube.

**¿Qué problema resuelve?** Los negocios tienen cámaras de seguridad instaladas pero no extraen ningún valor analítico de ellas. NX les agrega inteligencia sin cambiar su infraestructura: conteo de personas, demografía, detección de caídas, reconocimiento de empleados, cumplimiento de EPP, detección de fuego/humo, entre otros — dependiendo del paquete contratado.

**¿A quién se le vende?**
- **Comercio** (tiendas, centros comerciales): conteo de afluencia, edad/género, reconocimiento facial de empleados
- **Industrial** (fábricas, bodegas): cumplimiento de EPP, detección de fuego/humo, lectura de placas
- **Hogar** (residencias, condominios): detección de caídas, alertas de personas desconocidas

**Modelo de entrega:** Un técnico instalador ejecuta un solo script (`setup.sh`) en el Jetson en campo. El dispositivo se configura automáticamente, se conecta al DVR, descarga los modelos necesarios y empieza a enviar datos al backend NX. Todo el procesamiento de video ocurre **on-device** (sin subir video a la nube), lo que garantiza privacidad y funcionamiento sin dependencia de internet para la inferencia.

---

## Estructura del Repositorio

```
NX_tech/
├── deploy/                     # Código de producción (lo que va al Jetson)
│   ├── pipelines/              # Lógica del pipeline GStreamer + DeepStream
│   │   ├── app.py              # Pipeline principal (RTSP en vivo)
│   │   ├── app_video_testing.py  # Pipeline de testing con archivos MP4
│   │   ├── probes.py           # Probe GStreamer + handlers + API client + QA overlays
│   │   ├── mjpeg_server.py     # Servidor MJPEG HTTP (solo activo con NX_QA_ENABLED=true)
│   │   ├── config_loader.py    # Carga y merge de configuración
│   │   └── common/             # Utilidades (FPS, bus_call)
│   ├── qa_app/                 # Dashboard QA Visual (Streamlit)
│   │   ├── streamlit_app.py    # UI: video en vivo + detecciones + API calls + toggles
│   │   ├── Dockerfile.qa       # Imagen python:3.11-slim para el container qa_app
│   │   └── requirements.txt    # streamlit, redis, streamlit-autorefresh
│   ├── models/                 # Modelos TensorRT (binarios por dispositivo)
│   │   ├── peoplenet_vpruned_quantized_decrypted_v2.3.4/
│   │   ├── resnet_age_gender_FB2/
│   │   └── facedetect_ir/
│   ├── tools/                  # Scripts utilitarios
│   │   ├── download_models.py  # Descarga modelos públicos
│   │   ├── identify_dvr.py     # Auto-detección de marca/patrón DVR
│   │   ├── probe_cameras.py    # Detección de canales activos
│   │   ├── register_face.py    # Enrolamiento de rostros
│   │   └── update.sh           # Actualización inteligente (git pull + rebuild)
│   ├── clients/                # Config por cliente (config.yaml + .env)
│   ├── setup.sh                # Script de primera instalación en campo
│   ├── qa.sh                   # Arrancar/detener modo QA (./qa.sh / ./qa.sh stop)
│   ├── docker-compose.yml
│   ├── docker-compose.qa.yml   # Override QA: NX_QA_ENABLED=true + servicio qa_app
│   ├── Dockerfile.jetson
│   └── docker-entrypoint.sh
├── dev/                        # Código de desarrollo / experimentos
│   └── PLATFORM/NX-Platform/  # Backend FastAPI (sistema separado)
├── README.md                   # Documentación principal del proyecto
├── ErrorHistory.md             # Historial de errores resueltos
├── Future.md                   # Registro de mejoras técnicas futuras
└── CLAUDE.md                   # Este archivo
```

---

## Arquitectura del Pipeline

**Producción (sin tiler):**
```
DVR (RTSP) → rtspsrc → h264/h265parse → nvv4l2decoder
  → nvstreammux → nvinfer (PeopleNet PGIE, gie-id=1)
  → nvtracker → [SGIEs opcionales por paquete]
  → nvvideoconvert(RGBA) → capsfilter(RGBA)
  → [probe: analytics en full-res] → fakesink
```

**QA mode (`NX_QA_ENABLED=true`) — dual probe:**
```
DVR (RTSP) → rtspsrc → h264/h265parse → nvv4l2decoder
  → nvstreammux → nvinfer (PeopleNet PGIE, gie-id=1)
  → nvtracker → [SGIEs opcionales por paquete]
  → nvvideoconvert(RGBA) → capsfilter(RGBA)
  → [Probe A: analytics full-res por cámara]
  → nvmultistreamtiler(640×360)
  → [Probe B: overlays + MJPEG tileado] → fakesink
```

> **Sin tiler en producción.** El tiler fue eliminado del path de producción — era un remanente del display que reducía la resolución disponible para face recognition, fall detection y re-ID a ~12-20px por cara. Todos los analytics corren ahora sobre frames full-res (ej. 1920×1080). En QA mode el tiler persiste únicamente para la visualización MJPEG: **Probe A** corre antes del tiler (full-res, escribe `_track_labels`) y **Probe B** corre después (frame tileado, solo overlays + MJPEG).

**Workers async (Python threads, no bloquean el pipeline):**
- `AppearanceWorker` — OSNet-x0.25 ONNX, re-ID entre cámaras
- `PoseWorker` — MoveNet Lightning ONNX, detección de caídas
- `FaceRecognizer` — InsightFace buffalo_l, reconocimiento facial
- `NxApiClient` — cola async para REST API backend
- `WsPositionClient` — telemetría de posiciones en tiempo real

**Paquetes (definen qué capacidades se activan):**
| Sector | Paquetes | Capacidades |
|--------|----------|-------------|
| comercio | comercio_basico/avanzado/total/enterprise | conteo, edad/género, reconocimiento facial |
| industrial | industrial_basico/avanzado/total/enterprise | conteo, EPP, placas, fuego/humo |
| hogar | hogar_basico/avanzado/total | conteo, detección de caídas, fuego/humo |

La capacidad activa se lee desde `/etc/nx_pipeline` en el Jetson.

---

## Capacidades del Sistema

### Conteo de Personas (`people_counting`) — ✅ Activo
Detecta y trackea personas en cada cámara. Emite eventos `person_entry` y `person_exit` con tiempo de permanencia. Acumula conteos y envía snapshots de analytics cada 60 segundos al backend.
- **Modelo:** PeopleNet v2.3.4 (ResNet-34, INT8, NVIDIA NGC) — detecta 3 clases: person, bag, face
- **Tracker:** NvDCF (correlación, recomendado ≤6 streams) o IOU (ligero, hasta 16 streams)
- **Siempre activo** en todos los paquetes

### Re-ID entre Cámaras (`appearance`) — ✅ Activo
Identifica cuando la misma persona aparece en cámaras distintas usando embeddings de apariencia. El matching ocurre **localmente en el Jetson** gracias al `ReIdManager`. Emite tres variantes de evento según el contexto:
- `person_entry` (`entry_type: "new"`) — persona nunca vista antes
- `person_entry` (`entry_type: "return"`) — misma persona, reapareció tras > 5 min de ausencia
- `person_channel_change` — misma persona, cambió de cámara dentro de la ventana de presencia (≤5 min)

La emisión de `person_entry` se **difiere** hasta que el embedding esté listo (deadline 30 frames / ~1 s a 30fps). Si el embedding no llega, se emite con `global_id: null`.

- **Embedding:** OSNet-x0.25 ONNX — vectores 512-dim L2-normalizados. `AppearanceWorker` (Python thread). Clave interna: `(pad_index, track_id)` — los track IDs son locales por cámara en DeepStream, por lo que la clave debe incluir el índice del stream.
- **Matching:** max-similitud coseno ≥ 0.55 sobre **galería de hasta 5 embeddings** por persona. `ReIdManager` — O(N×K) con K≤5, vectorizable con numpy. Nuevos ángulos se añaden a la galería solo si son suficientemente distintos a los existentes (sim < 0.85); cuando la galería está llena se reemplaza el miembro menos informativo.
- **Same-camera re-detection:** si `channel_change` ocurre con `prev_camera == camera_id` (tracker pierde y re-detecta en la misma cámara), se demota a `person_return` para no emitir un evento de cambio de cámara espurio.
- **Persistencia:** `deploy/reid_db.json` — sobrevive reinicios; TTL 1 hora sin actividad
- **Ventana de presencia:** 5 min (configurable en `reid_manager.py` como `PRESENCE_WINDOW_S`)
- Se activa automáticamente si el modelo existe en `models/osnet/`

### Edad y Género (`age_gender`) — ✅ Activo
Clasifica a cada persona detectada en una de 6 categorías: female_young, female_adult, female_senior, male_young, male_adult, male_senior. Requiere al menos 10 muestras del SGIE antes de confirmar la clasificación (sistema de votación para reducir falsos positivos).
- **Modelo:** ResNet-18 Pedestrian Attributes FB2 (FP16, SGIE gie-id=2)
- **Umbral:** bbox mínimo 64×160px, confianza ≥ 0.3
- **Parser custom:** `custom_softmax_parser.so` compilado en el entrypoint del contenedor

### Detección de Caídas (`fall_detection`) — ✅ Activo
Detecta cuando una persona cae al suelo mediante estimación de pose. Aplica 3 reglas geométricas: ángulo del torso > 45°, bbox más ancho que alto, caderas al mismo nivel que los tobillos. Emite alerta si ≥ 2 de 3 reglas se cumplen. Cooldown de 4 segundos por persona para evitar alertas repetidas.
- **Modelo:** MoveNet SinglePose Lightning ONNX (192×192, 17 keypoints COCO)
- **Worker:** `PoseWorker` (Python thread)
- **Descarga:** automática vía `download_models.py` durante `setup.sh`

### Reconocimiento Facial (`face_recognition`) — ✅ Activo
Identifica personas conocidas (empleados, residentes) a partir de una base de datos de embeddings faciales. Usa dos modelos en cadena: uno SGIE para detectar rostros con alta precisión, y un worker Python para extraer el embedding y compararlo con la DB. Requiere 3 coincidencias antes de bloquear la identidad por persona.
- **Detección:** FaceDetect IR (TLT FP16, SGIE gie-id=3, opera sobre class=0 del PGIE)
- **Embedding:** InsightFace buffalo_l — ArcFace 512-dim, threshold similitud coseno ≥ 0.50
- **Worker:** `FaceRecognizer` (Python thread)
- **DB:** `known_faces.json` (nombre → lista de embeddings). Se genera con `register_face.py`
- **Semántica por sector:** comercio/industrial → `employee_seen/presence/exit`; hogar → `known_person_seen/unknown_person_alert`

### Detección de EPP (`epp_detection`) — 🔄 Pendiente
Detecta cumplimiento de equipos de protección personal (cascos, chalecos reflectivos) en entornos industriales.
- **Integración prevista:** SGIE custom, handler `_EppHandler` en `probes.py` (stub listo)
- **Pendiente:** definir y entrenar/adaptar modelo

### Detección de Fuego y Humo (`fire_smoke`) — 🔄 Pendiente
Clasificador a nivel de frame que detecta la presencia de fuego o humo en la escena.
- **Integración prevista:** SGIE frame-level, handler `_FireSmokeHandler` en `probes.py` (stub listo)
- **Pendiente:** definir modelo

### Lectura de Placas (`license_plate`) — 🔄 Pendiente
Detecta vehículos y lee sus placas (LPD + LPR en dos etapas).
- **Integración prevista:** dos SGIEs en cadena, handler `_LicensePlateHandler` en `probes.py` (stub listo)
- **Pendiente:** integrar modelos LPD y LPR

---

## QA Visual App

Herramienta de inspección remota para el equipo NX. Permite que cualquier miembro del equipo vea en tiempo real lo que está detectando un Jetson desplegado en casa de un cliente, sin necesidad de estar físicamente en el lugar y sin afectar el pipeline de producción.

**Propósito:** En producción el pipeline corre con `fakesink` — no hay ningún output visual. El equipo no puede ver qué están detectando las cámaras, qué payloads se están enviando al backend, ni si los modelos están funcionando correctamente para ese cliente. El QA Visual App resuelve eso: activa el pipeline con overlays visuales y expone toda la metadata en un dashboard accesible vía Tailscale.

**Activación:** Solo cuando el técnico ejecuta `./qa.sh` en el Jetson. Usa un docker-compose override (`docker-compose.qa.yml`) que setea `NX_QA_ENABLED=true` en el container deepstream y agrega el container `qa_app`. Al salir (Ctrl+C o `./qa.sh stop`), el pipeline de producción se restaura automáticamente. **Cero impacto cuando no está activo.**

**Lo que muestra el dashboard:**
- **Tab "En Vivo":** Video MJPEG con bboxes/labels, log de detecciones en tiempo real, log de API calls
- **Tab "Grabaciones":** Biblioteca de clips grabados automáticamente; preview en el dashboard; botón para correr inferencia sobre el clip (reinicia el pipeline en modo playback)
- Selector de cámara: vista tileada (todas) o cámara individual
- Toggles por capacidad: apagar/prender `age_gender`, `fall_detection`, `face_recognition` sin reiniciar el pipeline
- Log de detecciones en tiempo real: track_id, label, confianza, alert de caída
- Log de API calls: JSON expandible de cada POST que el Jetson envía al backend
- Panel de resoluciones: tabla de resolución por componente (fuente DVR, PGIE, SGIEs, workers, tiler)
- Panel de FPS: fps total y fps por cámara, actualizado cada 5 segundos desde Redis
- **Editor de config** (sidebar): edita todas las variables de `config.yaml` en tiempo real. Cambios de pipeline/stream/tracker/canales/PGIE/SGIE requieren reinicio; los demás (entrada/salida, externas, conteo) aplican en caliente. El botón **💾 Guardar** escribe todos los valores actuales a `clients/<cliente>/config.yaml` usando `ruamel.yaml` (preserva comentarios). Las ediciones se persisten en Redis bajo `nx:qa:config_overrides` entre reinicios del dashboard.
- **Indicador de estado** (sidebar): muestra si el pipeline está grabando (⏺), en modo playback (⏯), o sin grabación activa.

**Cómo funciona internamente:**
1. **Probe A** (`pre_tiler_analytics_probe`, en `caps_rgba src-pad`): corre sobre frames RGBA full-res por cámara. Ejecuta todos los analytics (face recognition, fall detection, age/gender, re-ID, track lifecycle, API events). Llama `_update_fps_stats()` por frame. Escribe `_track_labels[track_id]` (face_name, fall, age_gender) para que Probe B pueda leerlos. Si `RecordingManager.is_recording`: lee el frame full-res y lo pasa a `RecordingManager.push_camera_frame()`.
2. **Probe B** (`osd_sink_pad_buffer_probe` → `_qa_overlay_probe`, en `tiler src-pad`): corre sobre el frame tileado 640×360. Solo dibuja bboxes de personas + labels (leídos de `_track_labels`). Encola el frame tileado en `tiled_frame_queue` y crops por cámara en `camera_frame_queues`. Publica `nx:qa:detections` a Redis.
3. `MjpegServer` (thread daemon en el container deepstream) consume esas queues, encoda a JPEG en background, y sirve HTTP multipart/x-mixed-replace en `:8080` bajo `/stream/all` y `/stream/<camera_id>`. También sirve `/viewer/<key>` — una página HTML mínima con `<img src="/stream/<key>">` mismo origen. En su encode loop pasa el frame tileado a `RecordingManager.push_tiled_frame()` si está grabando.
4. `RecordingManager` (thread daemon en el container deepstream): recibe notificaciones directamente desde los probes (`notify_detection()`, sin Redis). Cuando detecta personas, inicia una grabación MP4 con OpenCV VideoWriter. Graba `tiled.mp4` (640×360, solo QA) y `<camera_id>.mp4` (full-res por cámara, QA y producción). Cierra el clip tras 10 s sin detecciones (cooldown) o 5 min máximo. Descarta clips < 5 s. Auto-limpia cuando el total supera 10 GB. Publica estado en `nx:qa:recording_active` y `nx:qa:recording_info`. Directorio: `/nx_tech/recordings/<YYYYMMDD_HHMMSS>/` con `thumbnail.jpg`, `metadata.json`, `tiled.mp4`, `<cam_id>.mp4`. Se activa si `cfg.recording_enabled=true` (producción) o siempre en QA mode.
5. `app.py` en QA mode: cada 5 s verifica `nx:qa:playback_video` en Redis (via GLib.timeout_add). Si está seteada, envía EOS al pipeline y sale con código 42.
6. `docker-entrypoint.sh` (QA mode): loop que detecta si `nx:qa:playback_video` está en Redis; si es así arranca `app_video_testing.py --input <path> --no-loop`; al terminar borra la key y vuelve a `app.py`. En producción (`NX_QA_ENABLED` no seteado): exec directo sin loop.
7. `qa_app` (container Streamlit en `:8501`) embebe el viewer con `st.iframe(viewer_url)`. Tab "Grabaciones": lista clips de `/nx_tech/recordings/` (volumen compartido), muestra thumbnail+metadata, botón preview con `st.video()`, botón "▶ Correr Inferencia" que escribe `nx:qa:playback_video` en Redis.

**Tech stack:**
| Componente | Tecnología |
|-----------|-----------|
| UI del dashboard | Streamlit ≥1.32 (MIT), `streamlit-autorefresh` |
| Video streaming | HTTP MJPEG multipart/x-mixed-replace, `HTTPServer` + `ThreadingMixIn` |
| Video overlay | OpenCV (CPU, sobre frame RGBA del probe) |
| Metadata en tiempo real | Redis pub/sub (efímero, sin persistencia) — canales `nx:qa:detections`, `nx:qa:apicalls` |
| Estado del pipeline | Redis key `nx:qa:status` (JSON con client, channels, capabilities, component_resolutions) |
| FPS del pipeline | Redis key `nx:qa:pipeline_stats` (JSON: fps_per_camera, fps_total, ts — actualizado cada 5s por Probe A) |
| Feature toggles | Redis hash `nx:qa:capabilities` (leído por el probe antes de cada handler) |
| Config editor (QA) | Redis key `nx:qa:config_overrides` (JSON con todas las variables de config.yaml editables en el dashboard; el botón Guardar las persiste al archivo con `ruamel.yaml`) |
| Grabación de clips | `RecordingManager` — MP4 via OpenCV VideoWriter; `nx:qa:recording_active` + `nx:qa:recording_info` en Redis |
| Modo playback | Redis key `nx:qa:playback_video` (path al video) — detectado por `app.py` cada 5 s + manejado por `docker-entrypoint.sh` |
| Acceso remoto | Tailscale — la IP se extrae de `st.context.headers["host"]` en Streamlit |
| Container QA app | `python:3.11-slim` ARM64, sin GPU |

**Puertos:**
- `:8080` — MjpegServer (deepstream container, expuesto solo con QA activo)
- `:8501` — Streamlit dashboard (qa_app container, solo existe en `docker-compose.qa.yml`)

---

## Stack Tecnológico

### Infraestructura de video
| Tecnología | Versión | Uso |
|-----------|---------|-----|
| **NVIDIA DeepStream SDK** | 7.1 | Framework de inferencia de video en tiempo real |
| **GStreamer** | 1.x | Bus de elementos de media; el pipeline es un grafo GStreamer |
| **NVIDIA TensorRT** | 8.x (incluido en DeepStream) | Motor de inferencia GPU optimizado (INT8/FP16) |
| **nvv4l2decoder** | — | Decodificación de H.264/H.265 en hardware (NVDEC) |
| **nvstreammux** | — | Multiplexor de streams en batch para inferencia |
| **nvinfer** | — | Plugin GStreamer que ejecuta engines TensorRT |
| **nvtracker** | — | Plugin de tracking multi-objeto (NvDCF o IOU) |
| **pyds** | 1.1.11 | Bindings Python para la API de metadatos de DeepStream |

### Modelos de inferencia
| Modelo | Framework | Tarea | Activación |
|--------|-----------|-------|------------|
| **PeopleNet v2.3.4** | ONNX → TRT INT8 | Detección de personas, bolsas, rostros (PGIE) | Siempre activo |
| **ResNet-18 Pedestrian Attributes** | ONNX → TRT FP16 | Clasificación edad/género (SGIE) | `age_gender` |
| **FaceDetect IR** | TLT ETLT → TRT FP16 | Detección de rostros de alta precisión (SGIE) | `face_recognition` |
| **InsightFace buffalo_l (ArcFace)** | ONNX (CPU/GPU) | Embeddings faciales 512-dim para re-ID | `face_recognition` |
| **MoveNet SinglePose Lightning** | ONNX | Estimación de pose 17 keypoints, detección de caídas | `fall_detection` |
| **OSNet-x0.25** | ONNX | Appearance vectors 512-dim para re-ID entre cámaras | Siempre activo (si existe) |

### Librerías Python
| Librería | Uso |
|----------|-----|
| **onnxruntime** (GPU/aarch64) | Inferencia ONNX para MoveNet, OSNet, ArcFace |
| **insightface ≥ 0.7.3** | Pipeline de reconocimiento facial (detección + embedding) |
| **opencv-python-headless** | Manipulación de imágenes, crops, resize |
| **numpy** | Operaciones vectoriales, normalización de embeddings |
| **requests** | Cliente HTTP para REST API del backend |
| **websocket-client** | Telemetría de posiciones en tiempo real |
| **pyyaml / ruamel.yaml** | Lectura y escritura de config.yaml |
| **python-dotenv** | Carga de credenciales desde .env |

### Infraestructura de despliegue
| Tecnología | Uso |
|-----------|-----|
| **Docker Compose** | Orquestación de servicios en el Jetson |
| **Dockerfile.jetson** | Imagen ARM64 basada en `nvcr.io/nvidia/deepstream:7.1-samples-multiarch` |
| **Tailscale** | VPN mesh para acceso remoto al Jetson desde cualquier red |
| **TimescaleDB** (PostgreSQL 16) | Base de datos de series de tiempo para eventos y analytics |
| **Redis 7** | Cache y pub/sub para estado en tiempo real |

---

## Instrucción de Proceso — Imprimir Checklist Antes de Implementar

**Antes de comenzar cualquier implementación, Claude debe escribir en el chat un checklist con los pasos requeridos según las reglas aplicables a ese cambio específico.** El formato debe ser claro y breve, por ejemplo:

```
Checklist para este cambio:
- [ ] Regla 1: ¿Requiere confirmación del usuario? → [sí/no, por qué]
- [ ] Regla 2: Revisar README.md → [qué secciones aplican]
- [ ] Regla 3: Revisar setup.sh → [qué necesita cambiar o no]
- [ ] Regla 4: ¿Cómo encaja en la arquitectura modular? → [handler/worker/SGIE + paquetes afectados]
- [ ] Regla 5: ¿Hay links de descarga nuevos? → [sí/no]
- [ ] Regla 6: ¿Impacta el flujo de instalación en campo? → [sí/no, cómo]
- [ ] Regla 7: ¿La tecnología es open source y on-edge? → [verificación]
- [ ] Regla 8: ¿Hay conflictos con otras partes del proyecto? → [GPU, GIE IDs, config, etc.]
- [ ] Regla 9: Actualizar CLAUDE.md → [qué sección, obligatorio]
- [ ] Regla 2 (post): Actualizar README.md → [qué sección, obligatorio si aplica]
- [ ] Regla 10: ¿Hay errores que registrar en ErrorHistory.md? → [sí/no]
- [ ] Regla 11: ¿Hay mejoras futuras que registrar en Future.md? → [sí/no]
- [ ] Regla 12: ¿Cambia algún payload, endpoint o evento de API? → [sí/no — actualizar APIBackend.md]
- [ ] Regla 14: ¿El código nuevo/modificado tiene docstrings + comentarios en bloques y líneas importantes? → [verificar antes de dar la tarea por terminada]
```

**Las reglas 9 y 2 (post) son obligatorias en todo cambio que modifique comportamiento, constantes, flujos o archivos** — no dependen de juicio del agente. Si el cambio fue pequeño y ninguna descripción en CLAUDE.md ni README.md quedó desactualizada, indicarlo explícitamente ("sin cambios necesarios en documentación porque X").

No es necesario incluir reglas que claramente no aplican. El objetivo es que el usuario pueda ver el plan de trabajo antes de que se ejecute.

---

## Reglas de Trabajo

### 1. Preguntar antes de cambios arquitectónicos o eliminaciones

Antes de:
- Cambiar la estructura de directorios de `deploy/`
- Modificar el flujo del pipeline GStreamer en `app.py`
- Eliminar o refactorizar handlers en `probes.py`
- Cambiar el esquema de configuración en `config_loader.py`
- Modificar `docker-compose.yml`, `Dockerfile.jetson` o `docker-entrypoint.sh`
- Borrar modelos, configs de nvinfer, o archivos de `tools/`

**→ Detenerse y confirmar con el usuario antes de proceder.**

### 2. Revisar README.md antes de cambios Y actualizarlo al terminar

`README.md` es la fuente de verdad del proyecto:
- Define los paquetes y sus capacidades
- Documenta los patrones RTSP por marca de DVR
- Describe el flujo de instalación y actualización
- Explica el esquema de config y variables de entorno

**Antes de implementar:** leer las secciones relevantes para no contradecir lo ya documentado.

**Al terminar cualquier implementación:** revisar si el cambio afecta algo en README.md y, si es así, actualizarlo en ese mismo momento — no al final de la conversación ni cuando el usuario lo pida. Esto incluye: comportamiento de componentes, flujos de datos, eventos emitidos, constantes o umbrales configurables, y diagramas de arquitectura.

### 3. Siempre revisar setup.sh cuando se agrega algo al proyecto

`deploy/setup.sh` es la UX de instalación. Al agregar cualquier cosa nueva:
- ¿Necesita `setup.sh` descargar un modelo nuevo? → agregar a la sección de descargas
- ¿Hay una nueva variable de entorno? → agregar al `.env.example` y documentar en `setup.sh`
- ¿Cambia el Dockerfile o docker-compose? → verificar compatibilidad con el flujo de build en `setup.sh`
- ¿Hay un nuevo script de tool? → evaluar si debe invocarse desde `setup.sh`

### 4. Respetar la arquitectura modular y el sistema de paquetes

El proyecto usa un patrón de **capacidades por paquete**. Cada capacidad pertenece a ciertos sectores/paquetes según la necesidad del cliente — por ejemplo, `face_recognition` solo está en paquetes `comercio_*`, y `fall_detection` solo en paquetes `hogar_*`. Cualquier nueva tecnología debe integrarse siguiendo este patrón completo:

**Código del pipeline:**
- Nueva capacidad de inferencia → agregar como handler en `probes.py` siguiendo el patrón `_XxxHandler`
- Nuevo modelo SGIE → agregar entrada en `SGIE_CONFIGS` dict en `app.py`
- Worker Python (modelo no-DeepStream) → crear `xxx_worker.py` con patrón queue + thread, como `pose_worker.py`

**Sistema de capacidades y paquetes (`config_loader.py`):**
- Agregar la nueva capacidad a `KNOWN_CAPABILITIES`
- Determinar a qué paquetes pertenece: ¿es una feature de comercio, industrial, hogar, o varios? Revisar la tabla de paquetes en `README.md` para decidir en qué niveles (básico/avanzado/total/enterprise) tiene sentido incluirla
- Agregar la capacidad a los paquetes correspondientes en `PACKAGE_DEFINITIONS`
- Si aplica a un nuevo sector, crear los paquetes necesarios también en `PACKAGE_DEFINITIONS`

**Referencia de paquetes actuales:**
| Sector | Paquetes | Capacidades incluidas |
|--------|----------|-----------------------|
| comercio | basico/avanzado/total/enterprise | people_counting, age_gender, face_recognition (desde avanzado) |
| industrial | basico/avanzado/total/enterprise | people_counting, epp_detection, license_plate, fire_smoke |
| hogar | basico/avanzado/total | people_counting, fall_detection, fire_smoke |

No crear nuevos archivos `app_xxx.py` para casos especiales. Toda la lógica va en el `app.py` modular existente.

### 5. Verificar que los links de descarga funcionen

Antes de agregar o modificar cualquier URL de descarga (en `setup.sh`, `download_models.py`, `docker-entrypoint.sh` o `README.md`):
- Verificar que el link es accesible y descarga el archivo correcto
- Preferir URLs estables (releases de GitHub, registros oficiales de NGC/HuggingFace)
- Nunca usar links que requieran autenticación en `setup.sh` sin documentar cómo obtener las credenciales
- Documentar checksum o tamaño esperado cuando sea posible

### 6. Priorizar la experiencia de instalación en Jetson

El técnico instalador ejecuta `setup.sh` en campo, sin terminal interactiva avanzada ni conocimientos de Docker. Principios:
- `setup.sh` debe ser el único comando necesario (además de los flags documentados)
- Los errores deben ser claros en español y sugerir solución
- Las descargas de modelos deben hacerse automáticamente dentro del flujo de setup
- **Minimizar** pasos manuales post-setup: si algo se necesita siempre, lo ideal es que esté en `setup.sh`; si un paso manual es inevitable, debe estar documentado claramente en README.md con instrucciones paso a paso
- No agregar dependencias al host (solo Docker + Tailscale son dependencias del host)
- La duración del setup no es una restricción — puede tomar el tiempo que sea necesario; lo importante es que el proceso funcione de forma confiable y sin intervención inesperada

### 7. Usar tecnologías open source y compatibles con edge

Criterios para evaluar nuevas tecnologías:
- **Open source**: Licencia permisiva (MIT, Apache 2.0, BSD). Evitar licencias comerciales o restrictivas.
- **On-edge**: El modelo/librería debe poder correr en Jetson Orin Nano (ARM64, 8GB RAM, 1024 CUDA cores Ampere)
- **Sin cloud obligatorio**: No requerir APIs externas en el path crítico de inferencia
- **TensorRT-compatible**: Preferir modelos ONNX exportables → TensorRT engine (INT8 o FP16)
- **Tamaño razonable**: Modelos > 500MB requieren justificación explícita
- **Precedentes en el proyecto**: OSNet, MoveNet, PeopleNet, InsightFace buffalo_l son referencia

### 8. Verificar conflictos antes de implementar

Antes de implementar cualquier cambio, revisar:
- **GPU memory**: ¿El nuevo modelo cabe junto con PeopleNet + tracker + SGIEs activos? (Orin Nano tiene 8GB unificados)
- **NVDEC load**: ¿La resolución y cantidad de streams sigue dentro del límite documentado en `config_loader.py`?
- **GIE unique IDs**: Cada nvinfer necesita un `gie-unique-id` único (1=PeopleNet, 2=AgeGender, 3=FaceDetectIR)
- **Track ID namespace**: Los `track_id` son locales por cámara; el triplete `(jetson_id, camera_id, track_id)` es el key global
- **Queue sizes**: Los workers tienen queues con límite; agregar más workers reduce throughput disponible
- **Docker image size**: Agregar dependencias pesadas al `Dockerfile.jetson` aumenta tiempo de rebuild en campo
- **Conflictos de config**: Revisar `config_loader.py` para asegurarse que los nuevos parámetros no choquen con los existentes

### 9. Mantener este archivo actualizado — obligatorio al terminar cualquier implementación

**Esta actualización es parte de la tarea, no un paso opcional.** Toda implementación que modifique el comportamiento de un componente, cambie una constante o umbral, agregue o elimine un archivo, o altere el flujo del pipeline debe terminar con la actualización de este archivo. No esperar a que el usuario lo pida.

Revisar siempre al finalizar:
- ¿La sección de **Descripción Detallada de Archivos** refleja el estado actual? — Constantes, umbrales, firmas de métodos, comportamiento documentado
- ¿La sección de **Stack Tecnológico** necesita actualizarse? — Nueva librería, nuevo modelo, versión cambiada
- ¿La sección de **Arquitectura del Pipeline** sigue siendo precisa? — Flujo de datos, elementos GStreamer, probes
- ¿La sección de **Capacidades del Sistema** refleja el comportamiento actual? — Umbrales, eventos emitidos, lógica de decisión
- ¿La tabla de paquetes/capacidades cambió?

Este archivo es la guía de trabajo de Claude en este proyecto. Si no se mantiene actualizado, el próximo agente trabajará con información incorrecta y repetirá errores ya resueltos.

### 10. Consultar y mantener `ErrorHistory.md`

`ErrorHistory.md` es la primera fuente a consultar ante cualquier error, y la última acción al resolverlo.

**Antes de diagnosticar un error → leer `ErrorHistory.md`:**
- Buscar si el mensaje de error, traceback, o componente involucrado aparece en el historial
- Si hay una entrada que coincide, aplicar la solución documentada antes de intentar cualquier otra cosa
- Si la solución del historial no resuelve el problema, continuar con diagnóstico normal e indicarlo al usuario

**Al resolver un error → agregar entrada en `ErrorHistory.md`:**

```markdown
## [Fecha] — Título breve del error

**Contexto:** dónde ocurrió (archivo, componente, etapa del pipeline)

**Error en consola:**
```
<output exacto del error, traceback, o mensaje de log>
```

**Causa raíz:** explicación concisa de por qué ocurría

**Solución:** qué se cambió y en qué archivo(s)

**Fuente externa:** [título](url) — si se consultó documentación, issue, foro o artículo externo
```

Este historial sirve para:
- No repetir el mismo proceso de diagnóstico en el futuro
- Identificar patrones de errores recurrentes
- Compartir conocimiento con el equipo

### 11. Registrar mejoras futuras en `Future.md`

Cuando en una conversación surja una posible mejora — por ejemplo, "ahora usamos X que es simple, pero en el futuro podríamos usar Y que sería más preciso/eficiente" — registrarla en `Future.md` (en la raíz del repo) antes de continuar.

**Al agregar una entrada en `Future.md`:**

```markdown
## [Título de la mejora]

**Descripción:** qué es esta implementación futura y qué resuelve o mejora

**Por qué sería mejor:** ventaja concreta sobre la solución actual (precisión, velocidad, escalabilidad, etc.)

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: `_AgeGenderHandler` (líneas aprox. XXX–XXX)
- Descripción de lo que se reemplaza: el sistema de votación simple actual

**Tech stack propuesto:**
- Modelo / librería: nombre + versión + licencia
- Forma de integración: SGIE / worker Python / reemplazo de config / etc.

**Consideraciones:** dependencias, tamaño del modelo, compatibilidad con Jetson Orin Nano, esfuerzo estimado
```

`Future.md` no es un backlog de tareas — es un registro de ideas técnicas con suficiente contexto para poder evaluarlas e implementarlas después sin tener que redescubrir la conversación original.

### 12. Mantener `APIBackend.md` actualizado cuando cambia la API

`APIBackend.md` (en la raíz del repo) es el contrato entre el Jetson y el backend. Cada vez que se modifique algo relacionado con la comunicación Jetson ↔ Backend, actualizar este archivo también.

**Qué cuenta como cambio de API:**
- Agregar, renombrar o eliminar un campo en cualquier payload de `NxApiClient` (en `probes.py`)
- Agregar o modificar un tipo de evento (`type` field)
- Cambiar un endpoint (`/api/events`, `/api/analytics`, `/api/crops`, etc.)
- Cambiar la frecuencia de envío (e.g. `ANALYTICS_SEND_INTERVAL_SECS`)
- Agregar un nuevo feature que genere eventos (nuevo handler)
- Cambiar la semántica de un campo existente (e.g. cambiar unidades, rango de valores)

**Qué actualizar en `APIBackend.md`:**
- §3 si cambia un endpoint o sus campos comunes
- §4 si cambia el payload de un tipo de evento (mostrar el JSON actualizado)
- §5 si cambia telemetría continua (`analytics_snapshot`, posiciones, reference-frame)
- §7 si cambia cómo se calcula una métrica de negocio en el backend
- Agrega una nueva sección §4.x para cada nuevo feature con su payload completo

### 13. Escribir `Continue.md` cuando el usuario lo pide

Cuando el usuario diga "escribe un Continue.md" (o variantes como "crea el Continue.md", "genera el Continue.md"), crear o sobreescribir el archivo `Continue.md` en la raíz del repo con el siguiente contenido y formato exacto:

```markdown
# Continue.md — [fecha YYYY-MM-DD]

## Qué estábamos haciendo exactamente
[Descripción concreta de la tarea en curso: feature, bug, experimento. Una oración por punto.]

## Estado actual
**Qué funciona:**
- [ítem]

**Qué no funciona / está roto:**
- [ítem]

## Decisiones tomadas y por qué
- **[decisión]:** [razón concreta]

## Qué intentamos que NO funcionó
- **[enfoque]:** [por qué falló o fue descartado]

## Próximos pasos concretos
1. [paso concreto — archivo + qué cambiar]
2. ...

## Parámetros y valores concretos en juego
- [variable / config / threshold]: [valor actual y por qué importa]

## Error / síntoma actual (si aplica)
```
[traceback exacto, output de log, o descripción del comportamiento inesperado en este momento]
```

## Archivos modificados sin commitear
- `[archivo]` — [qué se cambió y si está funcional o a medias]

## Archivos y secciones que estábamos modificando
| Archivo | Función / sección | Qué se estaba cambiando |
|---------|-------------------|-------------------------|
| `deploy/pipelines/probes.py` | `_XxxHandler` | [descripción] |
```

**Reglas al escribir el Continue.md:**
- Ser específico: nombres de funciones, líneas aproximadas, valores concretos — no frases genéricas
- La sección "Qué intentamos que NO funcionó" es obligatoria aunque sea breve; es la más valiosa para no repetir errores
- La sección "Error / síntoma actual" es obligatoria si hay un error activo — pegar el traceback o log exacto, no parafrasearlo
- "Archivos modificados sin commitear" es obligatoria — si no hay ninguno, escribir "Ninguno"
- Los próximos pasos deben ser accionables desde cero: suficiente contexto para que Claude retome sin leer toda la conversación
- No incluir código extenso — solo referencias a archivos y funciones

### 14. Documentar siempre el código — docstrings y comentarios obligatorios

**Todo código Python en este proyecto debe cumplir los tres niveles de documentación:**

**Nivel 1 — Docstring de módulo (al inicio de cada archivo `.py`):**
- Qué hace el archivo y cuál es su rol en la arquitectura
- Relación con otros módulos (quién lo importa, a quién llama)
- Ejemplo de uso si no es obvio

**Nivel 2 — Docstring en toda función, método y clase:**
- Qué hace, qué recibe y qué retorna
- Efectos secundarios relevantes (escritura a disco, Redis, cola, API)
- Cuándo puede retornar None / lanzar excepción
- Para clases: invariantes del estado interno que el lector debe conocer

**Nivel 3 — Comentarios inline en bloques y líneas importantes:**
- Un comentario de sección (`# ── Nombre ─────`) antes de cada bloque lógico dentro de una función larga
- Comentario en líneas con lógica no obvia: expresiones matemáticas, indexación matricial, flags de estado, decisiones de diseño no evidentes en el nombre de la variable
- Comentarios en líneas con sintaxis densa (list comprehensions complejas, slicing múltiple, operaciones numpy en una línea)
- Comentario explicando el **por qué** de una constante o threshold numérico

**Lo que NO requiere comentario:**
- Líneas donde el nombre de la variable/función ya explica todo (ej. `logger.info(...)`, `result.append(item)`)
- Bloques de logging, imports, y asignaciones triviales
- Código que ya tiene un docstring inmediatamente encima

**Al crear o modificar un archivo Python:**
- Si el archivo no tiene docstring de módulo → agregarlo
- Si una función no tiene docstring → agregarlo antes de salir
- Si se agrega código con lógica compleja → agregar comentarios inline en ese momento

**Esta regla aplica tanto al código nuevo como al código modificado.** No es necesario documentar retroactivamente el código que no se tocó en la sesión actual.

---

## Descripción Detallada de Archivos

### `deploy/pipelines/` — Núcleo del pipeline

**`app.py`** (~360 líneas)
Pipeline de producción. Construye el grafo GStreamer dinámicamente según las cámaras y capacidades activas. Conecta fuentes RTSP del DVR (H.264 o H.265, detección automática), configura PeopleNet como PGIE, añade SGIEs opcionales según el paquete. **Sin tiler en producción** — el path es `caps_rgba → probe → fakesink`; el probe recibe frames RGBA full-res por cámara. Sin `nvdsosd`. Maneja el ciclo de vida de workers async (start/stop). Lee configuración a través de `config_loader.py`. Cuando `NX_QA_ENABLED=true`: crea el tiler (`nvmultistreamtiler` 640×360) después de `caps_rgba`, conecta **Probe A** (`pre_tiler_analytics_probe`) al src-pad de `caps_rgba` y **Probe B** (`osd_sink_pad_buffer_probe`) al src-pad del tiler, llama a `init_qa_cameras()`, `init_qa_grid()` e `init_pipeline_stats()`, arranca `MjpegServer` en :8080, y publica `nx:qa:status` (con `component_resolutions`) en Redis.

**`app_video_testing.py`** (~240 líneas)
Igual que `app.py` pero para archivos MP4 locales. Usa `filesrc + qtdemux` en lugar de `rtspsrc`. Útil para desarrollo y QA sin DVR físico. Sale con RTSP output en lugar de MJPEG. Acepta `--capabilities` y `--client` por CLI.

**`probes.py`** (~1800 líneas)
El motor central de analytics. Contiene tres probes y sus helpers:
- `NxApiClient`: cola async → thread worker → HTTP POST al backend (fire-and-forget, no bloquea). Con `NX_QA_ENABLED`: además publica a `nx:qa:apicalls` antes del POST.
- `_AgeGenderHandler`: acumula 10 votes del SGIE antes de confirmar clasificación
- `_FallDetectionHandler`: despacha crops al `PoseWorker`, aplica 3 reglas geométricas
- `_FaceRecognitionHandler`: cruza detecciones del SGIE FaceDetectIR con el `FaceRecognizer`
- `pre_tiler_analytics_probe`: **Probe A** (QA mode). Se conecta al src-pad de `caps_rgba` (RGBA full-res, batched). Gestiona todo el ciclo de vida de tracks, despacha handlers, llama `_update_fps_stats()`, escribe `_track_labels[track_id]` para Probe B. Lectura lazy del frame (solo cuando hay detecciones Y pixel workers activos).
- `osd_sink_pad_buffer_probe`: dispatcher. En QA mode delega a `_qa_overlay_probe`; en producción delega a `_production_analytics_probe`.
- `_qa_overlay_probe`: **Probe B** (QA mode, post-tiler). Lee frame tileado 640×360, dibuja bboxes de personas con labels de `_track_labels`, empuja frames a queues MJPEG, publica `nx:qa:detections`.
- `_production_analytics_probe`: analytics completos en frames full-res con lectura lazy; sin código QA.
- Stubs `_EppHandler`, `_FireSmokeHandler`, `_LicensePlateHandler` (pendiente integración)
- **QA helpers**:
  - `init_qa_cameras(channels)`: inicializa `camera_frame_queues` por cámara
  - `init_qa_grid(cols, rows, cell_w, cell_h)`: guarda dimensiones del tiler para crops por cámara
  - `init_pipeline_stats(channels)`: escribe `nx:qa:pipeline_stats` inicial en Redis (FPS en 0)
  - `_update_fps_stats(pad_index)`: cuenta frames, publica FPS a Redis cada 5 s
  - `_draw_qa_overlays(frame_bgr, qa_tracks)`: dibuja bboxes + labels con OpenCV (CPU, ~1 ms)
  - `_is_capability_active(cap)`: lee Redis hash `nx:qa:capabilities` para toggles de features
  - `_qa_publish(channel, data)`: fire-and-forget pub/sub a Redis

**`mjpeg_server.py`** (~130 líneas)
Servidor HTTP MJPEG daemon. Solo se instancia cuando `NX_QA_ENABLED=true` (desde `app.py`). Expone:
- `/stream/all` — frame tileado completo (640×360) con todos los bboxes dibujados
- `/stream/<camera_id>` — crop individual de esa cámara (e.g. `/stream/jetson-nx-001-ch01`)

Arquitectura interna de dos threads: `_encode_loop` (consume queues de frames, encoda a JPEG en background) + hilo HTTP (`run()`, sirve multipart/x-mixed-replace a 25 fps con lock mínimo). Calidad JPEG configurable (default 72). Cero impacto en producción sin QA.

**`config_loader.py`** (~330 líneas)
Carga y fusiona configuración desde 5 fuentes (prioridad: env vars > `/etc/nx_*` > `config.yaml` > `.env` > defaults). Define los 15 paquetes predefinidos (`PACKAGE_DEFINITIONS`), capacidades válidas, límites de NVDEC, y genera URLs RTSP interpolando el patrón del DVR. Retorna un `ClientConfig` dataclass. Campos configurables desde `config.yaml` (con defaults): `pgie_batch_size=0`, `pgie_interval=-1`, `sgie_interval=-1`, `reid_gallery_size=10` (máximo de embeddings por persona en `ReIdManager`; ajustar según número de cámaras del cliente), `recording_enabled=false` (activar grabación automática de clips cuando se detectan personas; funciona tanto en producción como en QA — en QA siempre graba independientemente de este campo).

**`common/bus_call.py`**
Handler genérico de mensajes del bus GStreamer (EOS, WARNING, ERROR). Estándar de ejemplos NVIDIA DeepStream.

**`common/FPS.py`**
Medidor de FPS con ventana de 5 segundos. Clase `GETFPS` con `get_fps()` y `print_data()`.

**`face_recognizer.py`** (~162 líneas)
Worker thread para reconocimiento facial. Carga `known_faces.json` (nombre → lista de embeddings). Para cada crop de rostro: extrae embedding 512-dim con InsightFace buffalo_l, calcula similitud coseno contra la DB, acumula 3 votos antes de bloquear identidad. Threshold: ≥ 0.50.

**`pose_worker.py`** (~213 líneas)
Worker thread para detección de caídas. Ejecuta MoveNet Lightning ONNX (entrada 192×192). Aplica 3 reglas: ángulo del torso > 45°, bbox más ancho que alto, caderas cerca de los tobillos. Caída si ≥ 2/3 reglas. Cooldown de 4 segundos por `track_id`.

**`appearance_worker.py`** (~139 líneas)
Worker thread para generación de embeddings de apariencia. Ejecuta OSNet-x0.25 ONNX (entrada 128×256), genera vector 512-dim L2-normalizado por persona. Crop enviado al worker en 3 momentos: (1) primer frame del track, (2) cada 15 frames hasta recibir la primera embedding, (3) cada 90 frames después del primer match para mantener el DB fresco. El resultado es consumido y limpiado (`clear_result`) por `_handle_appearance_reid()` en `probes.py`.

**`reid_manager.py`** (~235 líneas)
Gestor local de identidades cross-cámara. Mantiene un dict en memoria (`global_id → _Entry`) con **galería de embeddings**, timestamps y cámara actual. Cada `global_id` almacena hasta `GALLERY_MAX_SIZE=5` vectores que representan distintos ángulos/poses. El matching usa `max(query @ emb_i for emb_i in gallery)` — si cualquier ángulo coincide, la identidad se reconoce aunque el ángulo actual difiera del resto. API pública:
- `match_or_create(embedding, camera_id)` — matching por max-similitud sobre la galería, retorna `(global_id, event_type, prev_camera_id)`.
- `update_embedding(global_id, embedding)` — intenta añadir el embedding a la galería de un `global_id` conocido (diversity check: solo si `max_sim < GALLERY_DIVERSITY_THRESHOLD=0.85`); reemplaza el miembro menos informativo cuando la galería está llena.
- `flush()` — persiste a disco al apagar el pipeline.
Función libre `_gallery_add(gallery, embedding)` — lógica de adición/reemplazo en la galería; `_gallery_best_sim(gallery, embedding)` — max dot product contra todos los miembros. Persiste la DB en `deploy/reid_db.json` cada 30 s. **Migración automática** de esquema antiguo (`"embedding"` → `"gallery"`) en `_load()`. Carga al inicio descartando entradas con TTL vencido. Constantes configurables: `SIMILARITY_THRESHOLD=0.55` (guía: 0.65 muy estricto, 0.45 causa falsos positivos), `GALLERY_MAX_SIZE=10`, `GALLERY_DIVERSITY_THRESHOLD=0.85`, `PRESENCE_WINDOW_S=300`, `REID_TTL_S=3600`.

**`recording_manager.py`** (~230 líneas)
Worker thread para grabación automática de clips de video en QA mode. Suscribe a `nx:qa:detections` (Redis pub/sub) y arranca grabación cuando detecta personas. Estado: IDLE → RECORDING → IDLE. Graba `tiled.mp4` (desde `push_tiled_frame()`, llamado por MjpegServer) y `<camera_id>.mp4` full-res (desde `push_camera_frame()`, llamado por Probe A). Cooldown: 10 s sin detecciones. Duración mínima: 5 s (clips más cortos se descartan). Duración máxima: 5 min. Auto-prune: elimina clips más antiguos cuando total > 10 GB. Publica `nx:qa:recording_active` y `nx:qa:recording_info` a Redis. Directorio `/nx_tech/recordings/<YYYYMMDD_HHMMSS>/`: `thumbnail.jpg`, `metadata.json`, `tiled.mp4`, `<cam_id>.mp4`.

**`ws_client.py`** (~136 líneas)
WebSocket persistente hacia el backend. Envía snapshots de posiciones normalizadas (x, y, track_id) cada 10 segundos por cámara. Reconexión automática con backoff exponencial (1s → 30s). Silencioso si no hay conexión.

---

### `deploy/tools/` — Scripts utilitarios

**`setup.sh`** (~629 líneas)
**El único comando que ejecuta el técnico instalador.** Realiza la configuración completa del Jetson desde cero:
- Instala Docker CE, Tailscale, x11vnc
- Configura auto-login GDM y SSH con clave pública
- Escanea la red con nmap para encontrar DVRs en puerto 554
- Ejecuta `identify_dvr.py` para detectar marca y patrón RTSP
- Ejecuta `probe_cameras.py` para encontrar canales con cámaras activas
- Descarga modelos públicos (MoveNet, OSNet) vía `download_models.py`
- Escribe `/etc/nx_client`, `/etc/nx_sector`, `/etc/nx_pipeline`, `/etc/nx_dvr_ip`
- Construye la imagen Docker (`docker build`)
- Lanza el pipeline (`docker compose up -d`)

Flags principales: `--client`, `--package`, `--authkey`, `--api-key`, `--stream-type {main|sub}`, `--entry-exit-channels`, `--no-vnc`, `--no-docker`.

**`update.sh`** (~5 KB)
Actualización inteligente. Hace `git pull`, detecta si cambiaron el Dockerfile o requirements.txt, y solo reconstruye la imagen si es necesario. Reinicia el pipeline.

**`download_models.py`** (~4.7 KB)
Descarga modelos públicos que no están en el repo (MoveNet Lightning ONNX desde GitHub, OSNet desde un mirror). Verifica tamaño del archivo descargado.

**`identify_dvr.py`** (~18 KB)
Auto-detecta la marca del DVR probando patrones RTSP conocidos (Hikvision, Dahua, Reolink, Uniview, Axis, Hanwha, genérico). Soporta `--stream-type sub` para sub-streams en deployments de 16+ cámaras. Retorna la marca, patrón URL y cantidad de canales.

**`probe_cameras.py`** (~10.6 KB)
Dado un patrón RTSP y una lista de canales, usa `gst-discoverer` para verificar cuáles están activos y tienen video. Retorna solo los canales con señal válida.

**`register_face.py`** (~7.6 KB)
CLI para enrolamiento de rostros en la DB local. Acepta imágenes individuales, frames de video, o carpeta completa. Genera embeddings con InsightFace y los guarda en `known_faces.json`.

**`test_rtsp.py`** (~2 KB)
Test rápido de conectividad RTSP. Útil para verificar credenciales DVR antes de despliegue completo.

---

### `deploy/models/` — Modelos TensorRT

**`peoplenet_vpruned_quantized_decrypted_v2.3.4/`**
- `nvinfer_config.txt`: Config DeepStream para PGIE. `gie-unique-id=1`, INT8, batch=4, interval=4, 3 clases (person, bag, face).
- `resnet34_peoplenet_int8.onnx`: Modelo cuantizado INT8.
- `*.engine`: Engine TensorRT compilado por dispositivo (se regenera automáticamente).

**`resnet_age_gender_FB2/`**
- `config_infer.txt`: Config para SGIE de edad/género. `gie-unique-id=2`, FP16, opera sobre `class-ids=0` (personas) del PGIE.
- `custom_softmax_parser.so`: Plugin C++ compilado por `docker-entrypoint.sh` para parsear salida softmax del clasificador.

**`facedetect_ir/`**
- `config_infer.txt`: Config para SGIE de detección de rostros de alta precisión. `gie-unique-id=3`, FP16, opera sobre class=0 del PGIE.

---

### `deploy/clients/` — Configuración por cliente

**`clients/<nombre>/config.yaml`**
Config no-sensible del cliente: nombre, puerto DVR, patrón RTSP, canales activos, paquete, tipo de stream (main/sub), tracker (nvdcf/iou), canales de entrada/salida.

**`clients/<nombre>/.env`**
Credenciales DVR (`DVR_USER`, `DVR_PASS`). **Gitignoreado.** Se genera en `setup.sh`.

**`.env`** (raíz de deploy)
Credenciales del backend: `API_BASE_URL`, `API_KEY`, `WS_BASE_URL`. **Gitignoreado.**

---

### `deploy/qa_app/` — Dashboard QA Visual (Streamlit)

**`streamlit_app.py`** (~620 líneas)
Dashboard Streamlit accesible vía Tailscale desde cualquier dispositivo del equipo NX. Se autorefresea cada 500 ms (`st_autorefresh`). Sidebar: info del pipeline, selector de cámara, toggles de features, sección Entrada/Salida, editor de config completo, indicador de estado de grabación/playback. Main area con dos tabs:
- **Tab "En Vivo"**: video MJPEG embebido (`st.iframe`), log de detecciones, log de API calls.
- **Tab "Grabaciones"**: estado de grabación activa (Redis `nx:qa:recording_active`), estado de playback (Redis `nx:qa:playback_video`), lista de clips de `/nx_tech/recordings/` con thumbnail+metadata, botón **▶ Preview** (`st.video()`), botón **▶ Correr Inferencia** (escribe `nx:qa:playback_video` en Redis → pipeline se reinicia en modo playback), botón **🗑 Eliminar**, botón **🔴 Volver a En Vivo** (borra `nx:qa:playback_video`). El volumen `./recordings:/nx_tech/recordings` en `docker-compose.qa.yml` es el que permite a Streamlit leer los clips.

**`Dockerfile.qa`**
Imagen `python:3.11-slim`. Instala solo `streamlit`, `redis`, `streamlit-autorefresh` y `ruamel.yaml`. No necesita acceso a GPU. Healthcheck en `/_stcore/health`.

**`requirements.txt`**
`streamlit>=1.32`, `redis>=5.0`, `streamlit-autorefresh>=1.0.0`, `ruamel.yaml>=0.18`.

---

### `deploy/` — Archivos de orquestación Docker

**`docker-compose.yml`**
Tres servicios: `deepstream` (pipeline principal, puerto 8080 expuesto — activo solo con QA), `db` (TimescaleDB PostgreSQL 16, puerto 5432), `redis` (Redis 7, puerto 6379). Monta los directorios de pipelines, modelos, clientes y tools.

**`docker-compose.qa.yml`**
Override file para modo QA. Solo se carga desde `qa.sh`. Agrega `NX_QA_ENABLED: "true"` al servicio `deepstream` y añade el servicio `qa_app` (Streamlit, puerto 8501). Monta `./clients:/nx_tech/clients` y `./recordings:/nx_tech/recordings` en el servicio `qa_app` para que el botón Guardar pueda escribir configs y la tab Grabaciones pueda leer/previsualizar clips. El servicio `deepstream` también monta `./recordings:/nx_tech/recordings` para que `RecordingManager` pueda escribir clips. Nunca se usa en producción.

**`qa.sh`**
Script de activación del modo QA. Subcomandos: `start` (default) y `stop`. Al arrancar: detiene el pipeline de producción, inicia los containers con el override QA, espera a que Streamlit esté listo, imprime la URL Tailscale clickable. `Ctrl+C` o `stop` restauran la producción automáticamente vía `trap cleanup`.

**`Dockerfile.jetson`**
Imagen ARM64 basada en `nvcr.io/nvidia/deepstream:7.1-samples-multiarch`. Instala pyds 1.1.11, onnxruntime-gpu para aarch64, insightface ≥ 0.7.3.

**`docker-entrypoint.sh`**
Se ejecuta al iniciar el contenedor: (1) compila `custom_softmax_parser.so` para el SGIE de edad/género, (2) parchea el ONNX de PeopleNet para batch dinámico, (3) pre-descarga InsightFace buffalo_l si `face_recognition` está en el pipeline, (4) elimina engines stale si el ONNX fue modificado.

**`API_REFERENCE.md`**
Especificación completa de la API REST y WebSocket entre el Jetson y el backend NX. Todos los eventos, formatos JSON, campos requeridos, y semántica de severity.

---

### Archivos en la raíz del repo

**`README.md`** — Documentación principal. Fuente de verdad para paquetes, capacidades, patrones DVR y flujo de instalación. **Siempre revisar antes de hacer cambios.**

**`plan.md`** — Plan técnico para soporte de sub-streams (en progreso).

**`Planeacion-modular.md`** — Documento de diseño de la arquitectura modular actual (referencia histórica).

**`Plan_face_fall.md`** — Plan de implementación de face recognition y fall detection (referencia histórica).

**`ErrorHistory.md`** — Historial de errores resueltos. Ver regla 10 para el formato de entradas.

**`Future.md`** — Registro de mejoras técnicas futuras. Ver regla 11 para el formato de entradas.

---

## Variables de Entorno Importantes

| Variable | Fuente | Descripción |
|----------|--------|-------------|
| `NX_PIPELINE` | `/etc/nx_pipeline` | Capacidades activas, ej: `people_counting,age_gender` |
| `NX_CLIENT` | `/etc/nx_client` | Nombre del cliente, ej: `demo` |
| `NX_SECTOR` | `/etc/nx_sector` | Sector: `comercio`, `industrial`, `hogar` |
| `NX_DVR_IP` | `/etc/nx_dvr_ip` | IP del DVR detectada por setup.sh |
| `JETSON_ID` | `docker-compose.yml` | Identificador único del dispositivo |
| `API_BASE_URL` | `.env` | URL del backend NX |
| `API_KEY` | `.env` | Token de autenticación hacia el backend |
| `WS_BASE_URL` | `.env` | URL WebSocket para telemetría de posiciones |

---

## Notas de Rendimiento (Jetson Orin Nano)

- Máximo recomendado: 6 streams main (1920×1080) o 16 streams sub (960×544)
- `network-mode=1` (INT8) para PeopleNet; `network-mode=2` (FP16) si falla calibración INT8
- `classifier-async-mode=1` en SGIEs para no bloquear el pipeline
- Workers Python usan CPU + ONNX Runtime; no compiten con TensorRT por CUDA
- Los engines `.engine` se reconstruyen automáticamente al primer run por dispositivo (~5 min/modelo)
