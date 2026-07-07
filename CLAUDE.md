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
│   │   ├── probes.py           # Probe GStreamer + handlers + API client + stream overlays
│   │   ├── stream_server.py    # Servidor MJPEG HTTP (activo con NX_STREAM_ENABLED=true)
│   │   ├── config_loader.py    # Carga y merge de configuración
│   │   └── common/             # Utilidades (FPS, bus_call)
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
│   ├── stream.sh               # Activar/desactivar stream mode con bboxes
│   ├── docker-compose.yml
│   ├── docker-compose.stream.yml  # Override stream: NX_STREAM_ENABLED=true
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

**Stream mode (`NX_STREAM_ENABLED=true`):**
```
DVR (RTSP) → rtspsrc → h264/h265parse → nvv4l2decoder
  → nvstreammux → nvinfer (PeopleNet PGIE, gie-id=1)
  → nvtracker → [SGIEs opcionales por paquete]
  → nvvideoconvert(RGBA) → capsfilter(RGBA)
  → [Probe: analytics + dibuja bboxes → camera_frame_queues] → fakesink
StreamServer (:8080) consume camera_frame_queues y sirve /stream/<cam_id>
```

> **Sin tiler.** El probe recibe frames RGBA full-res por cámara. En stream mode (`NX_STREAM_ENABLED=true`), el probe también dibuja bboxes sobre el frame y lo encola en `camera_frame_queues` para `StreamServer`.

**Workers async (Python threads, no bloquean el pipeline):**
- `FaceRecognizer` — InsightFace buffalo_l, reconocimiento facial; claves UUID del backend
- `JetsonSyncClient` — Socket.IO /jetson namespace; recibe face_update del backend y llama sync_from_backend()
- `NxApiClient` — cola async para REST API backend
- `WsPositionClient` — telemetría de posiciones / heatmaps en tiempo real

> **OSNet re-ID:** los embeddings 512-dim se extraen directamente del SGIE `sgie-appearance` (gie-id=3) vía `NvDsInferTensorMeta` — síncronos, sin worker Python, sin queue. `ReIdManager` sigue corriendo en el probe thread.

**Paquetes (definen qué capacidades se activan):**
| Sector | Paquetes | Capacidades |
|--------|----------|-------------|
| comercio | comercio_basico/avanzado/total/enterprise | conteo; avanzado+ agrega edad/género; total+ agrega face recognition |
| industrial | industrial_basico/avanzado/total/enterprise | conteo; total+ agrega face recognition |
| hogar | hogar_basico/avanzado/total | conteo; avanzado+ agrega face recognition |

La capacidad activa se lee desde `/etc/nx_pipeline` en el Jetson.

---

## Capacidades del Sistema

### Conteo de Personas (`people_counting`) — ✅ Activo
Detecta y trackea personas en cada cámara. Emite eventos `person_entry` y `person_exit` con tiempo de permanencia. Acumula conteos y envía snapshots de analytics cada 60 segundos al backend.
- **Modelo:** PeopleNet v2.3.4 (ResNet-34, INT8, NVIDIA NGC) — detecta 3 clases: person, bag, face
- **Tracker:** NvDCF (correlación, recomendado ≤6 streams) o IOU (ligero, hasta 16 streams)
- **Siempre activo** en todos los paquetes
- **`person_count` en analytics snapshot:** se incrementa **solo cuando `match_or_create()` retorna `event_type == "new_person"`**, es decir, cuando `ReIdManager` crea un nuevo `_Entry` con un `global_id` fresco. Las visitas de retorno (`person_return`) y cambios de cámara (`channel_change`) no incrementan el contador — la misma persona física no se cuenta dos veces. Fallback sin ReID (`_reid_manager is None`): se cuenta por track, comportamiento legacy.

### Re-ID entre Cámaras (`appearance`) — ✅ Activo
Identifica cuando la misma persona aparece en cámaras distintas usando embeddings de apariencia. El matching ocurre **localmente en el Jetson** gracias al `ReIdManager`. Emite tres variantes de evento según el contexto:
- `person_entry` (`entry_type: "new"`) — persona nunca vista antes
- `person_entry` (`entry_type: "return"`) — misma persona, reapareció tras > 5 min de ausencia
- `person_channel_change` — misma persona, cambió de cámara dentro de la ventana de presencia (≤5 min)

La emisión de `person_entry` se **difiere** hasta que el embedding esté listo (deadline 30 frames / ~1 s a 30fps como fallback de seguridad). Con el SGIE el embedding llega en el mismo frame — el fallback aplica si el bbox nunca supera el mínimo de 96×192 px del SGIE (`input-object-min-width/height`, subido desde 32×32 — el modelo espera 256×128 nativo, y crops más chicos generaban embeddings ruidosos que contaminaban la galería de ReID). Este mínimo es un filtro de calidad intencional, no solo un edge case: una persona por debajo de ese tamaño no se cuenta en `person_count`, no obtiene ReID cross-cámara, no se le reconoce el rostro (gate en `global_id`), y no aporta posición/heatmap — cae en el fallback con `global_id=None` en vez de generar una identidad poco confiable.

- **Embedding:** OSNet-x1.0 — vectores 512-dim L2-normalizados, extraídos **directamente del SGIE DeepStream** (gie-id=3) vía `NvDsInferTensorMeta` en el probe. Sin `AppearanceWorker`, sin Python thread, sin cola. DeepStream gestiona el crop y el engine TRT.
- **Matching:** max-similitud coseno ≥ 0.68 (`SIMILARITY_THRESHOLD`) sobre **galería de hasta 10 embeddings** por persona (`GALLERY_MAX_SIZE`, configurable por cliente vía `reid_gallery_size`). `ReIdManager` — O(N×K), vectorizable con numpy. Nuevos ángulos se añaden a la galería solo si son suficientemente distintos a los existentes (sim < 0.85); cuando la galería está llena se reemplaza el miembro menos informativo.
- **Same-camera re-detection:** si `channel_change` ocurre con `prev_camera == camera_id` (tracker pierde y re-detecta en la misma cámara), se demota a `person_return` para no emitir un evento de cambio de cámara espurio.
- **Persistencia:** `deploy/reid_db.json` — sobrevive reinicios; TTL 1 hora sin actividad
- **Ventana de presencia:** 5 min (configurable en `reid_manager.py` como `PRESENCE_WINDOW_S`)
- Se activa automáticamente si el ONNX existe en `models/osnet/` (el engine TRT se compila en el primer arranque, ~2 min extra)
- **Limpieza cruzada con reconocimiento facial:** `match_or_create()` retorna una 4ª posición, `expired_ids` — los `global_id`s que `_expire_stale()` acaba de olvidar (TTL 1h). `probes.py::_handle_appearance_reid()` usa esa lista para llamar `FaceRecognizer.forget(gid)` y limpiar `_employee_by_global_id` — sin esto, el estado de votos/candado de caras y el tag de empleado por `global_id` crecerían indefinidamente, ya que `ReIdManager` y `FaceRecognizer` son diccionarios independientes que nada más sincroniza.
- **Log CSV persistente (`clients/<cliente>/logs/osnet_reid.csv`):** igual que `face_recognition.csv`, corre siempre en producción (no gateado por `NX_STREAM_ENABLED`) — una fila por cada creación/match/refresh de galería en `ReIdManager.match_or_create()`/`update_embedding()`, con `camera_id,track_id,global_id,event,similarity,gallery_size,added_angle,prev_camera,absent_s`. `event` es `new_person`/`person_return`/`channel_change`/`gallery_refresh`. `track_id` es solo para el log (no afecta el matching). Pensado para analizar después similitud y comportamiento de la galería (ej. detectar casos borderline cerca del `SIMILARITY_THRESHOLD`), no para debugging en vivo. `RotatingFileHandler` (stdlib): 20 MB × 5 archivos. El logger `logging.getLogger("nx.osnet_csv")` se crea dentro de `ReIdManager` (no en `probes.py`) porque ahí es donde ya se calculan `best_sim`/`gallery_size`/`absent`; `init_workers()` le pasa `csv_log_dir` (mismo `clients/<cliente>/logs/` que usa `_face_csv_logger`) al instanciarlo.

### Edad y Género (`age_gender`) — ✅ Activo
Clasifica a cada persona detectada en una de 6 categorías: female_young, female_adult, female_senior, male_young, male_adult, male_senior. Requiere al menos 10 muestras del SGIE antes de confirmar la clasificación (sistema de votación para reducir falsos positivos).
- **Modelo:** ResNet-18 Pedestrian Attributes FB2 (FP16, SGIE gie-id=2)
- **Umbral:** bbox mínimo 64×160px, confianza ≥ 0.3
- **Parser custom:** `custom_softmax_parser.so` compilado en el entrypoint del contenedor

### Reconocimiento Facial (`face_recognition`) — ✅ Activo
Identifica personas conocidas (empleados, residentes) a partir de una base de datos de embeddings faciales. Usa PeopleNet class 2 (face) para detectar rostros, luego un worker Python extrae el embedding y lo compara con la DB. No hay SGIE dedicado para caras — el SGIE FaceDetectIR fue eliminado.
- **Detección:** PeopleNet class_id=2 (face) — mismo PGIE que detecta personas, sin SGIE adicional. Filtro de tamaño mínimo `[class-attrs-2]` en `nvinfer_config.txt`: `detected-min-w/h=64` — descarta caras menores a 64×64px en la GPU, antes de que el crop llegue a `FaceRecognizer` (ver detalle y justificación del valor en la sección de modelos).
- **Embedding:** InsightFace buffalo_l — ArcFace 512-dim, threshold similitud coseno ≥ 0.50. Corre en **CPU** (`CPUExecutionProvider`, `ctx_id=-1`) — deliberado, no pendiente de optimizar con un flag. Se probó `CUDAExecutionProvider` (2026-07-07) y se revirtió sin desplegar: el wheel `onnxruntime-gpu` instalado en `Dockerfile.jetson` (`nschloe/onnxruntime-aarch64-ubuntu22`) es el mismo que causó "kernel Cask errors" (choque de contexto CUDA con TensorRT) durante la migración de OSNet — ver `Future.md` sección "CHANGE TO OSNET1". El camino real a GPU para face recognition es el mismo que ya se usó para OSNet: exportar buffalo_l (`det_10g.onnx` + `w600k_r50.onnx`) a TensorRT y correrlo como SGIE nativo de DeepStream, no onnxruntime-gpu con este wheel.
- **Worker:** `FaceRecognizer` (Python thread) — indexado por `global_id` de ReID, no por `track_id`. `track_id` se reinicia en cada cámara nueva, lo que obligaba a re-votar desde cero cada vez que el empleado cambiaba de cámara; con `global_id` la identidad ya bloqueada viaja automáticamente vía la continuidad de apariencia de `ReIdManager`. `probes.py::_FaceRecognitionHandler.process_face()` no alimenta al worker hasta que `_active_tracks[(pad_index, track_id)].global_id` esté resuelto — la espera es de pocos frames, insignificante frente al ciclo de votación.
- **Ventana de votos (`FACE_VOTES_REQUIRED=3`):** `deque(maxlen=3)` por `global_id`, se sigue alimentando aunque ya haya un candado — si la mayoría de la ventana cambia, se corrige el tag (`Face re-tagged` en logs). Salvaguarda contra que `ReIdManager`/OSNet le pase el `global_id` de un empleado a otra persona por error (ej. uniformes parecidos entre empleados) — la cara sigue siendo la única fuente de verdad para la identidad, ReID nunca la asigna por sí solo.
- **DB:** `known_faces.json` — formato nuevo: `{"<uuid>": {"name": "...", "embeddings": [[...]]}}`. Formato legacy (nombre-clave) sigue siendo compatible en lectura.
- **Registro automático:** `JetsonSyncClient` recibe `face_update` de backend via Socket.IO `/jetson` namespace, llama `sync_from_backend()` que hace GET `/api/employees/embeddings` y actualiza la DB en caliente sin reiniciar el pipeline
- **Ya no hay eventos discretos para comercio/industrial** (`employee_seen`/`employee_presence`/`employee_exit` eliminados). La identidad de empleado viaja dentro de `positions_snapshot` — `_accumulate_positions()` agrega `employee_id` (de `_employee_by_global_id`) y `face_confirmed` (booleano por ciclo de ~1s, `True` solo si se procesó una cara para ese `global_id` en ese ciclo) a cada posición. `_employee_by_global_id` dura mientras el `global_id` viva, pero **no es incondicional**: se limpia si `ReIdManager` expira el `global_id`, o si la ventana deslizante de votos (o un reload por revocación) hace que `FaceRecognizer` decida que ya no es ese empleado (ver `process_face`, rama `else`). El backend (`app/socket/positions.py`) solo persiste la asistencia de una estadía en cámara si tuvo al menos una confirmación de cara durante su vida.
- **Hogar** conserva `unknown_person_alert` como evento discreto (es alerta de intrusión, no asistencia) — pero comparte el mismo gate de `global_id` que el reconocimiento de empleados: en un Jetson sin OSNet instalado, tampoco dispara. Limitación aceptada y diferida — hogar no es prioridad de este rediseño.
- **`employee_id`:** UUID string del backend (`employees.id`) — no el nombre del empleado. Tageado sobre el `global_id`, nunca transmitido en eventos sueltos.
- **Overlay en stream mode:** el bbox label agrega `| <nombre> NN%` solo cuando `identity_key != "Unknown"`, resuelto a nombre legible vía `_face_recognizer.get_display_name()` — mismo lookup que ya usaba el log de consola `EMPLEADO`. (Antes había un bug: comparaba contra el literal `"Desconocido"` en vez de `"Unknown"` — la condición nunca filtraba nada — y dibujaba el UUID crudo en vez del nombre.)
- **Log CSV persistente (`clients/<cliente>/logs/face_recognition.csv`):** a diferencia de las líneas de consola `EMPLEADO`/`ROSTRO Desconocido` (gateadas por `NX_STREAM_ENABLED`), este log corre siempre en producción — una fila por cada muestra procesada de `process_face` (no dedupeado por track), con `camera_id,track_id,global_id,identity,similarity,status`. `identity` es el UUID crudo (o `"Unknown"`), no el nombre, para poder unir directamente contra `employees.id`. Pensado para análisis posterior de threshold/precisión, no para debugging en vivo. `RotatingFileHandler` (stdlib): 20 MB × 5 archivos por cliente. Logger `logging.getLogger("nx.face_csv")` se crea en `init_workers()` solo si `face_recognition` está en el pipeline — mismo bloque donde se instancia `FaceRecognizer`.

### Detección de EPP, Fuego/Humo, Placas — 🔄 Pendiente (no en MVP)
Removidas del MVP por falta de modelos entrenados. Ver `Future.md` para el plan de reintegración.

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
| **PeopleNet v2.3.4** | ONNX → TRT INT8 | Detección de personas, bolsas, rostros class 2 (PGIE) | Siempre activo |
| **ResNet-18 Pedestrian Attributes** | ONNX → TRT FP16 | Clasificación edad/género (SGIE) | `age_gender` |
| **InsightFace buffalo_l (ArcFace)** | ONNX (CPU/GPU) | Embeddings faciales 512-dim para re-ID | `face_recognition` |
| **OSNet-x1.0** | ONNX → TRT FP32 | Appearance vectors 512-dim para re-ID entre cámaras (~94% Rank-1 Market-1501) — SGIE gie-id=3 | Siempre activo (si ONNX existe) |

### Librerías Python
| Librería | Uso |
|----------|-----|
| **onnxruntime** (CPU, aarch64) | Inferencia ONNX para InsightFace ArcFace — deliberadamente CPU-only; el wheel `onnxruntime-gpu` instalado no es seguro de usar con CUDA aquí, ver sección Reconocimiento Facial |
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
- [ ] Regla 15: ¿Cambia el flujo general, un handler, un worker, o el pipeline? → [actualizar Concepts.md]
- [ ] Regla 16: ¿Se agrega/modifica/elimina un campo configurable? → [ClientConfig + load_config() + log_summary() + config.yaml sincronizados]
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
- Worker Python (modelo no-DeepStream) → crear `xxx_worker.py` con patrón queue + thread, como `appearance_worker.py` o `face_recognizer.py`

**Sistema de capacidades y paquetes (`config_loader.py`):**
- Agregar la nueva capacidad a `KNOWN_CAPABILITIES`
- Determinar a qué paquetes pertenece: ¿es una feature de comercio, industrial, hogar, o varios? Revisar la tabla de paquetes en `README.md` para decidir en qué niveles (básico/avanzado/total/enterprise) tiene sentido incluirla
- Agregar la capacidad a los paquetes correspondientes en `PACKAGE_DEFINITIONS`
- Si aplica a un nuevo sector, crear los paquetes necesarios también en `PACKAGE_DEFINITIONS`

**Referencia de paquetes actuales (MVP):**
| Sector | Paquetes | Capacidades incluidas |
|--------|----------|-----------------------|
| comercio | basico | people_counting |
| comercio | avanzado | people_counting, age_gender |
| comercio | total/enterprise | people_counting, age_gender, face_recognition |
| industrial | basico/avanzado | people_counting |
| industrial | total/enterprise | people_counting, face_recognition |
| hogar | basico | people_counting |
| hogar | avanzado/total | people_counting, face_recognition |

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
- **GIE unique IDs**: Cada nvinfer necesita un `gie-unique-id` único (1=PeopleNet, 2=AgeGender, 3=OSNet appearance SGIE)
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

### 15. Mantener `Concepts.md` actualizado cuando cambia la estructura general

`Concepts.md` (en la raíz del repo) es la guía de lectura del código — explica el flujo de datos,
el ciclo de vida de los tracks, el patrón worker, y cómo funciona cada detección.

**Actualizar `Concepts.md` cuando ocurra cualquiera de estos cambios:**
- Se agrega o elimina un handler (`_AgeGenderHandler`, `_FallDetectionHandler`, etc.)
- Se agrega o elimina un worker async (`AppearanceWorker`, `FaceRecognizer`, etc.)
- Cambia el flujo del pipeline GStreamer (nuevo elemento, cambio de probe, QA mode)
- Cambia cómo se emiten eventos al backend (nuevo tipo de evento, nueva lógica de ReID)
- Cambia el sistema de paquetes/capacidades (nuevo paquete, nueva capacidad)
- Cambia cómo funciona Redis en QA mode (nueva key, nuevo canal pub/sub)
- Cambia el flujo de instalación en campo (setup.sh, identify_dvr.py, probe_cameras.py)

**Lo que NO requiere actualizar `Concepts.md`:**
- Cambios en umbrales o constantes (eso va en CLAUDE.md y README.md)
- Cambios internos de implementación que no afectan el flujo visible desde afuera
- Fixes de bugs que no cambian el comportamiento descrito

**Formato:** mantener el mismo estilo — explicación conceptual en prosa + links a archivos con
`[nombre](ruta#Llinea)` para que sean clickables desde el IDE.

### 16. Mantener `config.yaml` y `config_loader.py` sincronizados

Cada vez que se agrega, elimina o cambia un campo configurable en `config_loader.py`, hay que mantener tres cosas en sincronía:

**a) `ClientConfig` dataclass** (`config_loader.py`):
- El campo existe con su tipo y valor por defecto correcto
- Está anotado con un comentario que indica el rango válido y el comportamiento del default

**b) `load_config()`** (`config_loader.py`):
- El campo se lee con `cfg.get("nombre_campo", default)` usando el mismo default que en el dataclass
- Si el campo afecta algo al arrancar, se loguea en `log_summary()`

**c) `clients/demo/config.yaml`**:
- El campo aparece en el archivo, ya sea activo o comentado
- Tiene un comentario que explica qué hace, cuál es el default, y un rango útil de valores
- Los campos opcionales van comentados con `# campo: valor_ejemplo` para que el técnico pueda activarlos sin buscar en el código
- El archivo `demo/config.yaml` es la plantilla de referencia — si se agrega un campo aquí, también hay que agregarlo a cualquier otro `clients/*/config.yaml` que exista

**Qué NO va en `config.yaml`:**
- Credenciales (`DVR_USER`, `DVR_PASS` → `.env`)
- IP del DVR (`dvr_ip` → `/etc/nx_dvr_ip`, escrita por `setup.sh`)
- Nombre del cliente (`client_name` → `/etc/nx_client`, escrita por `setup.sh`; se documenta en el yaml solo como referencia visual)

**Checklist al agregar un campo nuevo:**
- [ ] Campo en `ClientConfig` con tipo + default + comentario
- [ ] `cfg.get(...)` en `load_config()` con el mismo default
- [ ] Entrada en `log_summary()` si es relevante para diagnóstico
- [ ] Entrada comentada en `clients/demo/config.yaml` con descripción y rango
- [ ] Actualizar descripción de `config_loader.py` en CLAUDE.md (esta sección)

---

## Descripción Detallada de Archivos

### `deploy/pipelines/` — Núcleo del pipeline

**`app.py`** (~300 líneas)
Pipeline de producción. Construye el grafo GStreamer dinámicamente según las cámaras y capacidades activas. Conecta fuentes RTSP del DVR (H.264 o H.265, detección automática), configura PeopleNet como PGIE, añade SGIEs opcionales según el paquete. **Sin tiler** — el path siempre es `caps_rgba → probe → fakesink`; el probe recibe frames RGBA full-res por cámara. Sin `nvdsosd`. Maneja el ciclo de vida de workers async (start/stop). Si `NX_STREAM_ENABLED=true`: inicializa `camera_frame_queues` y arranca `StreamServer` en :8080 antes de iniciar el pipeline. El OSNet SGIE (`sgie-appearance`, gie-id=3) se agrega condicionalmente si `models/osnet/osnet_x1_0_market1501.onnx` existe en disco — independiente de `cfg.pipeline`.

**`app_video_testing.py`** (~230 líneas)
Igual que `app.py` pero para archivos MP4 locales. Usa `filesrc + decodebin` en lugar de `rtspsrc`. `decodebin` detecta el codec automáticamente. Las dimensiones del streammux se detectan con `cv2.VideoCapture` antes de construir el pipeline. Acepta `--capabilities`, `--client`, `--input` y `--no-loop` por CLI. Sink: siempre `fakesink` (mismo que producción). Si `NX_STREAM_ENABLED=true`: arranca `StreamServer` en :8080 para ver la inferencia sobre el video.

**`probes.py`** (~900 líneas)
El motor central de analytics. Probe único (`osd_sink_pad_buffer_probe`) en `caps_rgba src-pad` (frames full-res por cámara, sin tiler).
- `NxApiClient`: cola async → thread worker → HTTP POST al backend (fire-and-forget, no bloquea). Soporta callbacks de éxito por endpoint (`register_success_callback`) invocados desde el worker thread cuando el backend confirma 2xx.
- `_AgeGenderHandler`: acumula 10 votes del SGIE (gie-id=2) antes de confirmar clasificación; emite `person_classified`
- `_extract_osnet_embedding(obj_meta)`: lee el tensor 512-dim del SGIE OSNet (gie-id=3) desde `NvDsInferTensorMeta` — síncrono, sin thread. `_handle_appearance_reid()` lo llama por cada persona visible y lo pasa a `ReIdManager`.
- `_FaceRecognitionHandler`: cruza detecciones de cara de PeopleNet (class_id=2) con el `FaceRecognizer`, indexado por `global_id` (no `track_id`) una vez que ReID lo resuelve. Ya no emite eventos discretos para comercio/industrial — tagea `_employee_by_global_id`/`_face_confirmed_this_cycle`, consumidos por `_accumulate_positions` para que la identidad viaje en `positions_snapshot`. Solo `unknown_person_alert` (hogar) sigue siendo un evento discreto.
- `osd_sink_pad_buffer_probe`: probe único. Lazy frame read: GPU→CPU solo cuando workers necesitan pixels, `NX_STREAM_ENABLED=true`, o la escena está vacía y toca capturar reference frame (a lo sumo cada 30 s). Al final del loop de cámara, si stream mode activo: dibuja bboxes+labels con OpenCV y empuja a `camera_frame_queues[camera_id]`.
- **Reference frame — retry + cambio visual + filtro de brillo**: se evalúa cuando no hay personas visibles (`visible_ids` vacío) y el frame tiene suficiente iluminación (`_frame_is_bright_enough()`, media ≥ `REFERENCE_FRAME_MIN_BRIGHTNESS=30.0`/255 — rechaza frames nocturnos). El primer frame válido se envía y se reintenta cada `REFERENCE_FRAME_RETRY_SECS=30s` hasta confirmar 2xx. Una vez confirmado, solo se reenvía si han pasado `REFERENCE_FRAME_MIN_INTERVAL_SECS=86400s` (24 h) Y `_scene_changed()` detecta ≥ `REFERENCE_FRAME_CHANGE_THRESHOLD=0.15` (15 %) de diferencia normalizada por iluminación. **Importante:** el lazy frame read solo decodifica el frame cuando las condiciones de tiempo se cumplen (≥30 s sin confirmar, ó ≥24 h desde último confirmado) — no en cada frame vacío. Objetos no-persona detectados por PeopleNet (bolsos, caras sin cuerpo, `PGIE_CLASS_BAG`/`PGIE_CLASS_FACE`) no bloquean el reference frame. El backend guarda historial completo (INSERT, no UPSERT) para que las consultas históricas de heatmap usen el fondo correcto para cualquier período.
- **`_frame_is_bright_enough(frame_np)`**: redimensiona a 64×36, toma la media del canal gris; retorna `False` si media < `REFERENCE_FRAME_MIN_BRIGHTNESS` (30.0).
- **`_scene_changed(current_np, prev_np)`**: redimensiona a 64×36, normaliza por media para ignorar cambios de iluminación, compara diferencia absoluta media contra `REFERENCE_FRAME_CHANGE_THRESHOLD`.
- **Stream mode helpers**: `init_stream_grid(cols, rows, cell_w, cell_h)`, `tiled_frame_queue`, `_IS_STREAM_ENABLED`, `_track_labels` (dict compartido entre Probe A y Probe B, keyed por `track_id`), `_draw_tiled_overlays(frame_bgr, tracks)`, `tiled_overlay_probe` (Probe B, en el src pad del tiler — compone el frame final que consume `StreamServer`).
- **Label en stream (personas)**: muestra `#<display_id>` (número corto asignado la primera vez que el `global_id` de ReID resuelve, ver `_display_ids`) o `...` mientras espera. Los handlers appendean al prefijo existente: ej. `#3 | male_adult | 87%`. El `track_id` local ya no aparece en el label. Face recognition agrega `| <nombre> NN%` solo cuando hay match confirmado (`identity_key != "Unknown"`), resuelto vía `_face_recognizer.get_display_name()`.
- **Bboxes de cara en stream (debug)** — agregado para diagnosticar el pipeline de detección/reconocimiento facial: `osd_sink_pad_buffer_probe` registra en `_track_labels[face_track_id]` un label `Cara NN%` (confianza cruda de PeopleNet class 2) para **cada** detección en `face_metas`, sin el filtro `OSD_CONFIDENCE_THRESHOLD` ni el gate de `global_id` que usa `_face_handler.process_face` — el objetivo es ver exactamente qué está detectando el PGIE, no lo que ya pasó el pipeline de reconocimiento. `tiled_overlay_probe` dibuja también `PGIE_CLASS_FACE` (antes solo `PGIE_CLASS_PERSON`), en naranja (`(0, 200, 255)` BGR) para distinguirlo del verde de persona / rojo de caída. Las caras no tienen equivalente a `_active_tracks`/`_expire_lost_tracks`, así que `tiled_overlay_probe` poda cada frame las entradas `face: True` de `_track_labels` cuyo `track_id` ya no aparezca en el batch actual — evita crecimiento sin límite en sesiones largas.
- **Stream verbose output** (`_slog`, `_C`): cuando `NX_STREAM_ENABLED=true`, imprime líneas coloreadas a stdout (visibles en `docker logs -f`) por cada evento relevante: `DETECCIÓN` (tras ReID), `DEMOGRAFÍA` (clasificación edad/género), `EMPLEADO` (reconocimiento facial exitoso), `ROSTRO Desconocido` (cara vista sin match, una vez por track), y `[API]` (cada POST exitoso al backend). Desactivar colores ANSI con `NO_COLOR=1`. Sin overhead en producción.
- **Log CSV persistente de face recognition** (`_face_csv_logger`): a diferencia de `_slog`, corre siempre (no gateado por `NX_STREAM_ENABLED`). Escribe en `clients/<cliente>/logs/face_recognition.csv` una fila por cada muestra procesada en `_FaceRecognitionHandler.process_face` (no dedupeada por track): `timestamp,camera_id,track_id,global_id,identity,similarity,status`. Se inicializa en `init_workers()` junto con `FaceRecognizer`, vía `RotatingFileHandler` (stdlib, 20 MB × 5 archivos). Pensado para análisis posterior de threshold/precisión, no para debugging en vivo.

**`stream_server.py`** (~130 líneas)
Servidor HTTP MJPEG daemon para stream mode (`NX_STREAM_ENABLED=true`). Solo per-cámara, sin tiler. Expone:
- `/stream/<camera_id>` — MJPEG live con bboxes/labels dibujados por el probe
- `/viewer/<camera_id>` — HTML mínimo con `<img>` + JS de reconexión automática (reintenta cada 2 s si el stream cae)

Misma arquitectura de dos threads: `_encode_loop` (drena queues, encoda JPEG) + HTTP server (multipart/x-mixed-replace a 25 fps). Zero overhead cuando `NX_STREAM_ENABLED=false`.

**`config_loader.py`** (~280 líneas)
Carga y fusiona configuración desde 5 fuentes (prioridad: env vars > `/etc/nx_*` > `config.yaml` > `.env` > defaults). Define 11 paquetes predefinidos (`PACKAGE_DEFINITIONS`), 3 capacidades válidas (`people_counting`, `age_gender`, `face_recognition`), límites de NVDEC, y genera URLs RTSP interpolando el patrón del DVR. Retorna un `ClientConfig` dataclass. Campos configurables desde `config.yaml` (con defaults): `pgie_batch_size=0`, `pgie_interval=-1`, `sgie_interval=-1`, `reid_gallery_size=10`. Campos de umbral PGIE sobreescribibles por cliente: `pgie_topk`, `pgie_nms_iou_threshold`, `pgie_pre_cluster_threshold` (todos con default -1 = usar valor del archivo). Si alguno está seteado, `app.py` genera un config temporal en `/tmp/` vía `_apply_pgie_overrides()`. **Importante:** reescribe rutas relativas como absolutas para evitar `Cannot access ONNX file '/tmp/...'` (ver ErrorHistory.md 2026-05-28).

**`common/bus_call.py`**
Handler genérico de mensajes del bus GStreamer (EOS, WARNING, ERROR). Estándar de ejemplos NVIDIA DeepStream.

**`common/FPS.py`**
Medidor de FPS con ventana de 5 segundos. Clase `GETFPS` con `get_fps()` y `print_data()`.

**`face_recognizer.py`** (~330 líneas)
Worker thread para reconocimiento facial. Carga `known_faces.json` (dos formatos: legacy nombre-clave, nuevo UUID-clave `{"uuid": {"name": "...", "embeddings": [...]}}`) en `_load_db()`. Para cada crop de rostro: extrae embedding 512-dim con InsightFace buffalo_l, calcula similitud coseno contra la DB. Threshold: ≥ 0.50. `_locked`/`_votes` están indexados por `global_id` (no `track_id`) — `_votes` es un `deque(maxlen=FACE_VOTES_REQUIRED=3)` por `global_id` que se sigue alimentando aunque ya haya un candado, para poder corregirlo si la mayoría cambia (protección contra que ReID/OSNet confunda a dos empleados con uniformes parecidos).
- `enqueue(face_crop, identity_key, frame_num, camera_id)` / `get_result(identity_key)`: `identity_key` es el `global_id`, no el `track_id` — renombrado en esta migración.
- `forget(global_id)`: limpia `_locked`/`_votes` para un `global_id` que `ReIdManager` ya expiró — llamado desde `probes.py::_handle_appearance_reid()` con los `expired_ids` que retorna `match_or_create()`. Sin esto, ambos dicts crecerían indefinidamente.
- `sync_from_backend(action, employee_id)`: llama GET `/api/employees/embeddings`, reescribe JSON a disco y llama `reload()` — bloqueante, ejecutar en hilo separado
- `reload(raw_db)`: reemplaza `_db` y `_uuid_to_name` en memoria; resetea `_locked` y `_votes` para evitar votos stale
- `get_display_name(uuid_str)`: retorna nombre legible para OSD (de `_uuid_to_name`)
- En `start()`: lanza `sync_from_backend()` en hilo separado si `api_base_url` está configurado

**`jetson_sync_client.py`** (~100 líneas)
Worker Socket.IO que mantiene conexión persistente al namespace `/jetson` del backend. Autentica con `X-API-Key` en el dict `auth` de Socket.IO. En `face_update` recibido: despacha `sync_callback(action, employee_id)` en hilo separado (sin bloquear el event loop). También dispara un sync en `on_connect` para sincronizar si el Jetson estuvo offline. Reconexión automática gestionada por python-socketio.


**`reid_manager.py`** (~245 líneas)
Gestor local de identidades cross-cámara. Mantiene un dict en memoria (`global_id → _Entry`) con **galería de embeddings**, timestamps y cámara actual. Cada `global_id` almacena hasta `GALLERY_MAX_SIZE=10` vectores que representan distintos ángulos/poses. El matching usa `max(query @ emb_i for emb_i in gallery)`. API pública:
- `match_or_create(embedding, camera_id, threshold=None, create=True, track_id=None)` — retorna `(global_id, event_type, prev_camera_id, expired_ids)`. `expired_ids` son los `global_id`s que `_expire_stale()` acaba de olvidar en esta llamada — el caller (`probes.py`) los usa para limpiar `FaceRecognizer.forget()` y `_employee_by_global_id`. `track_id` es opcional y solo alimenta el log CSV, no afecta el matching.
- `update_embedding(global_id, embedding, track_id=None)` — añade a la galería con diversity check (sim < 0.85)
- `flush()` — persiste a disco al apagar el pipeline
Persiste la DB en `deploy/reid_db.json` cada 30 s. Constantes: `SIMILARITY_THRESHOLD=0.68`, `GALLERY_MAX_SIZE=10`, `PRESENCE_WINDOW_S=300`, `REID_TTL_S=3600`.
`__init__` acepta `csv_log_dir` opcional — si se pasa, activa el log CSV siempre-activo en `<csv_log_dir>/osnet_reid.csv` (ver sección "Re-ID entre Cámaras" arriba para el detalle de columnas). `probes.py::init_workers()` le pasa el mismo directorio `clients/<cliente>/logs/` que usa `face_recognition.csv`.

**`ws_client.py`** (~150 líneas)
WebSocket persistente hacia el backend. Envía snapshots de posiciones normalizadas (`global_id`, `x_norm`, `y_norm`, `employee_id`, `face_confirmed`) cada 1 segundo (`POSITION_SEND_INTERVAL` en `probes.py`) por cámara — usados por el backend para generar heatmaps y, si `employee_id` no es nulo, asistencia de empleados. Reconexión automática con backoff exponencial (1s → 30s). Silencioso si no hay conexión.

---

### `deploy/tools/` — Scripts utilitarios

**`setup.sh`** (~629 líneas)
**El único comando que ejecuta el técnico instalador.** Realiza la configuración completa del Jetson desde cero:
- Instala Docker CE, Tailscale, x11vnc
- Configura auto-login GDM y SSH con clave pública
- Escanea la red con nmap para encontrar DVRs en puerto 554
- Ejecuta `identify_dvr.py` para detectar marca y patrón RTSP
- Ejecuta `probe_cameras.py` para encontrar canales con cámaras activas
- Descarga modelos públicos (OSNet) vía `download_models.py`
- Escribe `/etc/nx_client`, `/etc/nx_sector`, `/etc/nx_pipeline`, `/etc/nx_dvr_ip`
- Construye la imagen Docker (`docker build`)
- Lanza el pipeline (`docker compose up -d`)

Flags principales: `--client`, `--package`, `--authkey`, `--api-key`, `--dvr-user`, `--dvr-pass`, `--stream-type {main|sub}`, `--entry-exit-channels`, `--no-vnc`, `--no-docker`.

`--dvr-user` / `--dvr-pass` crean `clients/<client>/.env` automáticamente (antes era un paso manual previo al script). Si no se pasan y el archivo no existe, el setup advierte y omite la detección automática del DVR.

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

**`dvr_watchdog.sh`** (~140 líneas)
Script daemon instalado por `setup.sh` como servicio systemd `nx-dvr-watchdog` en el host del Jetson (fuera de Docker). Cada 10 s (`POLL_INTERVAL`) verifica conectividad TCP directa a la IP configurada en `/etc/nx_dvr_ip` sobre el puerto RTSP del cliente (`get_dvr_port()` lee `dvr_port` de `clients/<cliente>/config.yaml`, default 554 — mismo default que `config_loader.py`), usando `/dev/tcp` de bash (sin dependencias nuevas). Tras `FAILURE_THRESHOLD=3` chequeos consecutivos fallidos (debounce contra blips de red), ejecuta `nmap -p <puerto> <subred>/24 --open -T4` para encontrar el DVR en su nueva IP. Si la encuentra: escribe la nueva IP en `/etc/nx_dvr_ip` y corre `docker restart` sobre el container detectado (`get_container()`, tolera el prefijo de proyecto de Docker Compose). Si no encuentra nada: espera 300 s (`COOLDOWN`) y reintenta. Al instalar, `setup.sh` sustituye el placeholder `@@WORK_DIR@@` con la ruta real del repo. Logs: `journalctl -u nx-dvr-watchdog -f`.
- **Diseño anterior (abandonado):** parseaba `docker logs` buscando `RTSP 'source-N' failed` y comparaba el conteo contra `len(channels)` de `config.yaml`. Se abandonó porque ese conteo no coincidía con los streams reales cuando el cliente tenía `external_channels` configurados (`app.py` los excluye de `active_channels`) — el watchdog nunca disparaba aunque todas las cámaras reales fallaran. Ver `ErrorHistory.md` 2026-07-01.

---

### `deploy/models/` — Modelos TensorRT

**`peoplenet_vpruned_quantized_decrypted_v2.3.4/`**
- `nvinfer_config.txt`: Config DeepStream para PGIE. `gie-unique-id=1`, INT8, batch=4, interval=4, 3 clases (person, bag, face). `[class-attrs-all]` fija topk/nms-iou-threshold/pre-cluster-threshold para las 3 clases (overridable por cliente vía `config.yaml`, ver `_apply_pgie_overrides()` en `app.py`). `[class-attrs-2]` (face) agrega `detected-min-w=64`/`detected-min-h=64` — descarta caras menores a 64×64px antes de que lleguen a `FaceRecognizer`, evitando gastar CPU en crops sin suficiente resolución para un embedding ArcFace confiable. Valor elegido por ausencia de una recomendación oficial: ni NVIDIA (PeopleNet documenta 10×10px @1920×1080 como piso de *anotación de entrenamiento*, no como mínimo para reconocimiento) ni InsightFace publican un mínimo; 64×64 es el punto más bajo con evidencia real (estudio de degradación de InsightFace por resolución) donde el modelo sigue dando resultados usables. Asume resolución nativa de cámara — en sub-stream el umbral es proporcionalmente más estricto (limitación conocida y diferida, igual que el filtro de tamaño de persona en OSNet). `_apply_pgie_overrides()` solo reescribe `[class-attrs-all]`, así que esto no interfiere con los overrides de topk/nms/pre-cluster-threshold por cliente.
- `resnet34_peoplenet_int8.onnx`: Modelo cuantizado INT8.
- `*.engine`: Engine TensorRT compilado por dispositivo (se regenera automáticamente).

**`resnet_age_gender_FB2/`**
- `config_infer.txt`: Config para SGIE de edad/género. `gie-unique-id=2`, FP16, opera sobre `class-ids=0` (personas) del PGIE.
- `custom_softmax_parser.so`: Plugin C++ compilado por `docker-entrypoint.sh` para parsear salida softmax del clasificador.

**`osnet/`**
- `config_infer_sgie_osnet.txt`: Config para SGIE OSNet appearance. `gie-unique-id=3`, FP32, `process-mode=2`, opera sobre `class-ids=0` (personas) del PGIE. `output-tensor-meta=1` expone el tensor para lectura en el probe. `model-engine-file` debe coincidir exactamente con el nombre que DeepStream genera al compilar (`<onnx>_b<batch-size>_gpu<gpu-id>_<network-mode>.engine`) — si no coincide, el engine nunca se cachea y se recompila (~2 min) en cada restart del container, no solo la primera vez. Ver `ErrorHistory.md` 2026-07-04.
- `osnet_x1_0_market1501.onnx`: Modelo descargado por `setup.sh` vía `download_models.py --reid`. No está en git.
- `osnet_x1_0_market1501.onnx_b8_gpu0_fp32.engine`: Engine TRT compilado por DeepStream (batch-size=8, gpu-id=0, FP32). Se genera en el primer arranque y se reutiliza en los siguientes mientras `batch-size`, `gpu-id` y `network-mode` no cambien. No está en git.

**`facedetect_ir/`** — ⚠️ No usado actualmente. El SGIE FaceDetectIR fue eliminado; la detección de rostros usa PeopleNet class_id=2 directamente. El directorio y su `config_infer.txt` se conservan como referencia pero no se cargan en `app.py`.

---

### `deploy/clients/` — Configuración por cliente

**`clients/<nombre>/config.yaml`**
Config no-sensible del cliente: nombre, puerto DVR, patrón RTSP, canales activos, paquete, tipo de stream (main/sub), tracker (nvdcf/iou), canales de entrada/salida.

**`clients/<nombre>/.env`**
Credenciales DVR (`DVR_USER`, `DVR_PASS`). **Gitignoreado.** Se genera en `setup.sh`.

**`.env`** (raíz de deploy)
Credenciales del backend: `API_BASE_URL`, `API_KEY`, `WS_BASE_URL`. **Gitignoreado.**

---

### `deploy/` — Archivos de orquestación Docker

**`docker-compose.yml`**
Dos servicios: `deepstream` (pipeline principal, puerto 8080 expuesto), `db` (TimescaleDB PostgreSQL 16, puerto 5432). Monta pipelines, modelos, clientes y tools.

**`docker-compose.stream.yml`**
Override mínimo para stream mode. Solo cargado desde `stream.sh`. Inyecta `NX_STREAM_ENABLED: "true"` al servicio `deepstream`. No agrega containers extra.

**`stream.sh`**
Script para activar/desactivar stream mode. `./stream.sh` reinicia deepstream con `NX_STREAM_ENABLED=true` e imprime las URLs `/viewer/<camera_id>` por cámara activa (Tailscale > IP local). `./stream.sh stop` vuelve a producción normal. `Ctrl+C` también restaura producción.

**`Dockerfile.jetson`**
Imagen ARM64 basada en `nvcr.io/nvidia/deepstream:7.1-samples-multiarch`. Instala pyds 1.1.11, onnxruntime-gpu para aarch64, insightface ≥ 0.7.3.

**`docker-entrypoint.sh`**
Se ejecuta al iniciar el contenedor: (1) compila `custom_softmax_parser.so` para el SGIE de edad/género, (2) parchea el ONNX de PeopleNet para batch dinámico, (3) pre-descarga InsightFace buffalo_l si `face_recognition` está en el pipeline, (4) elimina engines stale si el ONNX fue modificado. Luego hace `exec "$@"` para arrancar el pipeline directamente.

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
| `WS_BASE_URL` | `.env` | URL WebSocket para telemetría de posiciones / heatmaps |
| `NX_STREAM_ENABLED` | `docker-compose.stream.yml` | Activa stream mode (MJPEG con bboxes en :8080). Inyectado por `stream.sh`. |
| `NO_COLOR` | entorno del operador | Si `1`, desactiva códigos ANSI en los logs de `_slog` (útil para `grep` en `docker logs`). Default: `0`. |

---

## Notas de Rendimiento (Jetson Orin Nano)

- Máximo recomendado: 6 streams main (1920×1080) o 16 streams sub (960×544)
- `network-mode=1` (INT8) para PeopleNet; `network-mode=2` (FP16) si falla calibración INT8
- `classifier-async-mode=1` en SGIEs para no bloquear el pipeline
- Workers Python usan CPU + ONNX Runtime; no compiten con TensorRT por CUDA
- Los engines `.engine` se reconstruyen automáticamente al primer run por dispositivo (~5 min/modelo)
