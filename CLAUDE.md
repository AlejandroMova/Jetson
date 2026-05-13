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
│   │   ├── probes.py           # Probe GStreamer + handlers + API client
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
│   ├── docker-compose.yml
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

```
DVR (RTSP) → rtspsrc → h264/h265parse → nvv4l2decoder
  → nvstreammux → nvinfer (PeopleNet PGIE, gie-id=1)
  → nvtracker → [SGIEs opcionales por paquete]
  → nvmultistreamtiler → nvdsosd → appsink (MJPEG HTTP :8080)
```

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
Genera un vector de apariencia por persona para que el backend pueda reconocerla cuando reaparece en otra cámara, evitando contar la misma persona dos veces en un espacio con múltiples cámaras.
- **Modelo:** OSNet-x0.25 ONNX — genera embeddings 512-dim L2-normalizados
- **Worker:** `AppearanceWorker` (Python thread, no bloquea el pipeline)
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
- [ ] Regla 9: ¿Necesita actualizar CLAUDE.md? → [sí/no, qué sección]
- [ ] Regla 10: ¿Hay errores que registrar en ErrorHistory.md? → [sí/no]
- [ ] Regla 11: ¿Hay mejoras futuras que registrar en Future.md? → [sí/no]
```

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

### 2. Siempre revisar README.md antes de hacer cambios

`README.md` es la fuente de verdad del proyecto:
- Define los paquetes y sus capacidades
- Documenta los patrones RTSP por marca de DVR
- Describe el flujo de instalación y actualización
- Explica el esquema de config y variables de entorno

Si un cambio afecta algo documentado en `README.md`, actualizar el README también.

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

### 9. Mantener este archivo actualizado después de cada cambio significativo

Cada vez que se realice un cambio que altere la arquitectura, se agregue un nuevo archivo, se modifique el stack tecnológico, o cambie el comportamiento de algún componente importante:
- Revisar si la sección de **Descripción Detallada de Archivos** refleja el estado actual
- Revisar si la sección de **Stack Tecnológico** necesita actualizarse (nueva librería, nuevo modelo)
- Revisar si la sección de **Arquitectura del Pipeline** sigue siendo precisa
- Actualizar la tabla de paquetes/capacidades si se agrega o modifica alguna capacidad
- Si se agrega un archivo nuevo relevante, documentarlo aquí

Este archivo es la guía de trabajo de Claude en este proyecto. Si no se mantiene actualizado, las instrucciones pierden valor.

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

---

## Descripción Detallada de Archivos

### `deploy/pipelines/` — Núcleo del pipeline

**`app.py`** (~430 líneas)
Pipeline de producción. Construye el grafo GStreamer dinámicamente según las cámaras y capacidades activas. Conecta fuentes RTSP del DVR (H.264 o H.265, detección automática), configura PeopleNet como PGIE, añade SGIEs opcionales según el paquete, y expone un servidor MJPEG en `:8080` para visualización. Maneja el ciclo de vida de workers async (start/stop). Lee configuración a través de `config_loader.py`.

**`app_video_testing.py`** (~240 líneas)
Igual que `app.py` pero para archivos MP4 locales. Usa `filesrc + qtdemux` en lugar de `rtspsrc`. Útil para desarrollo y QA sin DVR físico. Sale con RTSP output en lugar de MJPEG. Acepta `--capabilities` y `--client` por CLI.

**`probes.py`** (~1300 líneas)
El motor central de analytics. Se conecta al OSD sink-pad del pipeline y se ejecuta en cada frame. Contiene:
- `NxApiClient`: cola async → thread worker → HTTP POST al backend (fire-and-forget, no bloquea)
- `_AgeGenderHandler`: acumula 10 votes del SGIE antes de confirmar clasificación
- `_FallDetectionHandler`: despacha crops al `PoseWorker`, aplica 3 reglas geométricas
- `_FaceRecognitionHandler`: cruza detecciones del SGIE FaceDetectIR con el `FaceRecognizer`
- `osd_sink_pad_buffer_probe`: función principal que procesa cada batch de frames, gestiona el ciclo de vida de tracks (entry/exit), despacha handlers, acumula crops y analytics
- Stubs `_EppHandler`, `_FireSmokeHandler`, `_LicensePlateHandler` (pendiente integración)

**`config_loader.py`** (~326 líneas)
Carga y fusiona configuración desde 5 fuentes (prioridad: env vars > `/etc/nx_*` > `config.yaml` > `.env` > defaults). Define los 15 paquetes predefinidos (`PACKAGE_DEFINITIONS`), capacidades válidas, límites de NVDEC, y genera URLs RTSP interpolando el patrón del DVR. Retorna un `ClientConfig` dataclass.

**`common/bus_call.py`**
Handler genérico de mensajes del bus GStreamer (EOS, WARNING, ERROR). Estándar de ejemplos NVIDIA DeepStream.

**`common/FPS.py`**
Medidor de FPS con ventana de 5 segundos. Clase `GETFPS` con `get_fps()` y `print_data()`.

**`face_recognizer.py`** (~162 líneas)
Worker thread para reconocimiento facial. Carga `known_faces.json` (nombre → lista de embeddings). Para cada crop de rostro: extrae embedding 512-dim con InsightFace buffalo_l, calcula similitud coseno contra la DB, acumula 3 votos antes de bloquear identidad. Threshold: ≥ 0.50.

**`pose_worker.py`** (~213 líneas)
Worker thread para detección de caídas. Ejecuta MoveNet Lightning ONNX (entrada 192×192). Aplica 3 reglas: ángulo del torso > 45°, bbox más ancho que alto, caderas cerca de los tobillos. Caída si ≥ 2/3 reglas. Cooldown de 4 segundos por `track_id`.

**`appearance_worker.py`** (~139 líneas)
Worker thread para re-ID entre cámaras. Ejecuta OSNet-x0.25 ONNX (entrada 128×256), genera vector 512-dim L2-normalizado por persona. Se encola cada 15 frames hasta tener resultado. El backend usa similitud coseno ≥ 0.65 para cruzar personas entre cámaras.

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

### `deploy/` — Archivos de orquestación Docker

**`docker-compose.yml`**
Tres servicios: `deepstream` (pipeline principal, puerto 8080 MJPEG), `db` (TimescaleDB PostgreSQL 16, puerto 5432), `redis` (Redis 7, puerto 6379). Monta los directorios de pipelines, modelos, clientes y tools.

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
