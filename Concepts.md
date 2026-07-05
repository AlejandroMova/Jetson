# Concepts.md — NX Computing AI | Guía de lectura del código

Este documento explica cómo funciona el sistema a nivel conceptual, con links directos al código.
El objetivo es que puedas entender el flujo sin leer línea por línea — solo cuando necesites un
detalle concreto sigues el link.

---

## 1. El flujo de datos general

El video pasa por estos pasos en orden:

| Paso | Elemento | Qué hace |
|------|----------|----------|
| 1 | DVR → RTSP | Las cámaras envían video H.264/H.265 por red |
| 2 | `nvv4l2decoder` | Decodifica el video en GPU (muy eficiente) |
| 3 | `PeopleNet PGIE` | Detecta personas (y bolsas/caras) en cada frame — corre en GPU |
| 4 | `NvTracker` | Asigna un `track_id` a cada persona y lo mantiene entre frames |
| 5 | SGIEs opcionales | Clasifican sobre cada persona: edad/género (`gie-id=2`). Las caras las detecta PeopleNet directamente (class 2) |
| 6 | `nvvideoconvert` | Convierte el frame a formato RGBA para que Python pueda leerlo |
| 7 | **Probe** | Aquí entra todo el código Python de analytics |

En producción el Probe es único y manda los resultados al backend.
En QA mode hay dos probes: Probe A (analytics, antes de reducir resolución) y Probe B (solo dibuja overlays para visualización).

**Archivos clave:**
- Pipeline de producción: [deploy/pipelines/app.py](deploy/pipelines/app.py)
- Pipeline con videos locales: [deploy/pipelines/app_video_testing.py](deploy/pipelines/app_video_testing.py)
- Toda la lógica de analytics: [deploy/pipelines/probes.py](deploy/pipelines/probes.py)

---

## 2. El ciclo de vida de un track

Hay **dos ciclos de vida paralelos** que no hay que confundir:

| | Track local (`_active_tracks`) | Identidad global (`ReIdManager._db`) |
|---|---|---|
| Identificador | `(pad_index, track_id)` — local por cámara | `global_id` — estable entre cámaras y reinicios |
| Vive en | RAM del probe (solo mientras la persona está en frame) | RAM + `reid_db.json` en disco |
| Muere cuando | El tracker pierde a la persona (sale del encuadre) | Pasan **1 hora** sin verla (`REID_TTL_S = 3600 s`) |

El `track_id` es temporal y local — DeepStream lo reasigna cada vez que la persona
entra al encuadre. El `global_id` es persistente — sobrevive reinicios del pipeline.

### Nacimiento del track local
Cuando PeopleNet detecta una persona, el tracker le asigna un `track_id`.
En el probe, si ese `(pad_index, track_id)` no existe en `_active_tracks`, se crea un `_TrackState`.

```
_active_tracks[(pad_index, track_id)] = _TrackState(...)
```

El par `(pad_index, track_id)` es clave compuesta porque DeepStream asigna track_ids
**locales por cámara** — dos cámaras pueden tener un `track_id=5` distinto.

- Definición de `_TrackState`: [probes.py línea 357](deploy/pipelines/probes.py#L357)
- Dónde se crea: [probes.py línea 1769](deploy/pipelines/probes.py#L1769)
- Dict global: [probes.py línea 1370](deploy/pipelines/probes.py#L1370)

### Vida
Mientras la persona está en cámara, cada frame el probe:
1. Actualiza `last_frame` en el `_TrackState`
2. Despacha los handlers activos (edad/género, caídas, cara)
3. Acumula conteos para el analytics snapshot

### Muerte del track local (track lost)
Cuando el tracker deja de ver una persona, su `track_id` desaparece de los metadatos.
El probe detecta esto comparando los tracks activos con los que llegaron en el frame actual.
Los que no aparecen → `_expire_lost_tracks()` los elimina, emite `person_exit`, y limpia
`_active_tracks`. Pero **la entrada en `ReIdManager._db` sigue viva** durante hasta 1 hora.

- Función de expiración: [probes.py línea 1459](deploy/pipelines/probes.py#L1459)

### Reentrada — misma cámara
Si la misma persona sale y vuelve a entrar al encuadre de **la misma cámara**:
- DeepStream le asigna un nuevo `track_id` → nuevo `_TrackState` en `_active_tracks`
- `AppearanceWorker` genera un nuevo embedding
- `ReIdManager.match_or_create()` lo reconoce por similitud coseno y devuelve el mismo `global_id`
- Como `prev_camera == camera_id`, el evento `channel_change` se **demota a `person_return`**
  para no emitir un cambio de cámara espurio cuando en realidad la persona nunca cambió de cámara

### Eventos emitidos en el ciclo de vida
| Situación | Evento al backend |
|-----------|------------------|
| Primera vez que se ve (sin match en DB) | `person_entry` con `entry_type: "new"` |
| Reentrada a cualquier cámara, ausente > 5 min | `person_entry` con `entry_type: "return"` |
| Cambio a otra cámara, ausente ≤ 5 min | `person_channel_change` |
| Cada 60 s mientras está en cámara | `analytics_snapshot` |
| Track desaparece del encuadre | `person_exit` (con tiempo de permanencia) |

---

## 3. El patrón worker (async queue)

**El problema:** Correr modelos de IA (OSNet, MoveNet, InsightFace) dentro del probe GStreamer
bloquearía el pipeline y bajaría los FPS a cero.

**La solución:** Todos los workers usan el mismo patrón de dos pasos separados en el tiempo:

**Frame N — el probe encola el trabajo:**
1. El probe recorta la bbox de la persona del frame (`crop`)
2. Llama `worker.enqueue(crop, track_id)` — esto solo mete el crop en una cola, O(1), sin esperar nada
3. El probe sigue procesando el siguiente objeto/frame inmediatamente

**Mientras tanto — el worker corre en paralelo:**
4. El hilo worker saca el crop de la cola
5. Corre el modelo ONNX (MoveNet, OSNet, etc.) — esto sí tarda, pero en otro hilo
6. Guarda el resultado en `_results[track_id]`

**Frame N+1 (o N+5, el que sea) — el probe lee el resultado:**
7. El probe llama `worker.get_result(track_id)`
8. Si el worker ya terminó → devuelve `PoseResult` con el resultado
9. Si el worker aún no terminó → devuelve `None`, y el probe simplemente espera al frame siguiente

El resultado puede llegar varios frames después de que se encola — eso está bien, los analytics
no necesitan respuesta inmediata frame a frame.

**Excepción — `FaceRecognizer` usa `global_id`, no `track_id`:** el resto de los workers de esta
sección se indexan por `track_id`, pero `FaceRecognizer.enqueue`/`get_result` se indexan por el
`global_id` de ReID — ver la sección "Handler: Reconocimiento Facial" más abajo para el porqué.

**Los tres workers:**

| Worker | Modelo | Propósito | Archivo |
|--------|--------|-----------|---------|
| `AppearanceWorker` | OSNet-x0.25 ONNX | Embedding 512-dim por persona para re-ID | [appearance_worker.py](deploy/pipelines/appearance_worker.py) |
| `PoseWorker` | MoveNet Lightning ONNX | Detección de caídas (17 keypoints) | [pose_worker.py](deploy/pipelines/pose_worker.py) |
| `FaceRecognizer` | InsightFace ArcFace | Identifica personas conocidas por cara | [face_recognizer.py](deploy/pipelines/face_recognizer.py) |

**Por qué los modelos se cargan en `start()` y no en `__init__()`:**
TensorRT inicializa su contexto CUDA cuando el pipeline hace `set_state(PLAYING)`.
Si ONNX Runtime se carga antes, hay conflictos de contexto CUDA. Por eso todos los workers
cargan el modelo en `start()`, que se llama después de arrancar el pipeline.

Ver ejemplo: [appearance_worker.py línea 64](deploy/pipelines/appearance_worker.py#L64)

---

## 4. Cómo se envían eventos al backend: NxApiClient

Todos los eventos van por `NxApiClient` — un cliente HTTP asíncrono con cola interna.

Cuando el probe quiere enviar un evento, hace esto:
1. Llama `api_client.post_fall_detected(camera_id, track_id, bbox, ...)` — O(1), regresa inmediatamente
2. Internamente eso llama `enqueue()`, que mete el payload en una `queue.Queue` (capacidad 512 items)
3. El probe sigue con el siguiente frame — nunca espera la red

En paralelo, el hilo worker de `NxApiClient`:
4. Saca el item de la cola
5. Hace el `POST /api/events` al backend (timeout 5 s)
6. Si el backend retorna 2xx: invoca el callback registrado para ese endpoint (si existe)
7. Si el backend está caído, lo logea y descarta — el pipeline nunca falla por esto

El probe **nunca espera** la respuesta HTTP. Si el backend está caído, los eventos se descartan
silenciosamente (la cola tiene límite de 512 items).

**Callbacks de éxito por endpoint:** `register_success_callback(endpoint, cb)` permite registrar una función
que se invoca desde el worker thread cuando el backend confirma 2xx. Actualmente se usa para el reference frame:
cuando el backend confirma la recepción, se almacena el frame como baseline para detectar cambios futuros.

- Clase completa: [probes.py línea 406](deploy/pipelines/probes.py#L406)
- Todos los métodos `post_*`: [probes.py desde línea 520](deploy/pipelines/probes.py#L520)

**Estructura base de todo evento:**
```json
{
  "event_id": "uuid",
  "type": "person_entry",
  "sector": "comercio",
  "jetson_id": "jetson-nx-001",
  "camera_id": "jetson-nx-001-ch01",
  "timestamp": "2025-05-24T10:00:00Z",
  "severity": "info"
}
```

En QA mode, cada evento también se publica a Redis (`nx:qa:apicalls`) para que Streamlit
lo muestre en el log de API calls.

---

## 5. Los handlers — cómo funciona cada detección

Los handlers son clases que procesan cada persona detectada en el probe.
Se despachan desde el probe para cada objeto con `class_id=0` (persona).

```python
for each person in frame:
    for handler in active_handlers:
        result = handler.process(obj_meta, frame_num, frame_np)
        if result:
            draw OSD label, emit API event
```

### Handler: Edad y Género

Usa el **SGIE ResNet-18** (`gie-unique-id=2`) que ya corrió en GPU sobre cada persona.
El handler lee el output del SGIE desde los metadatos de DeepStream — sin modelo extra en Python.

**El truco del sistema de votos:**
El SGIE puede equivocarse en frames individuales (persona girada, borrosa, lejos).
Por eso se acumulan **10 votos** antes de fijar el resultado. Una vez fijo, no cambia.

```
frame 1: "male_adult" → votos: [male_adult]
frame 2: "male_young" → votos: [male_adult, male_young]
...
frame 10: suficientes votos → ganador por mayoría → "male_adult" fijo
```

- Clase: [probes.py línea 850](deploy/pipelines/probes.py#L850)
- Constante `VOTES_REQUIRED=10`: [probes.py línea 195](deploy/pipelines/probes.py#L195)

### Handler: Detección de Caídas

No usa SGIE — usa `PoseWorker` (MoveNet ONNX en Python).

Cada `POSE_SAMPLE_INTERVAL=10` frames, el handler:
1. Recorta la bbox de la persona del frame completo
2. Encola el crop al `PoseWorker` (no bloqueante)
3. Lee el resultado del frame anterior

MoveNet devuelve **17 keypoints COCO** (hombros, caderas, tobillos, etc.).
El `PoseWorker` aplica 3 reglas geométricas:

| Regla | Cómo detecta caída |
|-------|--------------------|
| 1 | Ángulo del torso > 45° desde la vertical (torso horizontal) |
| 2 | Bbox más ancho que alto (persona acostada) |
| 3 | Caderas al mismo nivel que los tobillos (hips cerca del suelo) |

**Caída = 2 o más reglas se cumplen.** Cooldown de 4 s por persona para no repetir alertas.

- Handler: [probes.py línea 1002](deploy/pipelines/probes.py#L1002)
- Reglas geométricas: [pose_worker.py — método `_classify_fall`](deploy/pipelines/pose_worker.py#L176)
- Constante `POSE_SAMPLE_INTERVAL=10`: [probes.py línea 200](deploy/pipelines/probes.py#L200)

### Handler: Reconocimiento Facial

Usa **PeopleNet class_id=2** (caras, detectadas por el mismo PGIE) — no hay SGIE adicional para caras.
Luego `FaceRecognizer` (InsightFace ArcFace en Python) identifica a la persona.

**Indexado por `global_id`, no por `track_id`.** A diferencia de los otros workers (§3), que
reciben/devuelven resultados por `track_id`, `FaceRecognizer` usa el `global_id` de ReID como llave —
porque `track_id` se reinicia en cada cámara nueva, lo que obligaría a re-votar desde cero cada vez
que el empleado cambia de cámara. Con `global_id`, una identidad ya bloqueada viaja automáticamente
entre cámaras vía la continuidad de apariencia de `ReIdManager`, sin re-votar.

El probe recoge los objetos de cara directamente del PGIE y los pasa al handler:

```
PeopleNet emite un objeto con class_id=2 (cara)
  → _find_parent_track() → busca la persona (class_id=0) que contiene esa cara
  → busca _active_tracks[(pad_index, parent_track_id)].global_id
  → si global_id aún no está resuelto (ReID no ha corrido) → no procesa nada este frame, espera
  → recortar crop de cara del frame full-res
  → enqueue al FaceRecognizer worker, indexado por global_id
  → (próximo frame) get_result(global_id) → (uuid_str, similitud) o None
  → sistema de votos: ventana deslizante de 3 predicciones (deque), nunca se detiene aunque ya
    haya un candado — si la mayoría de la ventana cambia, se corrige el tag (salvaguarda contra
    que ReID/OSNet confunda a dos empleados con uniformes parecidos)
  → identidad confirmada → _employee_by_global_id[global_id] = uuid, y se marca
    _face_confirmed_this_cycle para ese global_id (consumido por _accumulate_positions)
```

**Ya no hay eventos discretos para comercio/industrial** (`employee_seen`/`employee_presence`/
`employee_exit` fueron eliminados). La identidad de empleado viaja en cambio dentro de
`positions_snapshot` (§6/`ws_client.py`), vía los campos `employee_id`/`face_confirmed` de cada
posición — ver `_accumulate_positions` en `probes.py` y `app/socket/positions.py` en el backend.
Solo hogar conserva un evento discreto, `unknown_person_alert`, porque es una alerta de intrusión,
no de asistencia — pero comparte el mismo gate: no se procesa ninguna cara (reconocida o no) hasta
que `global_id` esté resuelto, así que en un Jetson sin OSNet instalado tampoco dispara (limitación
aceptada y diferida, ver `CLAUDE.md`).

**`employee_id` es un UUID** asignado por el backend (de `employees.id`), tageado sobre el
`global_id` — el backend no necesita resolver por nombre, la tabla `employee_zone_intervals` tiene
FK directo.

**Limpieza de memoria:** cuando `ReIdManager._expire_stale()` olvida un `global_id` (1h sin verse),
`_handle_appearance_reid()` llama `FaceRecognizer.forget(global_id)` para limpiar también sus
diccionarios `_locked`/`_votes` — si no, crecerían indefinidamente (nada más los limpia por
`global_id`).

**Flujo de registro de empleados (automático desde plataforma):**
```
Admin activa empleado en plataforma
  → backend genera embedding ArcFace desde fotos
  → backend emite face_update en Socket.IO /jetson room
  → JetsonSyncClient recibe face_update
  → sync_from_backend() llama GET /api/employees/embeddings
  → known_faces.json actualizado (clave = UUID, valor = {name, embeddings})
  → FaceRecognizer.reload() — DB en memoria reemplazada sin reiniciar pipeline
```

**Formato `known_faces.json` (nuevo):**
```json
{
  "<employee-uuid>": { "name": "Juan Perez", "embeddings": [[...512 floats...]] }
}
```
Formato legacy (nombre-clave) sigue siendo compatible en lectura.

- Handler: [probes.py](deploy/pipelines/probes.py) — `_FaceRecognitionHandler`
- Worker: [face_recognizer.py](deploy/pipelines/face_recognizer.py) — `sync_from_backend`, `reload`, `get_display_name`
- Sync client: [jetson_sync_client.py](deploy/pipelines/jetson_sync_client.py) — Socket.IO /jetson namespace

---

## 6. Re-ID entre cámaras

**El problema:** Cuando una persona pasa de la cámara 1 a la cámara 2, el tracker le asigna
un nuevo `track_id` porque es un stream distinto. ¿Cómo sabemos que es la misma persona?

**La solución:** OSNet-x0.25 genera un vector de 512 números (embedding) que representa el
"aspecto visual" de la persona — ropa, silueta, color. Dos embeddings de la misma persona
tienen alta similitud coseno (≥ 0.55).

**Escenario A — persona nueva:**
1. Persona entra a cámara 1, AppearanceWorker genera embedding `[0.1, -0.3, 0.8, ...]`
2. `ReIdManager.match_or_create(embedding, "ch01")` busca en la DB — no hay nadie con similitud ≥ 0.55
3. Crea nuevo `global_id="a3f7c2"`, guarda el embedding en su galería
4. Emite `person_entry` con `entry_type: "new"`

**Escenario B — misma persona, otra cámara:**
1. Persona llega a cámara 2 (estuvo 5 min ausente), AppearanceWorker genera nuevo embedding
2. `ReIdManager.match_or_create(embedding, "ch02")` busca en DB → encuentra `"a3f7c2"` con similitud 0.72 → MATCH
3. Tiempo ausente > 5 min → emite `person_entry` con `entry_type: "return"`
4. Tiempo ausente ≤ 5 min → emite `person_channel_change`

**Escenario C — misma persona, misma cámara (salió y volvió):**
1. DeepStream le asigna un nuevo `track_id` porque el tracker la perdió
2. `ReIdManager.match_or_create(embedding, "ch01")` → MATCH con `"a3f7c2"`
3. Detecta que `prev_camera == camera_id` → el `channel_change` se **demota a `person_return`**
4. Evita emitir "cambio de cámara" cuando la persona nunca salió de la cámara

**La galería:** Cada persona almacena hasta 10 embeddings de distintos ángulos/poses.
El matching usa el máximo de similitud contra todos los ángulos de la galería — si cualquiera coincide,
la persona se reconoce aunque el ángulo actual sea distinto a los anteriores.

- Manager: [reid_manager.py](deploy/pipelines/reid_manager.py)
- Función de matching: [reid_manager.py — `_find_best_match`](deploy/pipelines/reid_manager.py#L213)
- `match_or_create`: [reid_manager.py — línea 134](deploy/pipelines/reid_manager.py#L134)
- Worker que genera embeddings: [appearance_worker.py](deploy/pipelines/appearance_worker.py)

**El entry diferido:** El embedding tarda ~1-2 frames en estar listo. El `person_entry`
se retrasa hasta que `_handle_appearance_reid()` tiene el embedding para incluir `global_id`.
Si después de `ENTRY_EMIT_DEADLINE_FRAMES=30` frames no hay embedding, se emite sin `global_id`.

- Lógica de entry diferido: [probes.py línea 1510](deploy/pipelines/probes.py#L1510)
- Deadline: [probes.py línea 222](deploy/pipelines/probes.py#L222)

---

## 7. El sistema de paquetes y capacidades

Cada cliente tiene un **paquete** contratado que define qué capacidades están activas.
Esto determina qué modelos se cargan, qué handlers corren, y qué eventos se emiten.

```
config.yaml:
  package: hogar_avanzado
      ↓
config_loader.py
      ↓
  pipeline: ["people_counting", "fall_detection"]
  sector: "hogar"
      ↓
app.py: solo carga el SGIE de fall_detection, no carga age_gender ni face_recognition
probes.py: solo instancia _FallDetectionHandler, no _AgeGenderHandler
```

- Definición de paquetes: [config_loader.py línea 67](deploy/pipelines/config_loader.py#L67)
- Capacidades válidas: [config_loader.py línea 54](deploy/pipelines/config_loader.py#L54)

En QA mode, los toggles de features en el dashboard de Streamlit sobrescriben esto en caliente
usando Redis hash `nx:qa:capabilities`.

---

## 8. QA mode — el pipeline dual

En producción el pipeline termina en `fakesink` — no hay output visual.
QA mode agrega un segundo path visual sin afectar el pipeline principal.

```
[Probe A] — se conecta ANTES del tiler, sobre frames RGBA full-res
  → corre todos los analytics (mismo código que producción)
  → escribe _track_labels[track_id] = {face_name, fall, age_gender}
  → si RecordingManager.is_recording: pasa frame full-res a push_camera_frame()

nvmultistreamtiler (640×360)

[Probe B] — se conecta DESPUÉS del tiler, sobre el frame compuesto reducido
  → solo dibuja bboxes + labels (lee _track_labels que escribió Probe A)
  → encola frame tileado en tiled_frame_queue
  → publica nx:qa:detections a Redis
```

`MjpegServer` consume `tiled_frame_queue`, encoda a JPEG, y sirve HTTP en `:8080`.
El dashboard Streamlit en `:8501` embebe el stream con `<img src="/stream/all">`.

- Probe A: [probes.py línea 1649](deploy/pipelines/probes.py#L1649)
- Probe B / overlay: [probes.py línea 1936](deploy/pipelines/probes.py#L1936)
- MjpegServer: [mjpeg_server.py](deploy/pipelines/mjpeg_server.py)
- RecordingManager: [recording_manager.py](deploy/pipelines/recording_manager.py)
- Dashboard Streamlit: [deploy/qa_app/streamlit_app.py](deploy/qa_app/streamlit_app.py)

---

## 9. Cómo se configura un cliente nuevo

El técnico instalador ejecuta un solo comando en el Jetson:

```bash
./setup.sh --client tienda_centro --package comercio_total --authkey <tailscale-key>
```

`setup.sh` hace todo automáticamente:
1. Instala Docker + Tailscale
2. Escanea la red buscando DVRs en puerto 554
3. `identify_dvr.py` → prueba patrones RTSP conocidos para detectar la marca del DVR
4. `probe_cameras.py` → sondea canales 1-16, filtra los que tienen señal
5. `download_models.py` → descarga MoveNet y OSNet desde URLs públicas
6. Escribe `/etc/nx_client`, `/etc/nx_pipeline`, `/etc/nx_sector` en el Jetson
7. Construye la imagen Docker y lanza el pipeline

- Setup: [deploy/setup.sh](deploy/setup.sh)
- Identificar DVR: [deploy/tools/identify_dvr.py](deploy/tools/identify_dvr.py)
- Sondear cámaras: [deploy/tools/probe_cameras.py](deploy/tools/probe_cameras.py)
- Descargar modelos: [deploy/tools/download_models.py](deploy/tools/download_models.py)

---

## 10. Redis — el bus de comunicación en QA mode

Redis solo se usa en QA mode (`NX_QA_ENABLED=true`). En producción no hay dependencia de Redis.

| Key / Canal | Quién escribe | Quién lee | Qué contiene |
|-------------|--------------|-----------|--------------|
| `nx:qa:detections` (pub/sub) | Probe B | Streamlit | Bboxes + labels del frame actual |
| `nx:qa:apicalls` (pub/sub) | NxApiClient | Streamlit | JSON de cada POST al backend |
| `nx:qa:status` | app.py al arrancar | Streamlit | Cliente, canales, capacidades, resoluciones |
| `nx:qa:pipeline_stats` | Probe A cada 5 s | Streamlit | FPS por cámara |
| `nx:qa:capabilities` | Streamlit (toggles) | Probe A/B | Qué handlers están activos |
| `nx:qa:recording_active` | RecordingManager | Streamlit | "1" si hay clip grabándose |
| `nx:qa:playback_video` | Streamlit | app.py | Path al video para modo playback |
| `nx:qa:config_overrides` | Streamlit editor | app.py al reiniciar | Variables de config editadas en dashboard |

---

## DVR IP auto-recovery (watchdog systemd)

El DVR del cliente usa DHCP, por lo que su IP puede cambiar tras un reinicio o renovación
del lease. Cuando eso pasa, todos los streams RTSP fallan y el pipeline queda corriendo sin video.

**Cómo funciona:**

```
Host Jetson (fuera de Docker)
  └── systemd: nx-dvr-watchdog
        │  cada 10 s: chequeo TCP directo a <IP configurada>:<puerto RTSP>
        │  (bash /dev/tcp, sin dependencias nuevas — no lee logs del pipeline)
        │
        ├─ Si responde → resetear contador de fallos, seguir monitoreando
        │
        └─ Si no responde 3 veces seguidas (debounce contra blips de red):
              nmap -p <puerto> <subred/24> --open -T4
              │
              ├─ nueva IP encontrada → escribe /etc/nx_dvr_ip → docker restart deepstream
              │                        pipeline se reconecta automáticamente
              │
              └─ no encontrada → espera 300 s → reintenta
```

El servicio corre en el **host** (no dentro de Docker), por lo que puede escribir
`/etc/nx_dvr_ip` directamente y hacer `docker restart` sin restricciones de bind mount.

**Por qué chequeo directo y no logs del pipeline:** la primera versión parseaba
`docker logs` buscando líneas `RTSP 'source-N' failed` y comparaba el conteo contra
los canales de `config.yaml`. Ese conteo no coincidía con los streams reales cuando el
cliente tenía `external_channels` configurados — el watchdog nunca disparaba aunque
todas las cámaras reales fallaran (ver `ErrorHistory.md` 2026-07-01). El chequeo TCP
directo a la IP configurada no depende de cuántas cámaras haya, ni del formato de logs
del pipeline, ni de que el container esté corriendo — verifica exactamente lo que
importa: si el DVR sigue respondiendo donde se supone que está.

**Instalación:** `setup.sh` copia `tools/dvr_watchdog.sh` a `/usr/local/bin/nx_dvr_watchdog.sh`
(sustituyendo `@@WORK_DIR@@` con la ruta real del repo) y crea el servicio systemd.

**Ver logs en campo:** `journalctl -u nx-dvr-watchdog -f`

**Archivos:**
- Script: [deploy/tools/dvr_watchdog.sh](deploy/tools/dvr_watchdog.sh)
- Instalación: [deploy/setup.sh](deploy/setup.sh) (sección 6d)

---

## Mapa mental rápido

```
¿Cómo llega un frame?          → app.py → GStreamer pipeline
¿Cómo se detecta algo?         → PeopleNet (PGIE) + SGIEs → metadatos en el probe
¿Cómo se procesa una persona?  → handlers en probes.py (_AgeGenderHandler, etc.)
¿Cómo se corre un modelo ONNX? → worker con enqueue/get_result pattern
¿Cómo se manda al backend?     → NxApiClient.post_xxx() → queue → HTTP POST
¿Cómo se identifica la misma persona entre cámaras? → ReIdManager + OSNet embeddings
¿Cómo se ve visualmente?       → QA mode: MjpegServer + Streamlit
¿Cómo se configura un cliente? → setup.sh (un solo comando)
¿Qué pasa si el DVR cambia de IP? → nx-dvr-watchdog chequea TCP directo, hace nmap, reinicia pipeline
```
