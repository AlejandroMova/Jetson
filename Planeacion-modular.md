# Plan: Arquitectura Modular de Modelos por Paquete NX Computing

## Context

El MVP actual tiene un pipeline fijo: PeopleNet (PGIE) → NvDCF Tracker → Age/Gender ResNet-18 (SGIE). El producto final requiere combinaciones distintas de modelos según el paquete contratado (Comercio, Industrial, Hogar × Básico/Avanzado/Total/Enterprise). El objetivo es diseñar la capa de inferencia DeepStream de forma que no haya 8 apps distintas, sino un sistema modular driven by config.

---

## Análisis por paquete (inferencia relevante)

### Comercio
| Plan | Modelos DeepStream necesarios |
|------|-------------------------------|
| Básico | PeopleNet (PGIE) — solo conteo |
| Avanzado | PeopleNet + tracker (hotspots/tiempo: backend) |
| Total | + SGIE Age/Gender (ya existe) |
| Enterprise | Igual Total + custom por cliente |

### Industrial
| Plan | Modelos DeepStream necesarios |
|------|-------------------------------|
| Básico | PeopleNet (PGIE) — conteo + actividad |
| Avanzado | + SGIE EPP/PPE (casco, chaleco sobre crops de persona) |
| Total | + PGIE LPD (license plate detector) + SGIE LPR (OCR de placa) + detector fuego/humo |
| Enterprise | Todo + custom |

### Hogar
| Plan | Modelos DeepStream necesarios |
|------|-------------------------------|
| Básico | PeopleNet (PGIE) — detección + alerta presencia (lógica en backend) |
| Avanzado | + detector de caídas (pose estimation o temporal CNN) |
| Total | + detector fuego/humo |

---

## Decisión arquitectónica recomendada: **Un solo app.py con carga modular de SGIEs**

### Por qué NO apps separadas
- Duplicación masiva de código (RTSP input, mux, OSD, MJPEG server idénticos en todos)
- Mantenimiento: un bug en el probe hay que arreglarlo en 8 archivos
- El 80% del pipeline es idéntico: solo los SGIEs cambian

### Por qué NO comentar/descomentar
- Error-prone, no escalable, no automatizable

### Por qué sí modular config-driven
- El campo `pipeline: [...]` ya existe en `ClientConfig` (config_loader.py)
- Solo hay que hacer que app.py lo interprete para añadir SGIEs dinámicamente
- probes.py ya tiene handlers separados por tipo de metadata

---

## Diseño de la solución

### 1. Pipeline capabilities (módulos)

Cada módulo = un nombre en `config.pipeline` → mapea a un config nvinfer + un handler en probes.py

```
people_counting      → Solo PeopleNet. Sin SGIE. (Básico Comercio/Industrial/Hogar)
age_gender           → SGIE ResNet-18 sobre class=0 de PeopleNet (ya existe)
epp_detection        → SGIE PPE sobre class=0 de PeopleNet (casco, chaleco, guantes)
fire_smoke           → Clasificador de frame (SGIE sobre todo el frame, no crops)
license_plate        → PGIE2 LPD sobre ROI + SGIE LPR (flujo separado de PeopleNet)
fall_detection       → SGIE pose/temporal sobre class=0 de PeopleNet
```

### 2. Cambios en app.py

`app.py` construye la cadena GStreamer así:
```python
SGIE_CONFIGS = {
    "age_gender":    "models/resnet_age_gender_FB2/config_infer.txt",
    "epp_detection": "models/epp/config_infer.txt",
    "fire_smoke":    "models/fire_smoke/config_infer.txt",
    "fall_detection":"models/fall_detection/config_infer.txt",
}

def build_pipeline(cfg: ClientConfig):
    # Siempre: PGIE PeopleNet + tracker
    # SGIEs: loop sobre cfg.pipeline
    sgies = []
    for cap in cfg.pipeline:
        if cap in SGIE_CONFIGS:
            sgie = Gst.ElementFactory.make("nvinfer", f"sgie_{cap}")
            sgie.set_property("config-file-path", SGIE_CONFIGS[cap])
            sgies.append(sgie)
    # license_plate es caso especial: añade un PGIE2 antes
```

### 3. Cambios en probes.py

El probe callback actual checa `gie-unique-id==2`. Se convierte en despachador:
```python
ACTIVE_HANDLERS = []  # poblado al arrancar según cfg.pipeline

def osd_sink_pad_buffer_probe(pad, info, u_data):
    # ...extrae metadata...
    for handler in ACTIVE_HANDLERS:
        handler.process(frame_meta, obj_meta, batch_meta)
```

Cada handler es una clase:
- `AgeGenderHandler` (ya existe, extraer del probe actual)
- `EppHandler` (nuevo)
- `FireSmokeHandler` (nuevo, opera a nivel frame no objeto)
- `FallDetectionHandler` (nuevo)

### 4. Client configs por paquete

```yaml
# clients/comercio_total/config.yaml
pipeline:
  - people_counting
  - age_gender

# clients/industrial_avanzado/config.yaml
pipeline:
  - people_counting
  - epp_detection

# clients/industrial_total/config.yaml
pipeline:
  - people_counting
  - epp_detection
  - license_plate
  - fire_smoke

# clients/hogar_avanzado/config.yaml
pipeline:
  - people_counting
  - fall_detection
```

---

## Modelos necesarios y estado

| Módulo | Modelo | Estado |
|--------|--------|--------|
| people_counting | PeopleNet v2.3.4 | ✅ Existe en repo |
| age_gender | ResNet-18 FB2 | ✅ Existe en repo |
| epp_detection | NVIDIA PPE Detector o custom SGIE | ❌ Por adquirir/entrenar |
| fire_smoke | FireNet (NVIDIA NGC) o clasificador frame | ❌ Por adquirir |
| license_plate | NVIDIA LPDNet + LPRNet (NGC) | ❌ Por adquirir |
| fall_detection | Pose-based (BodyPose + reglas) o temporal CNN | ❌ Por adquirir/entrenar |
| alerta_desconocidos | TBD — probablemente lógica de backend | ⏸ Pendiente de decisión |

### Fuentes de modelos NVIDIA (disponibles en NGC)
- **EPP/PPE**: `nvidia/tao/peoplesemsegnet` o custom con TAO Toolkit
- **LPD + LPR**: `nvidia/tao/lpdnet` + `nvidia/tao/lprnet` (soporte MX, Lat-AM plates)
- **Fire/Smoke**: Clasificadores disponibles en NGC o entrenar con TAO
- **Fall**: Más complejo — usar `nvidia/tao/bodyposenet` como SGIE + lógica en probe

---

## Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `deploy/pipelines/app.py` | Refactor `build_pipeline()` para SGIEs dinámicos |
| `deploy/pipelines/probes.py` | Extraer handlers a clases, despachar según `cfg.pipeline` |
| `deploy/pipelines/config_loader.py` | Validar que `pipeline` sea lista de capabilities conocidas |
| `deploy/clients/demo/config.yaml` | Ejemplo con `pipeline: [people_counting]` |
| Nuevos: `deploy/models/{epp,fire_smoke,license_plate,fall_detection}/config_infer.txt` | Configs nvinfer por módulo |

---

## Implementación — Todo de una vez (el código es modular, no hay riesgo)

### Cómo se declara el paquete (setup.sh → env var → validación en arranque)

```
setup.sh pregunta: "¿Qué paquete tiene este cliente?"
  → Escribe NX_PIPELINE=people_counting,age_gender a /etc/nx_pipeline
  → config_loader.py lo lee → resuelve capabilities activas
  → app.py valida en startup: todos los model files existen antes de crear el pipeline

# Para docker compose manual (sin setup.sh):
docker compose run -e NX_PIPELINE=people_counting,age_gender deepstream python3 pipelines/app.py
```

La validación en startup es clave: si falta un model file → error claro antes de iniciar GStreamer, no un crash a mitad del pipeline.

### Lo que se implementa ahora (scaffolding completo)

1. **config_loader.py** — añadir `/etc/nx_pipeline` como fuente; `VALID_CAPABILITIES` lista
2. **app.py** — `build_pipeline(cfg)` con `SGIE_CONFIGS` dict + carga dinámica + validación de archivos al arranque
3. **probes.py** — extraer `AgeGenderHandler`, añadir `HandlerRegistry`, dispatcher en probe callback
4. **setup.sh** — menú de selección de paquete, escribe `/etc/nx_pipeline`
5. **deploy/clients/** — configs por paquete (`comercio_basico`, `comercio_total`, etc.)
6. **Handlers stub** para módulos sin modelo aún (EPP, fire/smoke, LPR, fall) — registrados pero no activables hasta tener el ONNX

### Agregar un modelo NVIDIA NGC en el futuro (receta)

```bash
# 1. Descargar de NGC
ngc registry model download-version nvidia/tao/lpdnet:deployable_onnx_v1.0
# 2. Mover a deploy/models/license_plate/ con config_infer.txt + labels.txt
# 3. En app.py: añadir 1 línea a SGIE_CONFIGS
# 4. En probes.py: activar LicensePlateHandler (ya scaffoldeado)
# 5. En config_loader.py: añadir "license_plate" a VALID_CAPABILITIES
```
Total: ~3 archivos + 50 líneas de código por modelo nuevo. El pipeline existente no se toca.

---

## Verificación

```bash
# Comercio Básico: solo counting, NO debe crear SGIE
NX_PIPELINE=people_counting docker compose run deepstream python3 pipelines/app.py
# → Log: "No SGIEs loaded"

# Comercio Total: counting + age/gender (comportamiento actual)
NX_PIPELINE=people_counting,age_gender docker compose run deepstream python3 pipelines/app.py
# → Log: "Loaded SGIEs: age_gender (gie-id=2)"

# Después de integrar EPP:
NX_PIPELINE=people_counting,epp_detection docker compose run deepstream ...
# → OSD muestra bounding boxes con "EPP: sin casco" sobre crops de personas
```
