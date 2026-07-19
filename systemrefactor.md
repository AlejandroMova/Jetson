# systemrefactor.md — Diagnóstico y plan de refactor del conteo/ReID

**Fecha de la sesión:** 2026-07-19
**Cliente analizado:** Mova (cámaras `DEMOONE-ch02`, `DEMOONE-ch03` — canales 2 y 3)
**Datos usados:** `osnet_reid.csv` (15–18 jul, 499 filas) + `Testing/dataset_2026-07-15/` y
`dataset_2026-07-18/` (crops + `manifest.csv`)

> Este documento registra **qué se midió, qué se descartó y qué se decidió** en una sesión de
> diagnóstico del sobreconteo de personas. Está escrito para que otro agente o compañero pueda
> retomar sin releer la conversación. Ninguna medición aquí es especulativa salvo donde se
> marca explícitamente.

---

## TL;DR

El sobreconteo de ~23× (**465 `new_person` para ~20 personas reales**) **no era** el umbral de
OSNet, ni el `maxShadowTrackingAge`, ni el modelo de ReID.

**Era un sub-stream a 7 fps.** A esa tasa el tracker no lograba asociar dos detecciones
consecutivas de una persona caminando, partía a cada persona en decenas de `track_id`, y cada
fragmento generaba una identidad nueva.

**Acción tomada (2026-07-19):** cambiar `stream_type` de `sub` a `main`. El DVR entrega
**1280×720 @ 15 fps** en main → 2.1× más frames. Pendiente de verificar con datos.

---

## 1. El problema original

- **463–465 eventos `new_person`** en ~28 h contra ~20 personas físicas reales en el sitio.
- Tres rondas previas de calibración (2026-07-08, 07-14, 07-15/16) subieron
  `SIMILARITY_THRESHOLD` de 0.68 → 0.85, eliminaron el umbral de vistas parciales, agregaron
  buffer multi-frame y el umbral acotado de re-match rápido. **El sobreconteo persistió.**

---

## 2. Qué se midió en esta sesión

### 2.1 Distribución de similitud de `new_person` (458 filas con `best_sim` real)

| Rango | Casos |
|---|---|
| < 0.65 | 19 |
| 0.65–0.70 | 45 |
| 0.70–0.75 | 141 |
| 0.75–0.80 | 176 |
| 0.80–0.85 | 77 |
| **≥ 0.85** | **0** |

**Máximo histórico de `new_person`: 0.8423.** Nunca roza el umbral de 0.85. No existe una
acumulación pegada bajo el corte que un umbral más bajo rescataría limpiamente.

### 2.2 Eventos de match

- Solo **26 eventos** de match (20 `channel_change` + 6 `person_return`) contra 465 `new_person`
  → **tasa de match ~5%**.
- **Los 20 `channel_change` tienen `prev_camera == camera_id`** (0 cambios de cámara genuinos).
  Según la regla documentada en CLAUDE.md, deberían haberse degradado a `person_return`.
  → **Anomalía sin diagnosticar, ver §8.**
- El umbral de re-match rápido (`SIMILARITY_THRESHOLD_QUICK_REMATCH=0.75`) **disparó una sola
  vez** en todo el dataset (2026-07-16 18:13, `affcc91ed69e`, sim=0.8133, absent_s=1). Es
  demasiado angosto para tener impacto.

### 2.3 Duración real de tracks (desde `manifest.csv`, 1907 tracks del 15 jul)

| Métrica | Valor |
|---|---|
| **FPS del pipeline** | **7.02** (p10=6.88, p90=7.14 sobre 2858 pares) |
| Duración **mediana** de un track | **2.3 s** |
| p90 de duración | 11.0 s |
| Tracks con **un solo crop** | **39%** (746/1907) |
| `maxShadowTrackingAge=100` equivale a | 14.3 s @ 7 fps |

**Lectura clave:** el tracker tenía 14.3 s de ventana de shadow tracking y aun así el track
mediano moría a los 2.3 s. Los tracks **no morían por expirar la ventana** — morían porque el
tracker **no lograba asociar** dos detecciones consecutivas.

### 2.4 Cómo se derivó el FPS (método reproducible)

No existe ningún campo que reporte FPS. Se derivó así:

1. `frame_num` viene de `frame_meta.frame_num` ([probes.py:1775](deploy/pipelines/probes.py#L1775)) —
   incrementa +1 por cada frame que entra al pipeline para esa cámara.
2. `timestamp` lo pone el Jetson en `post_crop()` ([probes.py:731](deploy/pipelines/probes.py#L731))
   con `datetime.now(utc)`, **en la misma llamada** que lee `frame_num`. No es hora de recepción
   del backend.
3. Entre dos crops del mismo track: `Δframe_num / Δtiempo = fps del pipeline`.

Ejemplo crudo (`DEMOONE-ch02` track 480):
```
frame 65418  →  00:16:28.522978
frame 65480  →  00:16:37.353270
62 frames ÷ 8.830 s = 7.02 fps
```

### 2.5 Test decisivo: ¿cómputo o fuente?

| Personas en escena | fps mediano |
|---|---|
| 1 | 7.02 |
| 2 | 7.01 |
| 3 | 7.02 |
| 4 | 7.01 |
| 5 | 7.02 |
| 6 o más | 7.02 |

OSNet corre **por persona**. Si fuera el cuello de botella, el fps caería al aumentar la carga.
**No se mueve ni una centésima** → el pipeline **no estaba limitado por cómputo**. El 7.02 clavado
todo el día es un reloj externo: la fuente.

---

## 3. Hipótesis probadas y DESCARTADAS

Registro explícito para no repetirlas.

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| `SIMILARITY_THRESHOLD` mal calibrado | ❌ Descartada | Máx. `new_person` = 0.8423, nunca cerca de 0.85 |
| Bajar el umbral globalmente ayudaría | ❌ Descartada (ya en rondas 1–2) | Fusiona personas distintas (6/9 pares en 0.708–0.713) |
| `maxShadowTrackingAge` muy corto | ❌ Descartada | Ya estaba en 100 (14.3 s) el 18 jul y la fragmentación persistió |
| OSNet SGIE ahogando el pipeline | ❌ **Descartada en esta sesión** | (a) `sgie_interval` ya estaba en 3; (b) fps plano bajo carga |
| Tracker `nvdcf_accuracy` roto sin arreglo | ⚠️ Corregida | El modelo `resnet50_market1501.etlt` **sí es descargable** de NGC |

---

## 4. Causa raíz encontrada

**El sistema corría sobre el sub-stream del DVR a 7 fps.**

Aritmética de la asociación del tracker (bbox de persona ≈ 55 cm de ancho, IoU = (W−d)/(W+d)):

| | sub (7 fps) | main (15 fps) |
|---|---|---|
| Detecciones/seg (`pgie_interval=1`) | 3.5 | 7.5 |
| Tiempo entre detecciones | 286 ms | 133 ms |
| Avance de persona caminando (1.4 m/s) | ~40 cm | ~19 cm |
| **IoU entre detecciones** | **0.16** ❌ | **0.49** ✅ |
| `minMatchingScore4Iou` (umbral) | 0.2575 | 0.2575 |

A 7 fps la asociación caía **por debajo del umbral del tracker**. Esa es la explicación
cuantitativa del track mediano de 2.3 s y, en cascada, de las 465 identidades.

---

## 5. Cambios aplicados (2026-07-19)

### 5.1 `stream_type: sub` → `main` ✅ APLICADO

`gst-discoverer-1.0` sobre el canal 2 en main (`subtype=0`) reporta:

```
video #1: H.264 (High Profile)
Width: 1280   Height: 720   Frame rate: 15/1
```

**Config resultante de `clients/Mova/config.yaml`:**
```yaml
rtsp_url_pattern: "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=0"
stream_type: main
channels: [2, 3]
tracker: nvdcf_extended_shadow
pgie_interval: 1
sgie_interval: 3
```

⚠️ **Nota importante:** `stream_type` y `rtsp_url_pattern` son campos **independientes**.
`stream_type` solo fija el lienzo del streammux; **no cambia la URL**. Cambiar uno sin el otro
no produce error visible — el pipeline arranca normal y simplemente no mejora nada. Usar
`identify_dvr.py --stream-type main --update-config` para que los ponga consistentes.

### 5.2 `stream_width` / `stream_height` — PENDIENTE

El DVR entrega **1280×720**, pero `stream_type: main` fija el lienzo en **1920×1080** →
se escala 720p a 1080p, pagando 2.25× el costo de píxeles por información inventada.

**La normalización NO se rompe** (el fix del 2026-07-10 hace que divida por el lienzo, y los
bboxes vienen en ese mismo lienzo). El costo es GPU desperdiciada.

**Arreglo** en `clients/Mova/config.yaml`:
```yaml
stream_width: 1280
stream_height: 720
```

**Verificado en código** que el override funciona
([config_loader.py:382-383](deploy/pipelines/config_loader.py#L382)):
```python
stream_width=int(cfg.get("stream_width", _res["width"])),
stream_height=int(cfg.get("stream_height", _res["height"])),
```
Y ese valor alimenta **ambos** puntos críticos, por lo que no pueden desincronizarse:
- [app.py:333](deploy/pipelines/app.py#L333) → `streammux.set_property("width", ...)` (lienzo real)
- [app.py:305](deploy/pipelines/app.py#L305) → `init_stream_resolution(...)` (divisores de normalización)

**Comprobación tras reiniciar:**
```bash
docker compose logs deepstream 2>&1 | grep -iE "Resolution|Tracker" | head
# debe decir:  Resolution : 1280x720 (main)
```

---

## 6. Efectos secundarios a vigilar

### 6.1 `maxShadowTrackingAge` se acorta en tiempo real ⚠️

Está en **frames**, no en segundos:

| | 7 fps | 15 fps |
|---|---|---|
| 100 frames = | **14.3 s** | **6.7 s** |

Subir el fps **reduce a menos de la mitad** la tolerancia a oclusiones largas. Se gana asociación
cuadro a cuadro pero se pierde ventana de oclusión.

**Decisión:** dejarlo en 100 para la primera medición (no confundir el experimento con tres
variables a la vez). Si persiste fragmentación por oclusión, subir a ~215 para recuperar los 14.3 s.

### 6.2 El umbral de similitud puede necesitar recalibración ⚠️

`SIMILARITY_THRESHOLD=0.85` se calibró con embeddings del **sub-stream**. Crops más grandes y con
más detalle real **mueven la distribución de similitudes**. Reverificar con los datos nuevos antes
de asumir que 0.85 sigue siendo correcto (mismo criterio que ya se aplicó para fp16).

### 6.3 Posible saturación de GPU

A 15 fps son 2.1× más frames por segundo. El pipeline **no era** compute-bound a 7 fps (§2.5),
pero podría serlo ahora. Si el fps medido sale en 10–12 en vez de 15, eso es lo que pasó — se
negocia con `pgie_interval` / `sgie_interval`. No es un fracaso, es el siguiente ajuste.

### 6.4 Beneficio extra: filtros de tamaño mínimo

Los filtros están en **píxeles**:

| Filtro | Umbral | A 960×544 | A 1280×720 |
|---|---|---|---|
| OSNet SGIE (`input-object-min-width/height`) | 96×192 px | mucha gente no llega | 1.33× más grande |
| Cara (`detected-min-w/h`) | 64×64 px | idem | idem |

Quien no llega al mínimo de OSNet **no recibe embedding**: cae al fallback con `global_id=None`
→ no se cuenta, sin ReID, sin cara, sin posición. Ya estaba documentado en CLAUDE.md como
limitación conocida diferida del sub-stream.

---

## 7. Auditoría de impacto en el Platform

**Hallazgo estructural:** *toda* la plataforma está keyed por el `global_id` que emite el Jetson
(ver `AV-Platform/Backend/app/socket/positions.py`). La fragmentación no rompe "el conteo" —
degrada casi todas las features a la vez. El corolario es positivo: **arreglar la asignación de
`global_id` arregla todo de un solo golpe.**

| Feature | Depende de | Impacto de fragmentación |
|---|---|---|
| **People Count** | `SCARD` de SET de gids | 🔴 Inflado ~23× |
| **Trayectorias** | `traj:...{gid}` LIST | 🔴 Destruida (fragmentos de 2 s, no recorridos) |
| **Dwell Time** | `dwell_first/last` por gid | 🔴 Deflacionado |
| **Queue Wait** | `queue_first/last` por gid | 🔴 Deflacionado |
| **Conversión** | `conv_state` dwell acumulado | 🔴 Sub-cuenta: ningún fragmento llega al umbral de 45 s; además Otsu se calibra sobre datos corruptos |
| **Peak Hours** | SET de gids por hora | 🟡 Absolutos mal, **forma correcta** |
| **Demografía** | `reid_demo:{gid}` | 🟡 Huecos: clasificar necesita ~50 frames (`VOTES_REQUIRED=10 × VOTE_SAMPLE_INTERVAL=5`); los fragmentos mueren antes |
| **Heatmap (dwell)** | `hincrbyfloat(celda, elapsed)` | 🟢 **Sano** — acumula sin mirar identidad |
| **Heatmap (conteo/celda)** | `crossed` por gid+celda | 🟡 Distorsionado; la dirección depende de `DWELL_THRESHOLD_SECS` (fragmentos largos → infla; cortos → deflaciona) |
| **Filas (ocupación)** | cuerpos en ROI | 🟢 **Sano** |
| **Asistencia/Empleados** | face → `employee_id` | 🟡 Hoy independiente; el plan nuevo lo ataría a gid → heredaría el problema |

---

## 8. Anomalía abierta (sin diagnosticar)

**Los 20 eventos `channel_change` del dataset tienen `prev_camera == camera_id`.** Según la regla
documentada en CLAUDE.md ("si `channel_change` ocurre con `prev_camera == camera_id` se demota a
`person_return`"), las 20 deberían haberse registrado como `person_return`. Ninguna lo fue.

No afecta la identidad (mismo `global_id` de cualquier forma) ni `person_count`, pero sí cualquier
analítica que lea `event_type` para trayectos cross-cámara. **No se investigó a fondo** — quedó
fuera del alcance de esta sesión.

---

## 9. Plan por capas (si el cambio de stream no basta)

Ordenado por relación valor/esfuerzo. **Solo continuar si la medición de §10 muestra que la
fragmentación persiste.**

### Capa 0 — Recuperar FPS ✅ EN CURSO
Ya descrito en §5. Es el cambio más barato y ataca la causa raíz medida.

### Capa 1 — Re-asociación ReID en el tracker
Habilitar el submódulo de ReID de NvDCF (`reidType: 2`), que compara detecciones nuevas contra
tracks recién perdidos **por apariencia** y **restaura el `track_id` original**. A diferencia del
shadow tracking, no está limitado por ventana de frames.

El bloqueo histórico (`nvdcf_accuracy` "roto") era **solo el modelo faltante**, que sí se descarga:
```
https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt
```
- **Evaluar que reemplace al SGIE OSNet** — sería *quitar* un componente, no agregar.
- **Riesgo:** ResNet50 es más pesado que OSNet-x1.0 en Orin Nano. Medir antes de comprometerse.

### Capa 2 — Métricas sin identidad (red de seguridad)
Correctas **en vivo**, independientes de la calidad del ReID:
- **Ocupación / aforo / largo de fila** → contar cuerpos en un ROI. Sin identidad.
- **Ley de Little** (`L = λW` → `W = L/λ`): dwell y espera promedio = ocupación promedio ÷ tasa de
  llegadas. Teorema de teoría de colas (John Little, 1961), válido para cualquier sistema estable
  sin importar la distribución de llegadas.
  - **Límite:** da la **media**, no la distribución. "¿Cuántos se quedaron >10 min?" sigue
    necesitando identidad.
  - **Dependencia:** λ necesita cruces de línea, datos del POS, o el conteo ya corregido.
- **Heatmap de dwell** → ya sano.

### Capa 3 — Resolución de identidad nocturna (clustering)
Para lo que genuinamente necesita identidad: conteo único, trayectorias, conversión, empleados.

**Qué es:** de noche, agrupar los fragmentos del día en personas reales.
1. Comparar todos contra todos (en vivo es imposible; de noche sobra tiempo).
2. Grafo: fragmento = nodo, arista si se parecen lo suficiente.
3. **Prohibir** aristas imposibles: dos fragmentos vistos **simultáneamente en la misma cámara**
   son definitivamente personas distintas.
4. Componentes conexas = personas reales.

**Los dos superpoderes que el tiempo real no tiene:**
- **Cierre transitivo** — si A~B (0.80) y B~C (0.82) pero A~C (0.62), en vivo A y C nunca se unen;
  en el grafo quedan conectados **a través de B**. Ataca directamente el problema frente-vs-espalda.
- **Exclusión mutua** — bloquea el modo de falla (fusionar personas distintas) que obligó a subir
  el umbral a 0.85, permitiendo **usar un umbral más bajo con seguridad**.

**Implicaciones honestas:**
- ✅ Más preciso que el matching en vivo (más información disponible).
- ❌ **Diferido** — no arregla el número en vivo, solo los reportes.
- ❌ No es perfecto: con uniformes seguirá fusionando gente parecida.
- ⚠️ **Nuevo modo de falla:** una corrida mala corrompe un día entero de golpe.
  **Mitigación obligatoria: nunca borrar los fragmentos crudos.** El clustering debe ser una vista
  derivada y **re-ejecutable**.

Encaja natural en el pipeline nocturno que ya existe (`app.tasks.scheduler` → tablas `*_daily`).

### Capa 4 — Line-crossing donde haya puerta
Contador por cruce de línea virtual en la entrada (`nvdsanalytics`, sin modelo nuevo). Inmune a la
fragmentación: solo necesita que el track exista los pocos frames del cruce.

- **Semántica elegida por el usuario: footfall retail** (cada entrada = una visita).
- **No aplicable en la bodega actual** — ninguna cámara ve una puerta. En retail real sí habrá.
- **Nota:** `people_count` hoy es `SCARD` de gids, **no** un contador de cruces. Adoptarlo implica
  cambiar de dónde sale la métrica (toca el contrato Jetson↔backend).

---

## 10. Medición de verificación (siguiente sesión)

Exportar un dataset nuevo desde `/superadmin/dataset` **con actividad real de gente** y correr el
mismo método de §2.3/§2.4. Comparación directa contra la línea base:

| Métrica | Base (sub, 7 fps) | Meta (main, 15 fps) |
|---|---|---|
| fps medido | 7.02 | ~15 |
| Duración mediana de track | 2.3 s | 15 s+ |
| Tracks con 1 solo crop | 39% | <15% |
| `new_person` por hora | 17.4 | mucho menor |
| Tasa de match | ~5% | mucho mayor |

**Cómo interpretar:**
- **fps sube Y track mediano sube** → hipótesis confirmada; las Capas 1 y 3 pasan de urgentes a
  opcionales.
- **fps sube pero track mediano NO** → la fragmentación tiene otra causa además del frame rate →
  proceder con la Capa 1.
- **fps sale en 10–12 en vez de 15** → nos volvimos compute-bound → negociar con
  `pgie_interval` / `sgie_interval`.

---

## 11. Plan de empleados por dwell — análisis

Propuesta del usuario: detectar empleados por acumulación de dwell (~3 h/día en un `global_id`),
excluirlos de estadísticas de cliente, mandar su galería al backend, clustering nocturno para
fusionar candidatos, y confirmación humana del dueño con fotos. Reconocimiento facial eliminado.

**Estado: bloqueado por la fragmentación, en su forma actual.**
Requiere que un `global_id` acumule ~3 h. Hoy ningún `global_id` sobrevive más de minutos (465
identidades para ~20 personas; `REID_TTL_S=3600` lo expira a la hora). **El umbral nunca se
dispararía.**

**Pero el paso de clustering nocturno de la propuesta es el instinto correcto** — es la semilla de
la Capa 3. Generalizándolo de "solo candidatos a empleado" a **todas las identidades**:
- El umbral de 3 h se vuelve calculable (se suma el dwell del **cluster**, no del fragmento).
- La galería enviada a revisión es la **unión** de todos los fragmentos → más ángulos → mejor
  reconocimiento futuro.
- El paso "galería mezclada → marcar para revisión" **es exactamente** la restricción de exclusión
  mutua, ya intuida en la propuesta original.

⚠️ **Riesgo específico:** con uniformes, el clustering fusionará empleados entre sí. El
reconocimiento facial es la **única** señal que distingue a dos personas con la misma ropa.
**Recomendación: no eliminarlo, sino degradarlo de detector primario a desambiguador *dentro* del
cluster.**

---

## 12. Apéndice — Scripts de medición

Los scripts usados viven en el scratchpad de la sesión (no versionados). Se reconstruyen fácil:

**FPS y duración de tracks** — desde `manifest.csv`, agrupar por `(camera_id, track_id)`, ordenar
por `frame_num`, y para pares consecutivos calcular `Δframe_num / Δtimestamp`. La mediana sobre
todos los pares es el fps. El span `primer→último` de cada track es su duración observada
(censurada por `CROP_MAX_PER_PERSON=5`).

**fps vs carga** — agrupar los mismos pares por número de tracks distintos activos en esa cámara
ese minuto. Si el fps cae al subir la carga → compute-bound. Si es plano → limitado por la fuente.

**FPS de la fuente (DVR)** — arma la URL desde la config y sondea:
```bash
cd ~/dev/NX-JETSON/deploy && python3 - <<'PY'
import re, pathlib, subprocess
client = pathlib.Path('/etc/nx_client').read_text().strip()
dvr_ip = pathlib.Path('/etc/nx_dvr_ip').read_text().strip()
cfg = pathlib.Path('clients/%s/config.yaml' % client).read_text()
env = {}
for line in pathlib.Path('clients/%s/.env' % client).read_text().splitlines():
    line = line.strip()
    if line.startswith('export '):
        line = line[7:]
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
port = re.search(r'^dvr_port:\s*(\d+)', cfg, re.M).group(1)
pat  = re.search(r'''^rtsp_url_pattern:\s*["'](.+?)["']''', cfg, re.M).group(1)
ch   = int(re.search(r'^channels:\s*\[(.*?)\]', cfg, re.M).group(1).split(',')[0])
url  = pat.format(user=env['DVR_USER'], password=env['DVR_PASS'],
                  dvr_ip=dvr_ip, port=port, ch=ch)
r = subprocess.run(['gst-discoverer-1.0', '-t', '15', url], capture_output=True, text=True)
for l in (r.stdout + r.stderr).splitlines():
    if re.search(r'frame ?rate|width|height|Codec|video', l, re.I):
        print(l.replace(env['DVR_PASS'], '****'))
PY
```

---

## 13. Referencias externas

- [NvDCF ReID: modelo `resnet50_market1501` faltante (foro NVIDIA)](https://forums.developer.nvidia.com/t/nvdcf-reid-miss-resnet50-market1501-etlt-b100-gpu0-fp16-engine/283335)
- [ReID y re-asociación de targets en DeepStream (RidgeRun)](https://developer.ridgerun.com/wiki/index.php/ReID_and_Target_Reassociation_using_NVIDIA_Deepstream)
- [Gst-nvtracker — documentación NVIDIA](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
