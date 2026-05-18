# Future Improvements — NX Computing AI

Registro de mejoras técnicas identificadas durante el desarrollo. Cada entrada documenta una posible implementación futura con suficiente contexto para evaluarla e implementarla sin tener que reconstruir la conversación original.

Ver regla 11 de CLAUDE.md para el formato de entradas y el protocolo completo.

---

## EMA adaptativo con pesos por calidad de crop en ReIdManager

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
