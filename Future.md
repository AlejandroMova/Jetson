# Future Improvements — NX Computing AI

Registro de mejoras técnicas identificadas durante el desarrollo. Cada entrada documenta una posible implementación futura con suficiente contexto para evaluarla e implementarla sin tener que reconstruir la conversación original.

Ver regla 11 de CLAUDE.md para el formato de entradas y el protocolo completo.

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
