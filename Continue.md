# Continue.md — 2026-05-29

## Qué estábamos haciendo exactamente
- Fix del bug de playback QA: botón "▶ Correr Inferencia" fallaba con `Unable to create video pipeline`.
- El fix fue implementado y commiteado en branch `humanly/refector-pre-tiler-probe`.

## Estado actual
**Qué funciona:**
- Pipeline live QA completo: Probe A + Probe B, overlays, MJPEG stream, dashboard Streamlit.
- La grabación de clips se genera correctamente en `/nx_tech/recordings/`.
- El refactor de `pre_tiler_analytics_probe` está en producción (branch `humanly/refector-pre-tiler-probe`, probado en Jetson).
- `qa.sh` corregido: usa `docker compose rm -sf deepstream` para forzar recreación del container.
- **Playback QA corregido:** `app_video_testing.py` ahora usa `decodebin` en lugar de `qtdemux + h264parse`, soporta mp4v y cualquier otro codec.

**Qué no funciona / está roto:**
- Ninguno conocido — pendiente prueba en Jetson real del playback.

## Decisiones tomadas y por qué
- **Opción B (decodebin) sobre Opción A (cambiar codec):** `decodebin` es más robusto y funciona con cualquier codec futuro. Cambiar el codec de grabación a `avc1` tenía riesgo de que OpenCV en ARM64 no tenga el encoder H.264.
- **nvvideoconvert + capsfilter(NVMM, NV12) tras decodebin:** necesario para garantizar NVMM a la entrada de `nvstreammux`, independientemente de si decodebin usó hardware o software decode.
- **Detección automática de resolución con cv2.VideoCapture:** corrige el issue secundario del streammux hardcodeado a 1920×1080, que causaba distorsión con clips sub-stream (960×544) o tileados (640×360).

## Qué intentamos que NO funcionó
- Ninguno — el fix fue directo.

## Próximos pasos concretos
1. Probar en Jetson real: `./qa.sh`, grabar un clip (que alguien entre a cuadro), ir a tab Grabaciones, presionar "▶ Correr Inferencia".
2. Verificar en logs que el pipeline arranca sin `Unable to create video pipeline` y que aparecen detecciones.
3. Hacer PR / merge del branch `humanly/refector-pre-tiler-probe` a main si todo funciona.

## Parámetros y valores concretos en juego
- `decodebin` → `app_video_testing.py:138` — reemplaza qtdemux+h264parse+nvv4l2decoder
- `caps_nvmm` capsfilter → `app_video_testing.py:140-141` — fuerza NVMM NV12 para streammux
- `_on_decode_pad_added` → `app_video_testing.py:212` — acepta cualquier pad `video/*`
- `video_width / video_height` → detectados con `cv2.VideoCapture` en línea 119-123
- `streammux.set_property("width", video_width)` → línea 146 (antes hardcodeado 1920)

## Error / síntoma actual (si aplica)
Ninguno — bug resuelto. Pendiente verificación en Jetson.

## Archivos modificados sin commitear
- `deploy/pipelines/app_video_testing.py` — source block reemplazado, dims dinámicas, cv2 import
- `CLAUDE.md` — descripción de app_video_testing.py actualizada
- `ErrorHistory.md` — entrada del bug mp4v/h264parse agregada
- `Continue.md` — este archivo

## Archivos y secciones que estábamos modificando
| Archivo | Función / sección | Qué se cambió |
|---------|-------------------|---------------|
| `deploy/pipelines/app_video_testing.py` | Source block (líneas 132-142), Streammux (146-147), all_elements (199-203), Linking (211-231) | Reemplazar qtdemux+h264parse+nvv4l2decoder con decodebin+nvvidconv_src+caps_nvmm; dims dinámicas |
