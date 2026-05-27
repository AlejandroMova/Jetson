# Continue.md — 2026-05-25

## Qué estábamos haciendo exactamente
- Implementar el sistema de grabación QA + biblioteca de clips + modo playback (plan completo en `/home/alex/.claude/plans/hola-claude-este-es-mellow-hinton.md`)
- Al final de la sesión: descubierto que el DVR en `192.168.10.17` no responde (cambió de IP o está apagado). El pipeline arranca bien pero sin video.
- Pendiente implementar: auto-recuperación cuando el DVR cambia de IP (documentado en Future.md).

## Estado actual
**Qué funciona:**
- `recording_manager.py` — grabación automática de clips cuando se detectan personas (IDLE → RECORDING → IDLE)
- `mjpeg_server.py` — pasa frames al RecordingManager vía `push_tiled_frame()`
- `probes.py` — Probe A y production probe llaman `notify_detection()` y `push_camera_frame()` al RecordingManager
- `app.py` — instancia RecordingManager cuando `cfg.recording_enabled or _IS_QA_ENABLED`; detecta modo playback via GLib.timeout cada 5 s; sale con código 42 para señalizar cambio a playback
- `docker-entrypoint.sh` — loop live/playback en QA mode; playback corre `app_video_testing.py --input <path> --no-loop`
- `app_video_testing.py` — acepta `--input` y `--no-loop`
- `docker-compose.qa.yml` — YAML correcto (un solo bloque `deepstream` con `environment` + `volumes`)
- `docker-compose.yml` — volumen `./recordings:/nx_tech/recordings` agregado al servicio `deepstream` de producción
- `config_loader.py` — campo `recording_enabled: bool = False` en `ClientConfig`
- `streamlit_app.py` — tab "📹 Grabaciones" con lista de clips, preview, inferencia, eliminar; toggle `recording_enabled` en sidebar config editor; sidebar muestra estado de grabación/playback
- `recordings/.gitkeep` — directorio trackeado en git

**Qué no funciona / está roto:**
- DVR en `192.168.10.17` no responde (ping falla). El Jetson está en `192.168.10.183`, subnet `192.168.10.0/24`. Sin video en el MJPEG porque no entran frames RTSP.
- No hay auto-recuperación de IP del DVR — pendiente implementar (ver Future.md y próximos pasos).

## Decisiones tomadas y por qué
- **RecordingManager NO usa Redis pub/sub como trigger:** los probes llaman `notify_detection()` directamente, lo que funciona tanto en QA como en producción (en producción no hay Redis QA).
- **`tiled.mp4` solo en QA:** el tiler no existe en producción, así que en producción solo se graban los `<cam_id>.mp4` full-res.
- **`recording_enabled` en config.yaml, no en docker-compose:** el usuario no quiere editar docker-compose en campo porque puede romper cosas. La config es editable desde el QA dashboard sin reiniciar containers.
- **`recording_enabled` publicado en `nx:qa:status`:** para que el dashboard Streamlit pueda inicializar el toggle con el valor actual del pipeline.

## Qué intentamos que NO funcionó
- **Doble bloque `deepstream` en docker-compose.qa.yml:** agregamos el volumen en un segundo bloque `deepstream` separado, lo que causó error YAML "mapping key already defined". Solución: fusionar ambos en un solo bloque.

## Próximos pasos concretos
1. **Implementar auto-recuperación DVR** (está en Future.md — el usuario confirmó que quiere esto):
   - En `deploy/pipelines/app.py`: llevar la cuenta de streams RTSP fallidos en `_on_bus_message`. Si todos fallan dentro de los primeros 30 s de pipeline en PLAYING → ejecutar `nmap -p 554 <subnet> --open` (subnet derivado de IP del Jetson con `ip route get 8.8.8.8`). Si encuentra una IP diferente a la actual → escribir en `/etc/nx_dvr_ip` → `sys.exit(1)` para que el entrypoint reinicie el pipeline.
   - No activar por fallas individuales — solo si el 100% de los streams falla.
2. **Encontrar el DVR manualmente en lo que se implementa el punto 1:**
   - Dentro del Jetson: `nmap -p 554 192.168.10.0/24 --open -oG - | grep "/open"`
   - Actualizar: `echo "192.168.10.XX" | sudo tee /etc/nx_dvr_ip && ./qa.sh stop && ./qa.sh`
3. **Probar el flujo de grabación end-to-end** cuando el DVR esté disponible: verificar que se creen clips en `/nx_tech/recordings/`, que aparezcan en la tab Grabaciones con thumbnail y metadata, y que el botón "▶ Correr Inferencia" reinicie el pipeline en modo playback.

## Parámetros y valores concretos en juego
- DVR IP configurada: `192.168.10.17` (en `/etc/nx_dvr_ip` y `config.yaml`)
- Jetson IP: `192.168.10.183`
- Subnet correcto para scan: `192.168.10.0/24`
- Directorio de grabaciones: `/nx_tech/recordings` (bind mount `./recordings:/nx_tech/recordings`)
- Cooldown grabación: 10 s sin detecciones → cierra clip
- Duración mínima: 5 s (clips más cortos se descartan)
- Duración máxima: 5 min
- Auto-prune: elimina clips más antiguos cuando total > 10 GB
- Exit code 42: señal de `app.py` al entrypoint para cambiar a modo playback

## Error / síntoma actual
```
RTSP 'source-0' failed: gst-resource-error-quark: Could not open resource for reading and writing. (7) — pipeline continues.
[... mismo error para source-0 a source-15 ...]
```
El pipeline arranca, el MjpegServer levanta en :8080, pero sin frames porque el DVR no responde.

## Archivos modificados sin commitear
- `deploy/pipelines/recording_manager.py` — nuevo archivo (completo)
- `deploy/pipelines/mjpeg_server.py` — agrega `recorder` parameter + `push_tiled_frame()` en encode loop
- `deploy/pipelines/probes.py` — agrega `_recording_manager` global + `notify_detection()` + `push_camera_frame()` en Probe A y production probe
- `deploy/pipelines/app.py` — instancia RecordingManager, lo pasa a MjpegServer, agrega GLib.timeout para playback mode, publica `recording_enabled` en nx:qa:status
- `deploy/pipelines/config_loader.py` — agrega `recording_enabled: bool = False` a ClientConfig
- `deploy/pipelines/app_video_testing.py` — agrega `--input` y `--no-loop` args
- `deploy/docker-entrypoint.sh` — loop live/playback en QA mode
- `deploy/docker-compose.yml` — volumen recordings en deepstream
- `deploy/docker-compose.qa.yml` — YAML corregido, volumen recordings en deepstream
- `deploy/qa_app/streamlit_app.py` — tab Grabaciones + toggle recording_enabled en config editor + indicador estado sidebar
- `deploy/recordings/.gitkeep` — nuevo (directorio trackeado)
- `CLAUDE.md` — actualizado (RecordingManager, config_loader, streamlit_app, docker-compose.qa)
- `Future.md` — entrada auto-recuperación DVR agregada

## Archivos y secciones que estábamos modificando
| Archivo | Función / sección | Qué se estaba cambiando |
|---------|-------------------|-------------------------|
| `deploy/pipelines/app.py` | bloque QA setup (~líneas 292–320) | RecordingManager + publicar recording_enabled en nx:qa:status |
| `deploy/qa_app/streamlit_app.py` | config editor sidebar (~líneas 460–520) | toggle recording_enabled + reset list |
| `Future.md` | nueva entrada | auto-recuperación DVR |
