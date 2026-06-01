# Continue.md — 2026-06-01

## Qué estábamos haciendo exactamente
Diagnosticando y corrigiendo el flujo completo de "Correr Inferencia" en el QA Visual Dashboard. El botón clickeaba, seteaba la key en Redis, pero el pipeline nunca cambiaba a modo playback. Luego, una vez que el pipeline sí paraba, el dashboard mostraba "refused to connect" en lugar del video de inferencia.

## Estado actual
**Qué funciona:**
- El watcher thread (`playback-watcher`) está implementado y debería detectar la key en ~3s
- `app.py` limpia `playback_info` al arrancar (resuelve stale data de sesiones anteriores con Ctrl+C)
- Streamlit muestra pantalla de "cargando" mientras el pipeline transiciona (no el stream live)
- Streamlit solo muestra el iframe MJPEG cuando `playback_info` está seteado (inferencia realmente corriendo)

**Qué no hemos probado aún:**
- El flujo completo end-to-end con los tres fixes juntos
- La última prueba del usuario mostró "refused to connect" — era por `playback_info` stale de sesión anterior con Ctrl+C. Fix aplicado, sin probar todavía.

## Decisiones tomadas y por qué
- **Thread dedicado en lugar de `GLib.timeout_add`:** El GLib main loop estaba saturado con mensajes de GStreamer (16 cámaras + inferencia) y los timers nunca se disparaban. El thread corre independientemente y usa `GLib.idle_add(loop.quit)` para parar el loop de forma thread-safe.
- **`GLib.idle_add(loop.quit)` en lugar de `pipeline.send_event(EOS)`:** `rtspsrc` es una live source y puede ignorar o no propagar eventos EOS downstream. `loop.quit()` para el GLib main loop directamente; el cleanup ocurre en el `finally` block via `pipeline.set_state(Gst.State.NULL)`.
- **`playback_info` como señal de "inferencia lista":** `app_video_testing.py` ya seteaba esta key cuando arrancaba. Usarla en Streamlit para distinguir "transicionando" (no mostrar iframe) de "corriendo" (mostrar iframe) evita que el usuario vea el stream live de noche durante la transición.
- **Limpiar `playback_info` en `app.py` al arrancar:** Ctrl+C usa `docker kill` (SIGKILL) y no corre los `finally` blocks de Python. Si `app_video_testing.py` fue matado así, `playback_info` queda en Redis. `app.py` la borra al arrancar para garantizar estado limpio.

## Qué intentamos que NO funcionó
- **`pipeline.send_event(Gst.Event.new_eos())`:** Primera implementación. Falló porque `rtspsrc` es live source y no propagó el EOS downstream. El loop nunca recibió el mensaje EOS y `app.py` corría indefinidamente.
- **`loop.quit()` desde `GLib.timeout_add`:** Segunda implementación. Falló porque el GLib main loop estaba saturado con mensajes de GStreamer y el timer nunca se disparaba (o tardaba demasiado).

## Próximos pasos concretos
1. Hacer `./qa.sh stop && ./qa.sh` para levantar con el código nuevo
2. Ir a la tab Grabaciones, expandir un clip, clickear "▶ Correr Inferencia"
3. Verificar en los logs del container que aparece `[QA] Playback watcher iniciado` al arrancar
4. Verificar que tras clickear aparece `[QA] Playback solicitado: ... — deteniendo pipeline` en ~3s
5. Verificar que Streamlit muestra la pantalla de "cargando" durante la transición (no el video live)
6. Verificar que después de 30-120s aparece el iframe con el video de inferencia

**Cómo ver los logs del container deepstream:**
```bash
docker compose -f docker-compose.yml -f docker-compose.qa.yml logs -f deepstream
```

## Parámetros y valores concretos en juego
- `_time.sleep(3)`: intervalo de polling del watcher thread — detecta la key en 0-3s
- `nx:qa:playback_video`: key seteada por Streamlit al clickear el botón — path al video
- `nx:qa:playback_info`: key seteada por `app_video_testing.py` cuando arranca — señal de "inferencia lista"
- `sys.exit(42)`: código de salida de `app.py` para señalar al entrypoint que arranque playback
- `docker-entrypoint.sh` loop: si exit code es 42 o 0, continúa el loop; cualquier otro código sale

## Error / síntoma actual (si aplica)
```
(última prueba del usuario — 2026-06-01)
Tab Grabaciones → "Inferencia en curso: tiled.mp4"
iframe MJPEG → "100.67.192.58 refused to connect"

Causa: playback_info stale en Redis de sesión anterior parada con Ctrl+C (SIGKILL).
       SIGKILL no corre finally blocks → playback_info nunca se borró.
       Streamlit veía playback_info seteado → mostraba iframe → MjpegServer no estaba corriendo.

Fix aplicado: app.py ahora borra nx:qa:playback_info al arrancar en QA mode (junto a config_overrides).
```

## Archivos modificados sin commitear
- `deploy/pipelines/app.py` — watcher thread (`_playback_watcher`) reemplaza `GLib.timeout_add`; borra `nx:qa:playback_info` al arrancar en QA mode
- `deploy/qa_app/streamlit_app.py` — pantalla de "cargando" durante transición; lee `nx:qa:playback_info` para saber cuándo mostrar el iframe

## Archivos y secciones que estábamos modificando
| Archivo | Función / sección | Qué se estaba cambiando |
|---------|-------------------|-------------------------|
| `deploy/pipelines/app.py` | `main()` — bloque QA startup (~línea 467) | `_redis_qa.delete("nx:qa:playback_info")` junto a los otros deletes al arrancar |
| `deploy/pipelines/app.py` | `main()` — bloque playback watcher (~línea 582) | Reemplazar `GLib.timeout_add` por thread dedicado con `GLib.idle_add(loop.quit)` |
| `deploy/qa_app/streamlit_app.py` | `tab_recordings` (~línea 735) | Leer `playback_info`, mostrar loading screen vs iframe según estado |
