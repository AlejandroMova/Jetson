# Continue.md — 2026-06-01

## Qué estábamos haciendo exactamente
Arreglando el flujo completo de "▶ Correr Inferencia" en el QA Visual Dashboard (tab Grabaciones). El botón seteaba Redis pero `app_video_testing.py` nunca arrancaba, o arrancaba pero el video aparecía y desaparecía en segundos.

## Estado actual
**Qué funciona:**
- El botón "Correr Inferencia" ahora arranca `app_video_testing.py` correctamente
- El watcher thread en `app.py` detecta `nx:qa:playback_video` en ≤3s y llama `loop.quit()`
- El entrypoint detecta el exit code 42 y arranca el playback (fix `set +e`/`set -e`)
- Streamlit muestra pantalla de "cargando" durante la transición (no el stream live)
- `playback_info` stale de sesiones anteriores se borra al arrancar `app.py`
- `docker-entrypoint.sh` montado como volumen: cambios toman efecto sin rebuild de imagen
- `Dockerfile.jetson` usa `ENTRYPOINT ["/bin/bash", "..."]`: no requiere execute bit en el archivo (resuelve `permission denied` con volume mount en NTFS/ext4)

**Qué no hemos probado aún:**
- El fix de pre-roll (`set_state(PAUSED)` antes de `PLAYING`) — resuelve que el video dure correctamente
  - Usuario reportó que el video aparecía y desaparecía en segundos
  - Causa: `filesrc` leía el archivo mientras TRT engines compilaban, llegaba a EOS antes de que el usuario viera frames
  - Fix: `pipeline.set_state(Gst.State.PAUSED)` + `pipeline.get_state(15min timeout)` → `playback_info` seteado → `PLAYING`
  - Este cambio está aplicado en `app_video_testing.py` pero pendiente de probar

## Decisiones tomadas y por qué
- **`set +e`/`set -e` alrededor de `"$@"` en entrypoint:** `set -e` en bash puede matar el script cuando `app.py` sale con código 42 (non-zero). Aunque el comportamiento de `set -e` en while loops es ambiguo entre versiones de bash, la protección explícita es más robusta.
- **Volume mount de `docker-entrypoint.sh` en `docker-compose.yml`:** El entrypoint está `COPY`-ado en la imagen. Sin el mount, cambios al entrypoint requieren `docker build`. Con el mount, aplican con solo reiniciar el container.
- **`ENTRYPOINT ["/bin/bash", "..."]` en Dockerfile.jetson:** Cuando el archivo se monta desde un host Windows/NTFS, el bit de ejecución no se preserva → `permission denied`. Invocar bash explícitamente elimina esa dependencia. También elimina el `RUN chmod +x` layer.
- **Pre-roll PAUSED antes de PLAYING:** Con `filesrc`, GStreamer puede seguir leyendo el archivo mientras `nvinfer` compila TRT engines. Al quedarse en PAUSED, el pipeline bloquea el source hasta que todos los elementos están listos. Solo después se setea `playback_info` para que Streamlit muestre el iframe.
- **`GLib.idle_add(loop.quit)` se mantuvo:** Aunque `loop.quit()` directo es también thread-safe, el ErrorHistory documenta que `idle_add` fue la solución elegida tras dos intentos fallidos anteriores. No contradice nada y funciona.

## Qué intentamos que NO funcionó
- **`pipeline.send_event(Gst.Event.new_eos())`:** rtspsrc ignora EOS como señal de terminación (live source). Documentado en ErrorHistory.md 2026-05-31.
- **`GLib.timeout_add` para detectar playback:** El GLib main loop estaba saturado con mensajes de GStreamer, el timer nunca disparaba. Documentado en ErrorHistory.md 2026-05-31.
- **Imagen Docker sin rebuild:** El `set +e` fix al entrypoint no tomaba efecto porque el archivo está `COPY`-ado en la imagen. El contenedor seguía usando el entrypoint viejo hasta que se montó como volumen.

## Próximos pasos concretos
1. Probar el fix de pre-roll: `./qa.sh stop && ./qa.sh`, clickear "▶ Correr Inferencia", verificar que el video dura los ~60s completos
2. Verificar en logs que aparece `[QA] Pre-rolling pipeline` y luego `[QA] Pre-roll completo`:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.qa.yml logs -f deepstream
   ```
3. Si el pre-roll tarda >15 min (TRT engines no cacheados), aumentar `_PREROLL_TIMEOUT_NS` en `app_video_testing.py` línea ~430
4. (Opcional) Actualizar ErrorHistory.md y CLAUDE.md con los fixes de esta sesión

## Parámetros y valores concretos en juego
- `_PREROLL_TIMEOUT_NS = 15 * 60 * 1_000_000_000`: timeout de 15 min para esperar PAUSED (compilación TRT). En `app_video_testing.py` ~línea 430.
- `nx:qa:playback_video`: key seteada por Streamlit → detectada por `_playback_watcher` en `app.py` cada 3s
- `nx:qa:playback_info`: key seteada por `app_video_testing.py` DESPUÉS del pre-roll → Streamlit muestra iframe
- `sys.exit(42)`: código de salida de `app.py` para señalar al entrypoint que arranque playback
- `set +e` / `set -e`: bloque en `docker-entrypoint.sh` ~línea 144 que protege la captura de exit code 42

## Error / síntoma actual (si aplica)
```
(pendiente de probar el fix de pre-roll)
Síntoma antes del fix:
  - Click "Correr Inferencia" → pantalla cargando →  video aparece por 2-5 segundos → desaparece
  - Causa: filesrc llegaba a EOS durante compilación TRT, video casi terminado cuando frames empezaban a fluir
```

## Archivos modificados sin commitear
- `deploy/docker-entrypoint.sh` — `set +e`/`set -e` alrededor de `"$@"` en el bloque live mode
- `deploy/docker-compose.yml` — volume mount de `docker-entrypoint.sh`
- `deploy/Dockerfile.jetson` — `ENTRYPOINT ["/bin/bash", "..."]`, eliminado `RUN chmod +x`
- `deploy/pipelines/app_video_testing.py` — pre-roll PAUSED antes de PLAYING; `playback_info` movido a después del pre-roll

## Archivos y secciones que estábamos modificando
| Archivo | Función / sección | Qué se estaba cambiando |
|---------|-------------------|-------------------------|
| `deploy/docker-entrypoint.sh` | bloque `else` del loop live (~línea 139) | `set +e`/`set -e` alrededor de `"$@"` |
| `deploy/docker-compose.yml` | volumes del servicio `deepstream` | volume mount `./docker-entrypoint.sh:/usr/local/bin/docker-entrypoint.sh` |
| `deploy/Dockerfile.jetson` | ENTRYPOINT (~línea 34) | `/bin/bash` explícito, sin `chmod +x` |
| `deploy/pipelines/app_video_testing.py` | bloque QA services + main loop (~línea 374–420) | pre-roll PAUSED; `playback_info` movido post-preroll |
