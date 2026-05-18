# Continue — Sesión 2026-05-18

Documento de continuidad para la próxima sesión de Claude. Resume el trabajo realizado, el estado actual, y lo que falta verificar.

---

## Qué se hizo en esta sesión

### 1. Diagnóstico del QA Visual — output mal interpretado

El usuario veía en la QA app números como `419481+0 · ch09` y pensaba que eran track IDs cambiando. **Diagnóstico:** esos números son los microsegundos del timestamp (`datetime.isoformat()[-12:-4]`), no track IDs. Los track IDs reales son `P#0`, `P#1`, etc. (locales por cámara). No había ningún bug.

### 2. Implementación del ReID Manager local

Se decidió mover el cross-camera re-ID del backend al Jetson para:
- Detectar si una persona es nueva vs. regresó vs. cambió de cámara
- Persistir identidades 1 hora (TTL)
- Evitar double-counting sin depender de latencia de red

**Archivos creados:**
- `deploy/pipelines/reid_manager.py` — nuevo. DB en memoria + JSON persistente (`deploy/reid_db.json`). `match_or_create(embedding, camera_id)` retorna `(global_id, event_type, prev_camera_id)`.

**Archivos modificados:**
- `deploy/pipelines/probes.py`:
  - `_TrackState`: 5 campos nuevos (`entry_emitted`, `entry_deadline`, `global_id`, `pending_bbox`, `pending_conf`)
  - `post_person_entry`: agrega `global_id` y `entry_type: "new"/"return"`
  - `post_person_channel_change`: método nuevo
  - `post_person_exit`: agrega `global_id`
  - `init_workers` / `stop_workers`: crea/flush ReIdManager
  - `_expire_lost_tracks`: maneja entries diferidas
  - `_handle_appearance_reid`: helper nuevo, unifica lógica de AppearanceWorker + ReID para ambos probes
  - Ambos probes (QA + producción): entry diferida hasta que embedding esté listo (deadline 90 frames)
  - Probe B: muestra `global_id` en label QA (`P#4·a3f2b1`)
  - Probe A: escribe `global_id` a `_track_labels`

**Lógica de eventos:**
| Caso | Evento |
|---|---|
| Primera vez visto | `person_entry` + `entry_type: "new"` |
| Mismo global_id, ausente > 5 min | `person_entry` + `entry_type: "return"` |
| Mismo global_id, ausente ≤ 5 min | `person_channel_change` + `prev_camera_id` |
| Sin OSNet model | comportamiento anterior: entry inmediato, `post_person_appearance` al backend |

### 3. Ajustes de parámetros (ReID no funcionaba visualmente)

**Problemas encontrados y corregidos:**

| Problema | Antes | Después |
|---|---|---|
| Threshold demasiado alto para CCTV real | `SIMILARITY_THRESHOLD=0.65` | `0.55` en `reid_manager.py` |
| Crop mínimo muy grande para sub-streams | `CROP_MIN_HEIGHT=128` | `96` en `probes.py` |
| Crop mínimo ancho | `CROP_MIN_WIDTH=64` | `48` en `probes.py` |
| Sin feedback visual en QA | label `P#4` | label `P#4·a3f2b1` (global_id[:6]) |
| Sin diagnóstico en logs | — | `logger.info("ReID track=%d cam=%s → %s gid=%s", ...)` |

---

## Estado actual

### Código — listo para prueba
- `reid_manager.py`: sintaxis OK, lógica completa
- `probes.py`: sintaxis OK, ambos probes actualizados

### Pendiente — verificar en Jetson
El pipeline **no se levantó en esta sesión**. Hay que:

1. Desplegar los cambios al Jetson
2. Confirmar en logs que el ReID está activo:
   ```
   ReIdManager active — DB: /app/reid_db.json   ← bueno
   OSNet model not found at ...                  ← problema
   ```
3. El modelo OSNet **existe** en el Jetson (`models/osnet/osnet_x0_25_market1501.onnx`, 830KB + 765KB).
4. En el QA visual, buscar sufijos `·xxxxxx` en los labels de bounding boxes.
5. En logs, buscar líneas `ReID track=X cam=chYY → channel_change gid=... prev=...` al moverse entre cámaras.

### Si el ReID aún no matchea tras los ajustes
- Bajar threshold a `0.45` en `reid_manager.py` (`SIMILARITY_THRESHOLD`)
- Revisar si las cámaras tienen ángulos muy distintos (el modelo fue entrenado en vistas laterales/frontales)
- Considerar reemplazar OSNet-x0.25 por OSNet-x1.0 (más grande, mejor accuracy) — ver `Future.md`

---

## Archivos clave tocados en esta sesión

| Archivo | Qué cambió |
|---|---|
| `deploy/pipelines/reid_manager.py` | **Nuevo** — ReID DB local |
| `deploy/pipelines/probes.py` | `_TrackState`, API methods, workers, ambos probes, helper |
| `CLAUDE.md` | Actualizado: sección Re-ID, archivo `reid_manager.py` en descripción |

---

## Contexto de conversación relevante

- El usuario usa el Jetson en `~/dev/NX-JETSON/deploy` (no `NX_tech/deploy`)
- El pipeline corre con sub-streams DVR (960×544) en configuración multi-cámara
- La QA app está activa vía `./qa.sh` y accede vía Tailscale
- El modelo OSNet está descargado y presente en el Jetson
- El container se llama `deepstream` (docker compose)
