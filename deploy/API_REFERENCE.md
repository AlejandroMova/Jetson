# NX Computing AI — Backend API Reference

This document is intended for the **backend developer**. It describes every JSON payload the Jetson sends, when it sends it, and what the backend should do with it.

---

## 1. System overview

| Layer | Role |
|-------|------|
| **Jetson** | Detects persons, classifies (age/gender, EPP, etc.), generates appearance vectors, emits REST events and WebSocket position snapshots. Stateless between restarts. |
| **Backend** | Receives all events, maintains session state, computes business metrics, serves the dashboard, triggers push notifications. |

**What the Jetson does NOT do:** cross-camera matching, global dwell time, heatmap rendering, push notifications, visit history.

---

## 2. Authentication

All REST requests include:
```
X-API-Key: <api_key>
Content-Type: application/json
```

WebSocket handshake includes the same header.

---

## 3. Common fields

Every `POST /api/events` payload shares these fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | string (UUID4) | Idempotency key — discard duplicates on retry |
| `type` | string | Event type (see table below) |
| `sector` | string | `"comercio"` \| `"industrial"` \| `"hogar"` |
| `jetson_id` | string | Device identifier (e.g. `"jetson-mova-001"`) |
| `camera_id` | string | `"{jetson_id}-ch{N:02d}"` (e.g. `"jetson-mova-001-ch01"`) |
| `timestamp` | string (ISO 8601 UTC) | Event time |
| `severity` | string | `"info"` \| `"medium"` \| `"high"` \| `"critical"` |

---

## 4. Event types by feature

### Feature: People counting (`people_counting` — always active)

#### `person_entry`
Fired the first time the tracker sees a person in this camera session.

```json
{
  "event_id": "uuid4",
  "type": "person_entry",
  "sector": "comercio",
  "jetson_id": "jetson-mova-001",
  "camera_id": "jetson-mova-001-ch01",
  "timestamp": "2025-05-01T14:32:00.123Z",
  "severity": "info",
  "track_id": 42,
  "bbox": { "left": 100, "top": 200, "width": 60, "height": 180 },
  "confidence": 0.92,
  "is_entry_exit_camera": true
}
```

`is_entry_exit_camera: true` means this camera covers an entrance/exit door. Use it to start a `PersonSession` (store visit).

#### `person_appearance`
Sent ~1-2 seconds after `person_entry`, once the AppearanceWorker has processed the crop. Contains the 512-dim OSNet appearance vector for cross-camera re-ID.

```json
{
  "type": "person_appearance",
  "track_id": 42,
  "appearance_vector": [0.12, -0.34, 0.07, ...]
}
```

`appearance_vector` is 512 floats, L2-normalized. `dot(v1, v2)` = cosine similarity (no further normalization needed). Use this to match the person against `recent_exits` from other cameras.

`person_entry` and `person_appearance` are joined by `(jetson_id, camera_id, track_id)`.

#### `person_exit`
Fired when the tracker loses the person for > 2 seconds.

```json
{
  "type": "person_exit",
  "track_id": 42,
  "dwell_seconds": 145.3,
  "is_entry_exit_camera": true
}
```

`dwell_seconds` is the time this person was visible in this specific camera. For total store dwell time, use your own `PersonSession.entry_time → exit_time`.

---

### Feature: Age/gender classification (`age_gender`)

#### `person_classified`
Sent once per person after ≥ 10 voting samples. If the person exits before 10 samples are collected, this event never arrives — that's expected.

```json
{
  "type": "person_classified",
  "track_id": 42,
  "bbox": { "left": 100, "top": 200, "width": 60, "height": 180 },
  "demographics": {
    "gender": "female",
    "age_group": "adult",
    "label": "female_adult",
    "confidence": 0.82
  }
}
```

---

### Feature: Fall detection (`fall_detection` — Hogar only)

#### `fall_detected`
Sent when MoveNet detects a fall (≥ 2/3 rules triggered). 4-second cooldown per track prevents duplicate alerts.

```json
{
  "type": "fall_detected",
  "severity": "critical",
  "track_id": 7,
  "bbox": { "left": 100, "top": 400, "width": 200, "height": 60 },
  "fall_score": 3,
  "avg_kp_conf": 0.72
}
```

Severity: `"critical"` for `hogar`, `"high"` for `industrial` (if ever enabled).

---

### Feature: Face recognition (`face_recognition`)

Event types differ by sector. The same model, different JSON keys.

#### Comercio / Industrial

```json
{ "type": "employee_seen",     "employee_id": "Juan Perez", "track_id": 15, "similarity": 0.87, "bbox": {...} }
{ "type": "employee_presence", "employee_id": "Juan Perez", "track_id": 15 }
{ "type": "employee_exit",     "employee_id": "Juan Perez", "track_id": 15, "dwell_seconds": 870.0 }
```

- `employee_seen`: first identification for this track. Open `EmployeeZoneInterval`.
- `employee_presence`: heartbeat every 30s. Update `last_heartbeat`.
- `employee_exit`: track lost. Close `EmployeeZoneInterval`.

If the Jetson restarts and `employee_exit` never arrives, estimate: `exit ≈ last_heartbeat + 30s`.

#### Hogar

```json
{ "type": "known_person_seen", "name": "Maria", "track_id": 3, "similarity": 0.91, "bbox": {...} }
{ "type": "known_person_exit", "name": "Maria", "track_id": 3, "dwell_seconds": 300.0 }
{ "type": "unknown_person_alert", "severity": "medium", "track_id": 7, "bbox": {...}, "face_snapshot_b64": "<jpg base64>" }
```

- `known_person_seen`: trigger "Maria arrived home" push notification.
- `unknown_person_alert`: trigger push with face photo. Severity `"medium"`.

---

### Feature: EPP detection (`epp_detection`) — model pending

```json
{
  "type": "epp_violation",
  "severity": "high",
  "track_id": 12,
  "violations": ["no_helmet", "no_vest"],
  "present": ["gloves"],
  "confidence": 0.87,
  "bbox": { "left": 80, "top": 150, "width": 70, "height": 200 }
}
```

---

### Feature: Fire/smoke detection (`fire_smoke`) — model pending

```json
{
  "type": "fire_smoke_alert",
  "severity": "critical",
  "detected": ["fire"],
  "confidence": 0.94,
  "frame_snapshot_b64": "<jpg base64>"
}
```

---

### Feature: License plate (`license_plate`) — model pending

```json
{
  "type": "vehicle_detected",
  "severity": "info",
  "track_id": 5,
  "plate": "ABC-1234",
  "plate_confidence": 0.93,
  "bbox": { "left": 200, "top": 600, "width": 120, "height": 40 }
}
```

---

## 5. Continuous telemetry

### `POST /api/analytics` — analytics snapshot (every 60s per camera, hogar: every 3600s)

```json
{
  "type": "analytics_snapshot",
  "sector": "comercio",
  "jetson_id": "jetson-mova-001",
  "camera_id": "jetson-mova-001-ch01",
  "timestamp": "2025-05-01T14:33:00Z",
  "severity": "info",
  "period_seconds": 60,
  "people_count": 24,
  "gender_male": 11,
  "gender_female": 13,
  "age_gender_classes": { "male_adult": 7, "female_adult": 8, "female_young": 5 }
}
```

For `industrial`: also includes `epp_compliant`, `epp_violations`, `employees_present` when those models are integrated.

### `WS /ws/positions` — position snapshot (every 10s per camera)

The Jetson connects as a WebSocket client at startup. Each message:

```json
{
  "type": "positions_snapshot",
  "sector": "comercio",
  "jetson_id": "jetson-mova-001",
  "camera_id": "jetson-mova-001-ch01",
  "timestamp": "2025-05-01T14:32:15Z",
  "positions": [
    { "track_id": 42, "x_norm": 0.45, "y_norm": 0.62 },
    { "track_id": 43, "x_norm": 0.21, "y_norm": 0.34 }
  ]
}
```

`x_norm`, `y_norm` ∈ [0, 1] relative to frame size. To map to pixels: `pixel_x = x_norm × frame_width`. Overlay on the `reference-frame` the pipeline sends at startup.

### `POST /api/cameras/reference-frame` — background image (once per camera at startup)

```json
{
  "camera_id": "jetson-mova-001-ch01",
  "jetson_id": "jetson-mova-001",
  "frame_num": 30,
  "timestamp": "2025-05-01T14:30:02Z",
  "image_b64": "<jpg base64>",
  "width": 1920,
  "height": 1080
}
```

Sent only when the frame has zero detections. Use as heatmap background for that camera.

### `POST /api/crops` — person crop for dataset (up to 5 per person)

```json
{
  "camera_id": "jetson-mova-001-ch01",
  "jetson_id": "jetson-mova-001",
  "track_id": 42,
  "frame_num": 150,
  "timestamp": "2025-05-01T14:32:05Z",
  "image_b64": "<jpg base64>",
  "bbox": { "left": 100, "top": 200, "width": 64, "height": 128 }
}
```

Used for building the re-ID training dataset. Minimum crop size: 64×128 px.

---

## 6. Backend data structures

### `ActivePerson`

```
global_person_id:    str         UUID. May be replaced by a RecentExit match after person_appearance.
jetson_id:           str
camera_id:           str         Current camera.
track_id:            int         Local to this camera. NOT globally unique.
vector:              float[512]  L2-normalized. None until person_appearance arrives.
identity:            str | None  Employee/family name if identified. None otherwise.
identity_confidence: float
demographics:        dict | None {gender, age_group, confidence}. None until person_classified.
first_seen:          datetime    Timestamp of person_entry.
last_seen:           datetime
is_entry_exit_cam:   bool

Index: Dict["jetson_id:camera_id:track_id"] → ActivePerson
```

### `RecentExit`

Buffer of persons who left recently. Used for cross-camera matching.

```
(all ActivePerson fields, plus:)
exit_time:           datetime    Timestamp of person_exit.
local_dwell_seconds: float       Time visible in that camera (from Jetson).

TTL: 120 seconds from exit_time. Purge expired entries periodically.
```

### `PersonSession`

One store visit. Created only for `is_entry_exit_camera: true` events.

```
global_person_id:  str
identity:          str | None
demographics:      dict | None
entry_time:        datetime    First person_entry with is_entry_exit_camera=true.
exit_time:         datetime | None  person_exit with is_entry_exit_camera=true. None if still inside.
dwell_seconds:     float | None  exit_time - entry_time.
cameras_visited:   list[str]   camera_ids in chronological order.

Dedup window: ignore re-entries within 5 minutes of the last session close for the same global_person_id.
```

### `EmployeeZoneInterval`

Presence interval for a known person in one camera zone.

```
employee_id:       str
camera_id:         str
track_id:          int
entry_time:        datetime    Timestamp of employee_seen / known_person_seen.
exit_time:         datetime | None
duration_seconds:  float | None
last_heartbeat:    datetime    Last employee_presence received.
```

---

## 7. Cross-camera matching algorithm

```python
REID_THRESHOLD = 0.65   # minimum cosine similarity
REID_WINDOW_S  = 120    # seconds a RecentExit remains matchable

def on_person_appearance(event):
    person = active_persons[key(event)]
    person.vector = event["appearance_vector"]

    # Search recent_exits from same client, different camera, within time window
    best, best_sim = None, -1.0
    for candidate in recent_exits:
        if candidate.jetson_id != event["jetson_id"]:
            continue  # or allow cross-Jetson if same client
        if candidate.camera_id == event["camera_id"]:
            continue
        if candidate.vector is None:
            continue
        if (datetime.utcnow() - candidate.exit_time).total_seconds() > REID_WINDOW_S:
            continue
        sim = float(np.dot(person.vector, candidate.vector))  # both L2-normalized → cosine sim
        if sim > best_sim:
            best_sim, best = sim, candidate

    if best and best_sim >= REID_THRESHOLD:
        # Same person — inherit identity and global_person_id
        person.global_person_id = best.global_person_id
        person.identity = best.identity
        # Merge open PersonSession if exists
        merge_sessions(person.global_person_id, event["camera_id"])
```

---

## 8. Severity → notification mapping

| Severity | Events | Action |
|----------|--------|--------|
| `critical` | `fire_smoke_alert`, `fall_detected` (hogar) | Push notification immediately |
| `high` | `fall_detected` (industrial), `epp_violation` | Push notification ≤ 30s |
| `medium` | `unknown_person_alert` | Push with face photo |
| `info` | `person_entry`, `person_classified`, `employee_seen`, `vehicle_detected`, `analytics_snapshot` | Dashboard only |

---

## 9. Deduplication

Use `event_id` (UUID4) as an idempotency key. If the Jetson retries due to network timeout, discard the duplicate by `event_id`. For `PersonSession`: use the 5-minute dedup window by `global_person_id` to avoid counting fast re-entries twice.

---

## 10. Disconnection handling

- **REST events**: the Jetson queues up to 512 events with retries. If the backend is down for several minutes, the oldest events are silently discarded.
- **WebSocket positions**: the Jetson reconnects automatically (exponential backoff 1s → 30s). Snapshots lost during disconnection are not recovered — position telemetry is best-effort.
- **Jetson restart mid-session**: local track IDs reset. New `person_entry` events will arrive for physically the same persons. With `person_appearance` vectors, the backend can re-link them to open `PersonSessions` via cross-camera matching if the appearance vectors are sufficiently similar.
