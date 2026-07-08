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

## 2. Endpoints the backend must expose

| Method | Path | Triggered by | Frequency |
|--------|------|-------------|-----------|
| `POST` | `/api/events` | Every person/alert event | Per event |
| `POST` | `/api/analytics` | Analytics snapshot | Every 60s per camera (hogar: every 3600s) |
| `POST` | `/api/crops` | Person crop for dataset | Up to 5 per person |
| `POST` | `/api/cameras/reference-frame` | Empty frame at startup | Once per camera per session |
| `WS`   | `/ws/positions` | Position telemetry | Persistent connection; message every 10s per camera |

All REST endpoints must return `HTTP 200` or `HTTP 201`. Any non-2xx response is logged as an error by the Jetson but **not retried** — only network failures and timeouts trigger retries.

---

## 3. Authentication


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

## 4. Important invariants

**`track_id` is local to one camera.** Two cameras on the same Jetson can both have `track_id=42` referring to completely different physical persons. The globally unique key is always the triplet `(jetson_id, camera_id, track_id)`.

**`bbox` coordinates are in pixels at the stream resolution** configured for that client (default 1920×1080, or 960×544 for substreams). The values come directly from the DeepStream tracker — they are not normalized.

**`person_appearance` may never arrive** for a given track if:
- The person was too small in frame (crop < 64×128 px)
- The OSNet model failed to load on this Jetson
- The person exited before the AppearanceWorker finished processing the crop

The backend must handle `ActivePerson.vector = None` permanently and still process `person_exit` correctly.

**Event ordering is guaranteed within a single camera** (`person_entry` always before `person_exit` for the same track). Cross-camera ordering is not guaranteed — `person_entry` on ch02 can arrive before `person_exit` on ch01 for the same physical person.

**One Jetson, multiple cameras:** a Jetson processes 1–16 cameras in parallel. Each camera sends its own independent stream of events. A person walking through multiple cameras generates separate `person_entry`/`person_exit` pairs per camera, linked only by the appearance vector match in the backend.

---

## 5. Typical event sequence per person

```
person_entry         ← immediately on first detection
  │
  ├─ [~1-2s later]
  │   person_appearance   ← appearance vector ready (may never arrive — see §5)
  │
  ├─ [once, when ≥10 samples]
  │   person_classified   ← age/gender (age_gender capability only)
  │
  ├─ [once, when face identified]
  │   employee_seen / known_person_seen   ← face_recognition only
  │
  ├─ [every 30s while visible]
  │   employee_presence   ← face_recognition, comercio/industrial only
  │
person_exit          ← when tracker loses person for >2s
  │
  └─ employee_exit / known_person_exit   ← face_recognition, same frame as person_exit
```

`person_classified` and employee events can arrive in any order after `person_entry`. They all share the same `(jetson_id, camera_id, track_id)` triplet.

---

## 6. Event types by feature

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
Sent ~1-2 seconds after `person_entry`, once the AppearanceWorker processed the crop. May never arrive — see §5.

```json
{
  "event_id": "uuid4",
  "type": "person_appearance",
  "sector": "comercio",
  "jetson_id": "jetson-mova-001",
  "camera_id": "jetson-mova-001-ch01",
  "timestamp": "2025-05-01T14:32:02.450Z",
  "severity": "info",
  "track_id": 42,
  "appearance_vector": [0.12, -0.34, 0.07, ...]
}
```

`appearance_vector`: 512 floats, L2-normalized. `dot(v1, v2)` = cosine similarity directly (no further normalization needed). This is **OSNet appearance similarity** (clothing/silhouette) — different from the ArcFace face similarity in `employee_seen`. Both are cosine similarities on [0, 1] but are not comparable between each other.

Joined to `person_entry` by `(jetson_id, camera_id, track_id)`.

#### `person_exit`
Fired when the tracker loses the person for > 2 seconds.

```json
{
  "event_id": "uuid4",
  "type": "person_exit",
  "sector": "comercio",
  "jetson_id": "jetson-mova-001",
  "camera_id": "jetson-mova-001-ch01",
  "timestamp": "2025-05-01T14:34:25.700Z",
  "severity": "info",
  "track_id": 42,
  "dwell_seconds": 145.3,
  "is_entry_exit_camera": true
}
```

`dwell_seconds` is the time this person was visible **in this specific camera**. For total store dwell time, use `PersonSession.exit_time - entry_time` (backend-calculated). If `is_entry_exit_channels` was empty (`[]`) on the Jetson, `is_entry_exit_camera` is always `false` — no `PersonSession` should be created.

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

All three events include the common fields (`event_id`, `sector`, `jetson_id`, `camera_id`, `timestamp`, `severity`). Only the extra fields are shown below for brevity.

```json
{ "type": "employee_seen",     "employee_id": "Juan Perez", "track_id": 15, "similarity": 0.87, "bbox": {...} }
{ "type": "employee_presence", "employee_id": "Juan Perez", "track_id": 15 }
{ "type": "employee_exit",     "employee_id": "Juan Perez", "track_id": 15, "dwell_seconds": 870.0 }
```

- `employee_seen`: first identification for this track in this camera. Open `EmployeeZoneInterval`. `similarity` is ArcFace face similarity (0–1).
- `employee_presence`: heartbeat every 30s while the employee remains visible. Update `last_heartbeat`. If the employee is visible in 2 cameras simultaneously, you will receive one `employee_presence` per camera per 30s — group by `employee_id`, not `track_id`.
- `employee_exit`: track lost (same timing as `person_exit`). Close `EmployeeZoneInterval`.

If the Jetson restarts and `employee_exit` never arrives, estimate: `exit ≈ last_heartbeat + 30s`.

#### Hogar

```json
{ "type": "known_person_seen", "name": "Maria", "track_id": 3, "similarity": 0.91, "bbox": {...} }
{ "type": "known_person_exit", "name": "Maria", "track_id": 3, "dwell_seconds": 300.0 }
{ "type": "unknown_person_alert", "severity": "medium", "track_id": 7, "bbox": {...}, "face_snapshot_b64": "<jpg base64>" }
```

- `known_person_seen`: trigger "Maria arrived home" push notification. Sent once per track per session.
- `unknown_person_alert`: triggered once per unknown person per track (no repeat for the same track). `face_snapshot_b64` is a JPEG crop of just the face region in base64 — smaller than a full person crop, typically 60–160 px wide. Severity `"medium"`.
- Note: there is **no `known_person_presence` heartbeat** for hogar — only `known_person_seen` and `known_person_exit`.

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

`people_count` is the **number of new `person_entry` events in this period** (not the number of persons currently visible). Each snapshot resets the counter. To get people per hour, sum `people_count` across snapshots for that hour.

`gender_male` / `gender_female` and `age_gender_classes` count persons who received a `person_classified` in this period. They will be lower than `people_count` (only persons with ≥10 voting samples are classified).

For `industrial`: also includes `epp_compliant`, `epp_violations`, `employees_present` when those models are integrated.

### `WS /ws/positions` — position snapshot (every 10s per camera)

> **Backend must implement a WebSocket server** at the path `/ws/positions`.
> The Jetson is the **client** — it opens and maintains the connection.
> The backend is the **server** — it accepts the connection and receives messages.
>
> Implementation checklist:
> - Accept WebSocket upgrade at `GET /ws/positions`
> - Validate `X-API-Key` header on handshake (same key used for REST)
> - Receive JSON text frames and parse them as `positions_snapshot` (see format below)
> - No need to send anything back to the Jetson — this is unidirectional (Jetson → backend)
> - A single Jetson opens one connection. Multiple Jetsons = multiple simultaneous connections.
> - The Jetson reconnects automatically if the connection drops; the backend just needs to accept new connections.

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
  "bbox": { "left": 100, "top": 200, "width": 64, "height": 128 },
  "global_id": "a1b2c3d4e5f6"
}
```

`global_id` is optional — omitted if ReID hasn't resolved an identity for this track yet. Used for building the re-ID training dataset. Minimum crop size: 64×128 px.

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

---

## 11. PersonTracker — central backend class

This class is the single point of state for all Jetson events. It owns `active_persons`, `recent_exits`, `open_sessions`, and `zone_intervals`.

```python
import uuid, time
import numpy as np
from datetime import datetime, timezone
from typing import Optional

REID_THRESHOLD  = 0.65   # cosine similarity (both vectors are L2-norm → just dot product)
REID_WINDOW_S   = 120    # seconds a RecentExit stays in the buffer
SESSION_DEDUP_S = 300    # ignore re-entry within 5 min of last session close


class PersonTracker:
    def __init__(self):
        self.active_persons: dict[str, ActivePerson]       = {}
        self.recent_exits:   list[RecentExit]              = []
        self.open_sessions:  dict[str, PersonSession]      = {}  # keyed by global_person_id
        self.zone_intervals: list[EmployeeZoneInterval]    = []

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _key(jetson_id: str, camera_id: str, track_id: int) -> str:
        return f"{jetson_id}:{camera_id}:{track_id}"

    def _purge_recent_exits(self):
        now = datetime.now(timezone.utc)
        self.recent_exits = [
            r for r in self.recent_exits
            if (now - r.exit_time).total_seconds() <= REID_WINDOW_S
        ]

    def find_match(self, vector: list[float], jetson_id: str, camera_id: str) -> Optional["RecentExit"]:
        """Return the best RecentExit candidate for cross-camera re-ID, or None."""
        v = np.array(vector, dtype=np.float32)
        self._purge_recent_exits()
        best, best_sim = None, -1.0
        for candidate in self.recent_exits:
            if candidate.jetson_id != jetson_id:
                continue  # same device; for multi-Jetson same client, remove this check
            if candidate.camera_id == camera_id:
                continue  # must be a different camera
            if candidate.vector is None:
                continue
            sim = float(np.dot(v, np.array(candidate.vector, dtype=np.float32)))
            if sim > best_sim:
                best_sim, best = sim, candidate
        return best if best and best_sim >= REID_THRESHOLD else None

    # ── event handlers ──────────────────────────────────────────────────────────

    def on_person_entry(self, event: dict):
        key = self._key(event["jetson_id"], event["camera_id"], event["track_id"])
        person = ActivePerson(
            global_person_id = str(uuid.uuid4()),
            jetson_id        = event["jetson_id"],
            camera_id        = event["camera_id"],
            track_id         = event["track_id"],
            vector           = None,
            identity         = None,
            identity_confidence = 0.0,
            demographics     = None,
            first_seen       = datetime.fromisoformat(event["timestamp"]),
            last_seen        = datetime.fromisoformat(event["timestamp"]),
            is_entry_exit_cam = event.get("is_entry_exit_camera", False),
        )
        self.active_persons[key] = person

        if person.is_entry_exit_cam:
            self._open_session(person)

    def on_person_appearance(self, event: dict):
        key = self._key(event["jetson_id"], event["camera_id"], event["track_id"])
        person = self.active_persons.get(key)
        if person is None:
            return  # person_entry may have been lost; ignore
        person.vector = event["appearance_vector"]

        match = self.find_match(person.vector, event["jetson_id"], event["camera_id"])
        if match:
            old_id = person.global_person_id
            person.global_person_id = match.global_person_id
            person.identity         = match.identity
            person.identity_confidence = match.identity_confidence
            if person.is_entry_exit_cam:
                self._merge_session(old_id, match.global_person_id, event["camera_id"])

        # Track camera path in the open session
        session = self.open_sessions.get(person.global_person_id)
        if session and event["camera_id"] not in session.cameras_visited:
            session.cameras_visited.append(event["camera_id"])

    def on_person_classified(self, event: dict):
        key = self._key(event["jetson_id"], event["camera_id"], event["track_id"])
        person = self.active_persons.get(key)
        if person is None:
            return
        person.demographics = event.get("demographics")
        session = self.open_sessions.get(person.global_person_id)
        if session:
            session.demographics = person.demographics

    def on_person_exit(self, event: dict):
        key = self._key(event["jetson_id"], event["camera_id"], event["track_id"])
        person = self.active_persons.pop(key, None)
        if person is None:
            return

        # Close the store session if this was an entry/exit camera
        if event.get("is_entry_exit_camera"):
            self._close_session(person.global_person_id, event["timestamp"])

        # Move to recent_exits for cross-camera matching
        recent = RecentExit(
            **person.__dict__,
            exit_time           = datetime.fromisoformat(event["timestamp"]),
            local_dwell_seconds = event.get("dwell_seconds", 0.0),
        )
        self.recent_exits.append(recent)

    def on_employee_seen(self, event: dict):
        """Handles employee_seen (comercio/industrial) and known_person_seen (hogar)."""
        key = self._key(event["jetson_id"], event["camera_id"], event["track_id"])
        person = self.active_persons.get(key)
        name = event.get("employee_id") or event.get("name", "")
        if person:
            person.identity = name
            person.identity_confidence = event.get("similarity", 0.0)
            # Propagate identity to all active persons with the same global_person_id
            for p in self.active_persons.values():
                if p.global_person_id == person.global_person_id and p.identity is None:
                    p.identity = name
                    p.identity_confidence = person.identity_confidence

        self.zone_intervals.append(EmployeeZoneInterval(
            employee_id  = name,
            camera_id    = event["camera_id"],
            track_id     = event["track_id"],
            entry_time   = datetime.fromisoformat(event["timestamp"]),
            exit_time    = None,
            duration_seconds = None,
            last_heartbeat   = datetime.fromisoformat(event["timestamp"]),
        ))

    def on_employee_presence(self, event: dict):
        """Handles employee_presence (comercio/industrial). Hogar has no equivalent."""
        name = event.get("employee_id", "")
        track_id = event["track_id"]
        for interval in reversed(self.zone_intervals):
            if interval.employee_id == name and interval.track_id == track_id and interval.exit_time is None:
                interval.last_heartbeat = datetime.fromisoformat(event["timestamp"])
                break

    def on_employee_exit(self, event: dict):
        """Handles employee_exit (comercio/industrial) and known_person_exit (hogar)."""
        name = event.get("employee_id") or event.get("name", "")
        track_id = event["track_id"]
        exit_ts = datetime.fromisoformat(event["timestamp"])
        for interval in reversed(self.zone_intervals):
            if interval.employee_id == name and interval.track_id == track_id and interval.exit_time is None:
                interval.exit_time       = exit_ts
                interval.duration_seconds = event.get("dwell_seconds",
                    (exit_ts - interval.entry_time).total_seconds())
                break

    # ── session helpers ─────────────────────────────────────────────────────────

    def _open_session(self, person: "ActivePerson"):
        now = person.first_seen
        existing = self.open_sessions.get(person.global_person_id)
        if existing and existing.exit_time:
            if (now - existing.exit_time).total_seconds() < SESSION_DEDUP_S:
                return  # re-entry too fast — don't count as a new visit
        self.open_sessions[person.global_person_id] = PersonSession(
            global_person_id = person.global_person_id,
            identity         = person.identity,
            demographics     = person.demographics,
            entry_time       = now,
            exit_time        = None,
            dwell_seconds    = None,
            cameras_visited  = [person.camera_id],
        )

    def _close_session(self, global_person_id: str, timestamp_iso: str):
        session = self.open_sessions.get(global_person_id)
        if session and session.exit_time is None:
            session.exit_time     = datetime.fromisoformat(timestamp_iso)
            session.dwell_seconds = (session.exit_time - session.entry_time).total_seconds()

    def _merge_session(self, old_id: str, new_id: str, camera_id: str):
        """After re-ID match: merge the new entry into the existing session."""
        old_session = self.open_sessions.pop(old_id, None)
        existing    = self.open_sessions.get(new_id)
        if existing:
            if camera_id not in existing.cameras_visited:
                existing.cameras_visited.append(camera_id)
        elif old_session:
            old_session.global_person_id = new_id
            self.open_sessions[new_id] = old_session
```

**Important**: `on_person_entry` creates the `ActivePerson` immediately. `on_person_appearance` may arrive 1–3 seconds later and is the point where cross-camera matching happens. Do not wait for `person_appearance` before creating the `PersonSession` — just update it in place when the match is found.

---

## 12. Business calculations

### Store dwell time

```python
# From PersonSession — reliable cross-camera total
dwell_seconds = (session.exit_time - session.entry_time).total_seconds()

# The Jetson also sends dwell_seconds per track in person_exit,
# but that covers only one camera. Use backend timestamps for the
# true cross-camera total.
```

### Unique visitor count

```python
# Count closed PersonSessions per time period (day/hour)
visits = [
    s for s in open_sessions.values()
    if s.exit_time and period_start <= s.exit_time <= period_end
]
unique_visitors = len(visits)  # already deduplicated by global_person_id + 5 min window
```

### People per hour (analytics snapshot)

```python
# analytics_snapshot.people_count = new person_entry events in that period
# (it resets each snapshot — do NOT cumulate blindly)
people_per_hour = sum(
    snap["people_count"]
    for snap in analytics_snapshots
    if camera_id == snap["camera_id"]
    and hour_start <= snap["timestamp"] <= hour_end
)
```

### Heatmap pixel mapping

```python
# Positions arrive as normalized coords relative to the camera frame
# reference_frame for that camera has width/height from POST /api/cameras/reference-frame
pixel_x = round(pos["x_norm"] * frame_width)
pixel_y = round(pos["y_norm"] * frame_height)

# Accumulate into a NxM grid for the heatmap
grid[grid_y][grid_x] += 1

# Normalize by time window to make periods comparable
heatmap_value = grid[y][x] / total_snapshots_in_period
```

### Employee zone timeline

```python
# Build a per-employee ordered timeline from EmployeeZoneInterval
timeline = sorted(
    [i for i in zone_intervals if i.employee_id == target_employee],
    key=lambda i: i.entry_time,
)
# Detect zone absence alert: last known interval ended > threshold ago
last = timeline[-1] if timeline else None
if last and last.exit_time is None:
    # Employee still in zone — check heartbeat for crash detection
    since_heartbeat = (datetime.now(timezone.utc) - last.last_heartbeat).total_seconds()
    if since_heartbeat > 90:  # missed 3 heartbeats — assume Jetson crashed
        estimated_exit = last.last_heartbeat + timedelta(seconds=30)
```

### Identity propagation across cameras

When `on_employee_seen` fires for one camera, the identity spreads to all `ActivePerson` records with the same `global_person_id`:

```python
for p in self.active_persons.values():
    if p.global_person_id == person.global_person_id and p.identity is None:
        p.identity = name
```

This means even if the employee's face is not directly visible in another camera, the backend knows who they are via the appearance match.

---

## 13. Face recognition + appearance vector — how they complement each other

The pipeline uses **two distinct embedding models** for person identification:

| Model | Library | Embedding | What it encodes | Used for |
|-------|---------|-----------|-----------------|---------|
| ArcFace (buffalo_l) | InsightFace | 512-dim | Face identity | Recognizing WHO a person is |
| OSNet-x0.25 | ONNX | 512-dim L2-norm | Body appearance (clothing, silhouette) | Cross-camera re-ID |

These two systems are **independent** and **complementary**:

### ArcFace — WHO is this person?

- Runs in `FaceRecognizer` worker (Python thread, async)
- Compares face crop against `known_faces.json` (registered employees or family members)
- `similarity` in `employee_seen` / `known_person_seen` is the ArcFace cosine similarity (0–1)
  - Typical threshold for positive ID: 0.45–0.50 (InsightFace default)
  - This value is **not comparable** to the OSNet similarity — different model, different distribution
- **Does not run on strangers** in comercio/industrial — only in hogar (`unknown_person_alert`)
- Requires face to be reasonably visible and ≥ 20×20 px (PeopleNet class 2 detection)

### OSNet — is this the same BODY I saw before?

- Runs in `AppearanceWorker` worker (Python thread, async)
- No registration required — works on anyone
- Generates a 512-dim vector from the full-body crop (64×128 px minimum)
- L2-normalized at inference time → cosine similarity = `np.dot(v1, v2)`
- Typical re-ID threshold: 0.65 (tune based on environment)
- `appearance_vector` in `person_appearance` is this vector — **NOT** an ArcFace embedding

### How they work together

```
Person enters camera 1
  │
  ├─ person_entry (immediate)
  │
  ├─ AppearanceWorker: body crop → OSNet → 512-dim vector
  │       → person_appearance (1–3s later)
  │             → backend: compare vs recent_exits → assign global_person_id
  │
  └─ FaceRecognizer: face crop → ArcFace → compare vs known_faces.json
          → employee_seen / known_person_seen (if face found and recognized)
                → backend: assign identity, propagate via global_person_id
                           to all cameras where this person is active

Person moves to camera 2 (face not visible this angle)
  │
  ├─ person_entry (new track_id, camera 2)
  ├─ person_appearance → OSNet match → inherit global_person_id from camera 1
  │                                      backend already knows: this is Juan Perez
  └─ NO employee_seen (face not visible) — but identity already known from camera 1
```

**Key rule**: `employee_id` / `name` on `employee_seen` always identifies the person for that `track_id`. The backend propagates it to all `ActivePerson` records with the same `global_person_id`. Once identity is known, the appearance vector keeps tracking them cross-camera without needing to see the face again.

### Face recognition event types by sector

| Sector | Recognized person → | Unknown person → |
|--------|---------------------|------------------|
| `comercio` | `employee_seen`, `employee_presence`, `employee_exit` | no event |
| `industrial` | `employee_seen`, `employee_presence`, `employee_exit` | no event |
| `hogar` | `known_person_seen`, `known_person_exit` | `unknown_person_alert` (with `face_snapshot_b64`) |

For comercio/industrial, unrecognized persons are tracked via people_counting only — no alert. For hogar, any unrecognized face in the face-detection zone triggers an alert with a photo.

### Similarity thresholds (summary)

| Model | Field | Typical threshold | Notes |
|-------|-------|------------------|-------|
| ArcFace (InsightFace) | `similarity` in `employee_seen` | ≥ 0.45 | Jetson already applies the threshold — only sends event if confident |
| OSNet (re-ID) | `appearance_vector` dot product | ≥ 0.65 | Backend applies in `find_match()` |

The Jetson never sends `employee_seen` below its own ArcFace threshold. The backend only needs to apply the OSNet threshold when calling `find_match()`.
