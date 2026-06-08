# NX Computing AI — Backend Integration Guide

**Audience:** Backend developer receiving data from Jetson devices.
**Purpose:** Complete reference of every payload the Jetson sends + concrete recipes for turning that data into business analytics.

> This document is the source of truth for the Jetson ↔ Backend contract.
> **Update it whenever a payload field, endpoint, or event type changes.**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Authentication](#2-authentication)
3. [Endpoints Reference](#3-endpoints-reference)
4. [Event Types by Feature](#4-event-types-by-feature)
5. [Continuous Telemetry](#5-continuous-telemetry)
6. [Core Data Structures](#6-core-data-structures)
7. [Analytics Recipes](#7-analytics-recipes)
8. [Cross-Camera Matching](#8-cross-camera-matching)
9. [Notifications & Severity](#9-notifications--severity)
10. [Reliability & Edge Cases](#10-reliability--edge-cases)

---

## 1. System Overview

```
┌──────────────────────────────┐         ┌─────────────────────────┐
│  Jetson Orin Nano (on-site)  │         │  Backend (cloud)        │
│                              │         │                         │
│  DeepStream pipeline         │─POST──▶│  /api/events            │
│  • PeopleNet (detect)        │─POST──▶│  /api/analytics         │
│  • NvDCF tracker             │─POST──▶│  /api/crops             │
│  • OSNet (body re-ID)        │─POST──▶│  /api/cameras/reference │
│  • ArcFace (face ID)         │─WS──▶  │  /ws/positions          │
│  • MoveNet (fall)            │         │                         │
└──────────────────────────────┘         └─────────────────────────┘
```

**What the Jetson does:**
- Detects and tracks persons frame-by-frame
- Classifies demographics (age/gender), identity (face), pose (fall)
- Runs on-device cross-camera re-ID (OSNet) and emits `global_id`
- Sends REST events asynchronously (fire-and-forget queue, max 512 items)
- Streams normalized positions via WebSocket every 10 seconds

**What the Jetson does NOT do:**
- Render heatmaps (it sends raw position data)
- Compute visit history across days
- Send push notifications
- Store any analytics — it is stateless across restarts

**What the backend must do:**
- Maintain `ActivePerson` state per `(jetson_id, camera_id, track_id)`
- Build `PersonSession` records (store visits) for entry/exit cameras
- Compute all business metrics (dwell time averages, traffic trends, demographics)
- Trigger push notifications on `critical`/`high`/`medium` severity events
- Persist everything — the Jetson's memory is ephemeral

---

## 2. Authentication

Every REST request and WebSocket handshake includes:

```
X-API-Key: <api_key>
Content-Type: application/json
```

Return `HTTP 200` or `201` for success. Any non-2xx is logged by the Jetson but **not retried** — only network failures trigger retries.

---

## 3. Endpoints Reference

| Method | Path | Purpose | Frequency |
|--------|------|---------|-----------|
| `POST` | `/api/events` | Every person/alert event | Per event |
| `POST` | `/api/analytics` | Aggregated snapshot per camera | Every 60s (hogar: 3600s) |
| `POST` | `/api/crops` | Person image crop for dataset | Up to 5 per track |
| `POST` | `/api/cameras/reference-frame` | Empty background frame | On startup (retry every 30 s until 2xx) + on scene change (≥15 % diff, min 24 h interval) |
| `WS` | `/ws/positions` | Real-time position telemetry | Persistent; message every 10s/camera |

### Common fields on every `/api/events` payload

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `event_id` | UUID4 string | `"a1b2-..."` | Idempotency key — deduplicate on this |
| `type` | string | `"person_entry"` | See §4 for all types |
| `sector` | string | `"comercio"` | `comercio` \| `industrial` \| `hogar` |
| `jetson_id` | string | `"jetson-mova-001"` | Device identifier |
| `camera_id` | string | `"jetson-mova-001-ch01"` | Format: `{jetson_id}-ch{N:02d}` |
| `timestamp` | ISO 8601 UTC | `"2025-05-01T14:32:00.123Z"` | Event time on device |
| `severity` | string | `"info"` | `info` \| `medium` \| `high` \| `critical` |

**`track_id` is LOCAL to one camera.** Two cameras can both have `track_id=42` for different persons.
The globally unique key is always `(jetson_id, camera_id, track_id)`.

**`global_id`** (optional field on some events): when present, it was assigned by the Jetson's on-device re-ID (OSNet). It links the same physical person across cameras within the same Jetson session. Use it to merge `PersonSession` records without waiting for `person_appearance`.

---

## 4. Event Types by Feature

---

### 4.1 People Counting (`people_counting` — always active)

#### `person_entry`

Fired the first time the tracker sees a person in this camera.

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "type": "person_entry",
  "sector": "comercio",
  "jetson_id": "jetson-mova-001",
  "camera_id": "jetson-mova-001-ch01",
  "timestamp": "2025-05-01T14:32:00.123Z",
  "severity": "info",
  "track_id": 42,
  "bbox": { "left": 100, "top": 200, "width": 60, "height": 180 },
  "confidence": 0.92,
  "is_entry_exit_camera": true,
  "entry_type": "new",
  "global_id": "d3f1a2b4-..."
}
```

| Field | Notes |
|-------|-------|
| `is_entry_exit_camera` | `true` → this camera covers an entrance/exit. Create a `PersonSession` for store visit tracking. |
| `entry_type` | `"new"` → person never seen before. `"return"` → same person returned after > 5 min absence (on-device re-ID recognized them). |
| `global_id` | Present when on-device re-ID matched this person across cameras. Use to link to an existing `PersonSession`. Absent if re-ID is not running or this is a truly new person. |

#### `person_exit`

Fired when the tracker loses the person for > 2 seconds.

```json
{
  "type": "person_exit",
  "track_id": 42,
  "dwell_seconds": 145.3,
  "is_entry_exit_camera": true,
  "global_id": "d3f1a2b4-..."
}
```

`dwell_seconds` = time visible **in this specific camera only**. For total visit duration, use `PersonSession.exit_time - entry_time` (backend-computed).

#### `person_channel_change`

Fired by the Jetson's on-device re-ID when the same person moves from one camera to another **within 5 minutes**.

```json
{
  "type": "person_channel_change",
  "track_id": 7,
  "bbox": { "left": 320, "top": 150, "width": 55, "height": 175 },
  "confidence": 0.88,
  "global_id": "d3f1a2b4-...",
  "prev_camera_id": "jetson-mova-001-ch01",
  "is_entry_exit_camera": false
}
```

This event fires instead of a plain `person_entry` when the Jetson is confident it's the same person. Update the `PersonSession.cameras_visited` list without counting as a new unique visitor.

#### `person_appearance`

Sent ~1–3 seconds after `person_entry`, once the OSNet AppearanceWorker processes the body crop.

```json
{
  "type": "person_appearance",
  "track_id": 42,
  "appearance_vector": [0.12, -0.34, 0.07, "... 512 floats total ..."]
}
```

`appearance_vector`: 512 floats, L2-normalized. `np.dot(v1, v2)` = cosine similarity directly. **May never arrive** if the person is too small (< 64×128 px crop) or exits before the worker finishes. The backend must handle `person.vector = None` at all times.

---

### 4.2 Age/Gender Classification (`age_gender`)

#### `person_classified`

Sent once per person after accumulating ≥ 10 voting samples from the SGIE. Never arrives if the person exits too quickly.

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

`label` is one of: `female_young`, `female_adult`, `female_senior`, `male_young`, `male_adult`, `male_senior`.

Store on `ActivePerson.demographics`. Propagate to `PersonSession.demographics`.

---

### 4.3 Face Recognition (`face_recognition`)

Two model pairs run independently:
- **ArcFace** (InsightFace buffalo_l): identifies WHO a person is against `known_faces.json`
- **OSNet** (appearance_worker): identifies WHICH body this is cross-camera (§4.1)

Event names differ by sector:

| Sector | Recognized | Heartbeat | Exit | Unknown |
|--------|-----------|-----------|------|---------|
| `comercio` | `employee_seen` | `employee_presence` | `employee_exit` | (no event) |
| `industrial` | `employee_seen` | `employee_presence` | `employee_exit` | (no event) |
| `hogar` | `known_person_seen` | (none) | `known_person_exit` | `unknown_person_alert` |

#### `employee_seen` / `known_person_seen`

Fired on first face identification for this track in this camera session.

```json
{
  "type": "employee_seen",
  "track_id": 15,
  "employee_id": "Juan Perez",
  "similarity": 0.87,
  "bbox": { "left": 210, "top": 100, "width": 55, "height": 165 }
}
```

```json
{
  "type": "known_person_seen",
  "track_id": 3,
  "name": "Maria",
  "similarity": 0.91,
  "bbox": { "left": 140, "top": 120, "width": 50, "height": 160 }
}
```

`similarity` = ArcFace cosine similarity (0–1). Jetson already applied its threshold (≥ 0.50) — events below that are never sent.

Open an `EmployeeZoneInterval`. For hogar `known_person_seen`, trigger "Maria arrived home" push notification.

#### `employee_presence`

Heartbeat every 30s while the employee is visible. No equivalent for hogar.

```json
{
  "type": "employee_presence",
  "track_id": 15,
  "employee_id": "Juan Perez"
}
```

Update `EmployeeZoneInterval.last_heartbeat`. If missed for > 90s, assume the Jetson crashed — estimate `exit ≈ last_heartbeat + 30s`.

#### `employee_exit` / `known_person_exit`

```json
{ "type": "employee_exit",    "employee_id": "Juan Perez", "track_id": 15, "dwell_seconds": 870.0 }
{ "type": "known_person_exit","name": "Maria",             "track_id": 3,  "dwell_seconds": 300.0 }
```

Close the `EmployeeZoneInterval`.

#### `unknown_person_alert` (hogar only)

```json
{
  "type": "unknown_person_alert",
  "severity": "medium",
  "track_id": 7,
  "bbox": { "left": 80, "top": 130, "width": 65, "height": 185 },
  "face_snapshot_b64": "<jpeg base64, ~60–160px wide>"
}
```

Triggered once per unrecognized person per track. Push notification with face photo.

---

## 5. Continuous Telemetry

### `POST /api/analytics` — aggregated snapshot

Sent every 60 seconds per camera (hogar: every 3600s). Counters reset after each send.

```json
{
  "event_id": "uuid4",
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
  "age_gender_classes": {
    "male_adult": 7,
    "female_adult": 8,
    "female_young": 5
  }
}
```

`people_count` = new `person_entry` events in this period (not persons currently visible).
`gender_male` / `gender_female` / `age_gender_classes` count only persons who received `person_classified` in this period — always ≤ `people_count`.

---

### `WS /ws/positions` — real-time position telemetry

The Jetson is the **WebSocket client**. The backend is the **server**.

Implementation checklist:
- Accept WebSocket upgrade at `GET /ws/positions`
- Validate `X-API-Key` on handshake
- Receive JSON text frames; parse as `positions_snapshot`
- No need to send back anything — unidirectional

Message format (sent every 10s per camera):

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

`x_norm`, `y_norm` ∈ [0, 1] relative to the camera's frame size. Map to pixels with:
```python
pixel_x = round(x_norm * frame_width)
pixel_y = round(y_norm * frame_height)
```

`frame_width` / `frame_height` come from the `POST /api/cameras/reference-frame` payload for that `camera_id`.

---

### `POST /api/cameras/reference-frame` — background image

Sent when the scene is empty (zero detections) and one of the following conditions is met:

1. **Startup / retry**: no confirmed frame yet for this camera — retried every 30 s until the backend returns 2xx.
2. **Scene change detected**: at least 24 h have elapsed since the last confirmed frame AND the current frame differs ≥ 15 % from the stored baseline (after normalizing for illumination), indicating a physical layout change (e.g. products rearranged).

```json
{
  "camera_id": "jetson-mova-001-ch01",
  "jetson_id": "jetson-mova-001",
  "frame_num": 30,
  "timestamp": "2025-05-01T14:30:02Z",
  "image_b64": "<jpeg base64>",
  "width": 1920,
  "height": 1080
}
```

The backend **inserts** a new row on every call (no UPSERT). Historical heatmap queries use `timestamp <= query_end ORDER BY timestamp DESC LIMIT 1` to find the background that was valid at any point in time. All normalized positions from `/ws/positions` are relative to the `width × height` of the frame valid for that period.

---

### `POST /api/crops` — person image crop

Up to 5 crops per person, minimum size 64×128 px.

```json
{
  "camera_id": "jetson-mova-001-ch01",
  "jetson_id": "jetson-mova-001",
  "track_id": 42,
  "frame_num": 150,
  "timestamp": "2025-05-01T14:32:05Z",
  "image_b64": "<jpeg base64>",
  "bbox": { "left": 100, "top": 200, "width": 64, "height": 128 }
}
```

Used for building the re-ID training dataset and for manual review.

---

## 6. Core Data Structures

These are the backend's in-memory records. Persist to your database after each update.

### `ActivePerson`

One record per live track. Key: `"{jetson_id}:{camera_id}:{track_id}"`.

```python
@dataclass
class ActivePerson:
    global_person_id:    str           # backend UUID; may be updated by re-ID match
    jetson_id:           str
    camera_id:           str
    track_id:            int           # local to camera; NOT globally unique
    vector:              list | None   # OSNet 512-dim; None until person_appearance arrives
    identity:            str | None    # employee name / person name if recognized
    identity_confidence: float         # ArcFace similarity
    demographics:        dict | None   # {gender, age_group, label, confidence}
    first_seen:          datetime      # person_entry timestamp
    last_seen:           datetime      # updated on each event for this track
    is_entry_exit_cam:   bool
```

### `RecentExit`

Buffer of persons who left recently. Used for cross-camera re-ID matching. TTL: 120s from `exit_time`.

```python
@dataclass
class RecentExit(ActivePerson):
    exit_time:           datetime
    local_dwell_seconds: float    # from Jetson's person_exit.dwell_seconds
```

### `PersonSession`

One store visit. Created only for `is_entry_exit_camera: true` cameras.

```python
@dataclass
class PersonSession:
    global_person_id:  str
    identity:          str | None
    demographics:      dict | None
    entry_time:        datetime        # first person_entry with is_entry_exit_camera=true
    exit_time:         datetime | None # person_exit with is_entry_exit_camera=true
    dwell_seconds:     float | None    # exit_time - entry_time (backend-computed)
    cameras_visited:   list[str]       # camera_ids in chronological order
```

Dedup rule: ignore re-entry within 5 minutes of last session close for the same `global_person_id`.

### `EmployeeZoneInterval`

One continuous presence interval for a known person in one camera zone.

```python
@dataclass
class EmployeeZoneInterval:
    employee_id:      str
    camera_id:        str
    track_id:         int
    entry_time:       datetime
    exit_time:        datetime | None
    duration_seconds: float | None
    last_heartbeat:   datetime   # updated by employee_presence every 30s
```

---

## 7. Analytics Recipes

This section shows how to compute each business metric from the raw events above.

---

### 7.1 Heatmap (Foot Traffic Density)

**What it shows:** Where in the camera frame people spend the most time.

**Data source:** `POST /api/cameras/reference-frame` (background image) + `WS /ws/positions` (normalized positions).

```python
import numpy as np
from PIL import Image
import base64, io

# Load background from reference-frame
def load_reference(payload: dict) -> tuple[np.ndarray, int, int]:
    img_bytes = base64.b64decode(payload["image_b64"])
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img), payload["width"], payload["height"]

# Accumulate position counts in a grid
GRID_COLS, GRID_ROWS = 64, 36   # adjust resolution as needed
grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

def on_positions_snapshot(msg: dict, frame_width: int, frame_height: int):
    for pos in msg["positions"]:
        # Map normalized [0,1] to grid cell
        col = min(int(pos["x_norm"] * GRID_COLS), GRID_COLS - 1)
        row = min(int(pos["y_norm"] * GRID_ROWS), GRID_ROWS - 1)
        grid[row][col] += 1

# Render heatmap overlay on reference frame
def render_heatmap(background: np.ndarray, grid: np.ndarray) -> np.ndarray:
    import cv2
    norm = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colormap = cv2.applyColorMap(cv2.resize(norm, (background.shape[1], background.shape[0])),
                                  cv2.COLORMAP_JET)
    return cv2.addWeighted(background, 0.6, colormap, 0.4, 0)
```

**Per-time-window heatmaps:** reset `grid` at the start of each time window (e.g., hour) to compare morning vs. afternoon traffic.

**Bbox-based heatmap (alternative):** use `bbox` from `person_entry` / `person_classified` if position telemetry is not running. Map bbox center to the grid: `center_x = bbox["left"] + bbox["width"] / 2`.

---

### 7.2 Average Dwell Time

**What it shows:** How long visitors spend in the store on average.

**Data source:** `PersonSession.dwell_seconds` (backend-computed) from `person_entry` + `person_exit` on entry/exit cameras.

```python
from datetime import datetime, timezone
from statistics import mean, median

def compute_dwell_stats(sessions: list[PersonSession], 
                         period_start: datetime, 
                         period_end: datetime) -> dict:
    dwells = [
        s.dwell_seconds
        for s in sessions
        if s.exit_time
        and period_start <= s.exit_time <= period_end
        and s.dwell_seconds is not None
        and s.dwell_seconds > 0
    ]
    if not dwells:
        return {"avg_seconds": 0, "median_seconds": 0, "count": 0}
    return {
        "avg_seconds":    round(mean(dwells), 1),
        "median_seconds": round(median(dwells), 1),
        "min_seconds":    round(min(dwells), 1),
        "max_seconds":    round(max(dwells), 1),
        "count":          len(dwells),
    }
```

**Note:** use `PersonSession` (backend-computed, entry_time to exit_time) rather than `person_exit.dwell_seconds` (Jetson-computed, single-camera only). The session dwell is the true cross-camera visit duration.

**Segmented by demographics:** filter sessions where `session.demographics["gender"] == "female"` to get average dwell by gender.

---

### 7.3 Unique Visitor Count & Traffic Volume

**What it shows:** How many unique people visited in a period, and how many total entries.

**Data source:** `PersonSession` (unique visitors) + `analytics_snapshot.people_count` (total entries).

```python
def unique_visitors(sessions: list[PersonSession], 
                    period_start: datetime, period_end: datetime) -> int:
    return sum(
        1 for s in sessions
        if s.exit_time and period_start <= s.exit_time <= period_end
    )

def total_entries_from_snapshots(snapshots: list[dict],
                                  camera_id: str,
                                  period_start: datetime, period_end: datetime) -> int:
    return sum(
        s["people_count"]
        for s in snapshots
        if s["camera_id"] == camera_id
        and period_start <= datetime.fromisoformat(s["timestamp"]) <= period_end
    )

# People per hour bar chart
def entries_per_hour(snapshots: list[dict], camera_id: str, date: str) -> dict[int, int]:
    hourly = {h: 0 for h in range(24)}
    for s in snapshots:
        if s["camera_id"] != camera_id:
            continue
        ts = datetime.fromisoformat(s["timestamp"])
        if ts.date().isoformat() == date:
            hourly[ts.hour] += s["people_count"]
    return hourly
```

---

### 7.4 Demographic Breakdown

**What it shows:** Gender and age group distribution of visitors.

**Data source:** `analytics_snapshot.age_gender_classes` (aggregated) + individual `person_classified` events (per-person detail).

```python
from collections import defaultdict

def demographic_report(snapshots: list[dict], camera_id: str,
                        period_start: datetime, period_end: datetime) -> dict:
    totals = defaultdict(int)
    for s in snapshots:
        if s["camera_id"] != camera_id:
            continue
        ts = datetime.fromisoformat(s["timestamp"])
        if not (period_start <= ts <= period_end):
            continue
        for label, count in s.get("age_gender_classes", {}).items():
            totals[label] += count

    total_classified = sum(totals.values())
    return {
        label: {
            "count": count,
            "pct": round(count / total_classified * 100, 1) if total_classified else 0
        }
        for label, count in sorted(totals.items())
    }

# Example output:
# {
#   "female_adult":  {"count": 42, "pct": 35.0},
#   "female_young":  {"count": 18, "pct": 15.0},
#   "male_adult":    {"count": 38, "pct": 31.7},
#   ...
# }
```

**Conversion funnel by demographic:** correlate `person_classified` with `PersonSession.dwell_seconds` to see which demographic stays longer.

---

### 7.5 Peak Hour Analysis

**What it shows:** Which hours of the day have the highest traffic.

**Data source:** `analytics_snapshot.people_count` — one snapshot per camera every 60s.

```python
import pandas as pd

def peak_hour_table(snapshots: list[dict], camera_id: str) -> pd.DataFrame:
    rows = [
        {
            "hour": datetime.fromisoformat(s["timestamp"]).hour,
            "weekday": datetime.fromisoformat(s["timestamp"]).weekday(),
            "people": s["people_count"],
        }
        for s in snapshots if s["camera_id"] == camera_id
    ]
    df = pd.DataFrame(rows)
    return df.groupby(["weekday", "hour"])["people"].sum().reset_index()
```

Use the resulting table to render a day-of-week × hour heatmap for operational planning (staffing, promotions).

---

### 7.6 Cross-Camera Journey Map

**What it shows:** How people move through the physical space (which cameras they appear in, in what order).

**Data source:** `PersonSession.cameras_visited` (built from `person_appearance` re-ID matches or `person_channel_change` events).

```python
from collections import Counter

def camera_transition_matrix(sessions: list[PersonSession]) -> dict[tuple, int]:
    transitions = Counter()
    for session in sessions:
        path = session.cameras_visited
        for i in range(len(path) - 1):
            transitions[(path[i], path[i+1])] += 1
    return dict(transitions)

# Example: {("ch01", "ch02"): 142, ("ch02", "ch03"): 87}
# Visualize as a Sankey diagram or directed graph
```

**Bottleneck detection:** cameras with high `in-count` but low `out-count` indicate areas where people spend extra time (or where there's congestion).

---

### 7.7 Employee Zone Timeline

**What it shows:** Which areas each employee was in, and for how long.

**Data source:** `EmployeeZoneInterval` records (from `employee_seen`, `employee_presence`, `employee_exit`).

```python
def employee_timeline(intervals: list[EmployeeZoneInterval], 
                       employee_id: str) -> list[dict]:
    return sorted(
        [
            {
                "camera_id":       i.camera_id,
                "entry_time":      i.entry_time.isoformat(),
                "exit_time":       i.exit_time.isoformat() if i.exit_time else None,
                "duration_min":    round(i.duration_seconds / 60, 1) if i.duration_seconds else None,
            }
            for i in intervals if i.employee_id == employee_id
        ],
        key=lambda x: x["entry_time"],
    )

def detect_missed_heartbeat(interval: EmployeeZoneInterval, 
                             now: datetime, threshold_s: float = 90) -> bool:
    if interval.exit_time is not None:
        return False
    since = (now - interval.last_heartbeat).total_seconds()
    return since > threshold_s   # Jetson likely crashed; estimate exit
```

---

### 7.8 Unknown Person Alert (Hogar)

**What it shows:** Timeline of unrecognized faces detected at the front door or secure zones.

**Data source:** `unknown_person_alert` events.

```python
def on_unknown_person_alert(event: dict):
    alert = {
        "alert_id":          event["event_id"],
        "timestamp":         event["timestamp"],
        "camera_id":         event["camera_id"],
        "track_id":          event["track_id"],
        "face_snapshot_b64": event["face_snapshot_b64"],
        "bbox":              event["bbox"],
    }
    db.save_alert(alert)
    push_service.send(
        title="Persona desconocida detectada",
        body=f"Cámara {event['camera_id']} — {event['timestamp']}",
        image_b64=event["face_snapshot_b64"],
        severity="medium",
    )
```

---

## 8. Cross-Camera Matching

Two re-ID layers work together:

| Layer | Where | How | When |
|-------|-------|-----|------|
| On-device (Jetson) | Inside Jetson | OSNet cosine similarity ≥ 0.65, 5-min window | `global_id` in `person_entry` / `person_channel_change` |
| Backend | Backend server | Same algorithm, 120s window on `RecentExit` | `person_appearance` event triggers it |

### Backend matching algorithm

```python
REID_THRESHOLD = 0.65
REID_WINDOW_S  = 120

def on_person_appearance(event: dict, tracker: PersonTracker):
    key    = f"{event['jetson_id']}:{event['camera_id']}:{event['track_id']}"
    person = tracker.active_persons.get(key)
    if person is None:
        return
    person.vector = event["appearance_vector"]

    match = tracker.find_match(person.vector, event["jetson_id"], event["camera_id"])
    if match:
        old_id = person.global_person_id
        person.global_person_id    = match.global_person_id
        person.identity            = match.identity
        person.identity_confidence = match.identity_confidence
        if person.is_entry_exit_cam:
            tracker.merge_session(old_id, match.global_person_id, event["camera_id"])

def find_match(vector, jetson_id, camera_id, recent_exits):
    v = np.array(vector, dtype=np.float32)
    now = datetime.now(timezone.utc)
    best, best_sim = None, -1.0
    for r in recent_exits:
        if r.jetson_id != jetson_id:
            continue
        if r.camera_id == camera_id:
            continue
        if r.vector is None:
            continue
        if (now - r.exit_time).total_seconds() > REID_WINDOW_S:
            continue
        sim = float(np.dot(v, np.array(r.vector, dtype=np.float32)))
        if sim > best_sim:
            best_sim, best = sim, r
    return best if best and best_sim >= REID_THRESHOLD else None
```

### ArcFace vs OSNet — not interchangeable

| Model | Field | Range | Threshold | Meaning |
|-------|-------|-------|-----------|---------|
| ArcFace (InsightFace) | `similarity` in `employee_seen` | 0–1 | Jetson applies ≥ 0.50 before sending | Face identity |
| OSNet | `appearance_vector` dot product | 0–1 | Backend applies ≥ 0.65 in `find_match` | Body/clothing re-ID |

Never mix these scores. A person can have high OSNet similarity (same clothing) but low ArcFace (different face angle), and vice versa.

---

## 9. Notifications & Severity

| Severity | Events | Required action |
|----------|--------|----------------|
| `critical` | (reserved for future: fall detection, fire/smoke) | Push notification immediately |
| `high` | (reserved for future: EPP violation) | Push notification ≤ 30s |
| `medium` | `unknown_person_alert` | Push with face photo |
| `info` | `person_entry`, `person_classified`, `employee_seen`, `analytics_snapshot` | Dashboard display only |

---

## 10. Reliability & Edge Cases

### Idempotency

Use `event_id` (UUID4) as an idempotency key. The Jetson retries on network timeouts — discard duplicates by `event_id` before processing.

### Missing `person_appearance`

`person_appearance` may never arrive for a track if:
- Person was too small (crop < 64×128 px)
- Person exited before AppearanceWorker finished
- OSNet model not loaded on this Jetson

The backend must handle `ActivePerson.vector = None` permanently and still process `person_exit` correctly. A `PersonSession` must be created and closed even without an appearance vector.

### Event ordering

Ordering is **guaranteed within one camera** (`person_entry` always before `person_exit` for the same track). Cross-camera ordering is **not guaranteed** — `person_entry` on ch02 can arrive before `person_exit` on ch01 for the same physical person.

### Jetson restart mid-session

Local `track_id` counters reset. New `person_entry` events arrive for the same physical persons. Re-ID via `person_appearance` can re-link them to open `PersonSessions` if appearance vectors match within the 120s window. Open `PersonSessions` without a matching `person_exit` should be closed at `last_seen + 2s` (or left open until the daily cleanup job).

### WebSocket disconnection

The Jetson reconnects automatically (exponential backoff 1s → 30s). Position snapshots during disconnection are not recovered — telemetry is best-effort. Heatmaps will have gaps but remain valid for the windows where data was received.

### REST queue overflow

The Jetson queues up to 512 events. If the backend is down for several minutes, oldest events are silently discarded. Do not assume events always arrive in contiguous order.

### `employee_exit` never arrives

If the Jetson crashes mid-session, `employee_exit` may never be sent. Use the heartbeat-based estimation:

```python
estimated_exit = interval.last_heartbeat + timedelta(seconds=30)
```

Apply this in a scheduled cleanup job (e.g., every 5 minutes) to close stale `EmployeeZoneInterval` records.
