# NX Computing AI — Jetson Edge Pipeline

Real-time CCTV analytics on NVIDIA Jetson using DeepStream 7.1.  
Modular inference pipeline: people detection is always active; additional models
(age/gender, EPP, fire/smoke, license plates, fall detection, face recognition) load
based on the client's contracted package.

---

## Repository layout

```
NX-JETSON/
├── deploy/                         # Everything that goes onto the Jetson
│   ├── pipelines/                  # Shared pipeline code — never edited per client
│   │   ├── app.py                  # Production pipeline (live DVR RTSP source)
│   │   ├── app_video_testing.py    # Testing pipeline (local MP4 file source)
│   │   ├── config_loader.py        # Merges /etc/nx_* + client config.yaml + .env
│   │   └── probes.py               # GStreamer probe + async API client
│   ├── clients/                    # One folder per client deployment
│   │   └── demo/
│   │       ├── config.yaml         # Non-sensitive config (channels, port, pipeline)
│   │       └── .env.example        # Template — copy to .env and fill credentials
│   ├── tools/                      # Utility scripts
│   │   ├── identify_dvr.py         # Auto-detect DVR brand + URL pattern
│   │   ├── probe_cameras.py        # Find which DVR channels have cameras
│   │   ├── download_models.py      # Download optional models (MoveNet, etc.)
│   │   ├── register_face.py        # Register known persons into face DB
│   │   └── update.sh               # Smart pull + conditional rebuild
│   ├── models/                     # Model files (TensorRT engines rebuilt per device)
│   ├── Dockerfile.jetson           # Jetson arm64 container image
│   ├── docker-entrypoint.sh        # Compiles custom softmax parser on container start
│   ├── docker-compose.yml          # Jetson deployment (deepstream + db + redis)
│   └── setup.sh                    # Jetson first-time setup (SSH, Tailscale, Docker)
└── dev/                            # Development, testing, unused models — not deployed
```

---

## First-time Jetson deploy

```bash
# 1. Clone repo on the Jetson
git clone https://github.com/AlejandroMova/NX-JETSON.git
cd NX-JETSON/deploy

# 2. Run setup — does everything automatically:
#    checks internet, installs Tailscale + SSH + VNC, scans for DVR on port 554,
#    builds Docker image, detects DVR URL pattern, detects active channels,
#    downloads OSNet model, starts pipeline.
#    --dvr-user / --dvr-pass  → DVR login (creates clients/<client>/.env automatically)
#    --api-key                → NX backend token (writes to .env)
#    --package                → contracted tier (writes /etc/nx_pipeline + config.yaml)
#    --stream-type sub        → use DVR sub-stream (960×544) for 16-camera deployments
sudo bash setup.sh \
  --client <client_name> \
  --hostname jetson-<client_name> \
  --package comercio_total \
  --authkey <tailscale-key> \
  --dvr-user admin \
  --dvr-pass <dvr-password> \
  --api-key <nx-api-token>

# 16-camera deployment (sub-stream to avoid NVDEC overload):
sudo bash setup.sh \
  --client <client_name> \
  --hostname jetson-<client_name> \
  --package industrial_basico \
  --stream-type sub \
  --authkey <tailscale-key> \
  --dvr-user admin \
  --dvr-pass <dvr-password> \
  --api-key <nx-api-token>

# 3. Inspect the live pipeline from your laptop (QA Visual — see section below)
./qa.sh
```

> **First run**: TensorRT builds engines for active models (~5 min each).  
> Subsequent starts take ~30 seconds.

> **DVR IP auto-recovery**: `setup.sh` installs a systemd service (`nx-dvr-watchdog`) that checks  
> direct TCP reachability to the configured DVR IP and automatically updates `/etc/nx_dvr_ip` +  
> restarts the pipeline if the DVR changes IP via DHCP. Check its status with `journalctl -u nx-dvr-watchdog -f`.

### Manual mode (if you need to run steps individually)

```bash
# Skip docker in setup, then run each step manually:
sudo bash setup.sh --client <client_name> --package comercio_total \
  --authkey <tailscale-key> --no-docker

docker compose build
docker compose run --rm deepstream python3 tools/identify_dvr.py --update-config
docker compose run --rm deepstream python3 tools/probe_cameras.py --update-config
docker compose up -d
```

> **Always run these tools inside the container.** Their dependencies come from
> `requirements.txt`, which is installed only in the Docker image — the Jetson host
> deliberately stays clean (Docker + Tailscale only). Running them directly, e.g.
> `python3 tools/identify_dvr.py`, fails with `ModuleNotFoundError: No module named 'dotenv'`
> (or `yaml`, `cv2`, `insightface`, depending on the tool). **This is by design, not a broken
> setup — never fix it with `pip3 install` on the host.** Prefix the command with
> `docker compose run --rm deepstream` instead.
>
> Exceptions that do run on the host, because they only use the standard library:
> `download_models.py`, `test_rtsp.py`, and `dvr_watchdog.sh` (pure bash).
> `register_face.py` is a separate case — it runs on your laptop, not the Jetson
> (see [Face Recognition](#face-recognition)).

### Overriding the pipeline without re-running setup

```bash
# Temporarily run a different package (does not persist):
docker compose run --rm \
  -e NX_PIPELINE=people_counting,age_gender \
  deepstream python3 pipelines/app.py

# Persist a new package on an already-deployed Jetson:
echo "people_counting,age_gender" | sudo tee /etc/nx_pipeline
docker compose restart deepstream
```

---

## Updating a deployed Jetson

Run the update script on the Jetson — it pulls the latest code, detects whether a Docker
rebuild is needed, and restarts the pipeline automatically:

```bash
# SSH into the Jetson, then:
cd ~/NX-JETSON/deploy
bash tools/update.sh
```

The script detects what changed and does the minimum work needed:

| What changed on GitHub | What the script does |
|---|---|
| `pipelines/*.py` — pipeline code | `git pull` + container restart (~5 sec) |
| `clients/*/config.yaml` — client config | `git pull` + container restart (~5 sec) |
| `requirements.txt` / `Dockerfile.jetson` | `git pull` + image rebuild + restart (~5 min) |

If you want to force a full rebuild regardless:
```bash
bash tools/update.sh --force-rebuild
```

> The `.env` credential file on the Jetson is never touched by updates —
> it lives outside the repo and survives `git pull` safely.

---

## QA Visual App

The QA Visual app lets any NX team member inspect a deployed Jetson remotely over Tailscale —
see the live camera feed with bounding-box overlays, monitor real-time detections, and watch
API call payloads, all from a browser. It runs entirely on the Jetson and has **zero impact
on production** when not active.

### Quick start

```bash
# SSH into the Jetson, then:
cd ~/NX-JETSON/deploy
./qa.sh          # start QA (prints Tailscale URL — Ctrl+click to open in browser)
./qa.sh stop     # stop QA from another terminal and restore production
```

`Ctrl+C` in the terminal where `qa.sh` is running also stops QA and restores the production pipeline.

### What you see

| Panel | Content |
|-------|---------|
| **Video** | MJPEG stream (25 fps) with coloured bounding boxes and labels for each active feature. |
| **Camera selector** | Sidebar dropdown — view all cameras tiled or any individual camera. |
| **Feature toggles** | Checkboxes per capability (`age_gender`, `fall_detection`, etc.) that take effect immediately without restarting the pipeline. `people_counting` is always on. |
| **Detections log** | Scrollable real-time log of every tracked person — track ID, label, confidence, fall alert. |
| **API calls log** | Collapsible JSON view of every POST the Jetson is sending to the backend. |

### How it works (architecture)

```
deepstream container (NX_QA_ENABLED=true):
  same GStreamer pipeline as production
  ├── probe (caps_rgba) draws bboxes on the tiled 640×360 frame (OpenCV, CPU)
  ├── extracts per-camera crops (numpy slice, ~0 ms)
  ├── publishes metadata to Redis pub/sub (nx:qa:detections, nx:qa:apicalls)
  └── MjpegServer thread serves annotated frames on :8080

qa_app container (Streamlit, :8501):
  subscribes to Redis pub/sub → updates UI panels at 500 ms
  embeds st.iframe → /viewer/<key> on MjpegServer (same-origin HTML)
  (browser fetches MJPEG directly from Jetson — no CORS, stream stable across rerenders)
```

### Performance notes

- CPU overhead: ~1-2 ms per frame for OpenCV drawing + JPEG encode (negligible on Orin Nano).
- Recommended: ≤ 8 streams with QA active on Orin Nano 8 GB. 16 streams are fine for production (fakesink), but enabling QA on 16 streams simultaneously may cause NVMM pressure.
- The Streamlit container is a lightweight `python:3.11-slim` image — no GPU access needed.
- Redis pub/sub messages are ephemeral; nothing is persisted to disk.

---

## Testing with a local video (no DVR needed)

```bash
# Place an MP4 in deploy/test_videos/
cp your_video.mp4 deploy/test_videos/

# Default (people_counting + age_gender):
docker compose run --rm deepstream \
    python3 pipelines/app_video_testing.py test_videos/your_video.mp4

# Test fall detection:
docker compose run --rm deepstream \
    python3 pipelines/app_video_testing.py test_videos/fall.mp4 \
    --capabilities people_counting,fall_detection

# Test face recognition:
docker compose run --rm deepstream \
    python3 pipelines/app_video_testing.py test_videos/office.mp4 \
    --capabilities people_counting,face_recognition --client demo
```

View output at `rtsp://<jetson-ip>:8554/ds-test` (VLC).

---

## Benchmarking: how many cameras does this Jetson handle?

`deploy/tools/benchmark_cameras.py` — a manual tool, **not part of `setup.sh`**, run when you
need to know the real ceiling for a given Jetson. It runs the real `app.py` (RTSP, not the
video-testing pipeline) across increasing camera counts and reports the max that stays real-time.

```bash
cd deploy
python3 tools/benchmark_cameras.py --list-variants       # qué config aplica cada modalidad
python3 tools/benchmark_cameras.py                      # default: fp32 / fp16 / fp16_sgie2, N=1,2,4,6,8
python3 tools/benchmark_cameras.py --counts 1,4,8,12,16
python3 tools/benchmark_cameras.py --variants fp16 --max 16 --step 2
```

It measures three modalities (all with the "ideal" settings `tracker: nvdcf_reid`,
`pgie_batch_size: 0`, `pgie_interval: 2`):
- `fp32` — `osnet_precision: fp32` (the accurate default)
- `fp16` — `osnet_precision: fp16` (cheaper, least accuracy impact of the levers tried)
- `fp16_sgie2` — `fp16` + `sgie_interval: 2` (OSNet is the diagnosed FPS bottleneck since it runs
  every frame regardless of `pgie_interval` — running it every 2 frames instead is the next-cheapest lever)

It simulates N cameras by cycling the client's configured `channels` (no RTSP server needed), stops
production `deepstream` while it measures (restored automatically on exit, including Ctrl+C), and
judges "handles it" as: FPS per stream stays ≥ 90% of what the DVR actually delivers (sondeado once
via `gst-discoverer-1.0`, so it isn't compute-bound) **and** GPU/RAM stay under headroom (`tegrastats`).
Requires the host tool `tegrastats` (comes with JetPack). See the module docstring for full details.

---

## Pipeline packages

The active models are controlled by the `pipeline` setting. Each capability maps to either a
DeepStream SGIE element or a Python worker thread. The startup sequence validates that all
required model files exist before GStreamer initializes.

### Package → capability mapping

Set `package:` in `config.yaml` — pipeline capabilities and sector are derived automatically.
Use `package: manual` to set `pipeline:` and `sector:` explicitly (testing / custom deployments).

| `package:` value | Pipeline capabilities | Sector |
|------------------|-----------------------|--------|
| `comercio_basico` / `comercio_avanzado` | `people_counting` | comercio |
| `comercio_total` / `comercio_enterprise` | `people_counting, age_gender, face_recognition` | comercio |
| `industrial_basico` | `people_counting` | industrial |
| `industrial_avanzado` | `people_counting, epp_detection` | industrial |
| `industrial_total` / `industrial_enterprise` | `people_counting, epp_detection, license_plate, fire_smoke, face_recognition` | industrial |
| `hogar_basico` | `people_counting` | hogar |
| `hogar_avanzado` | `people_counting, fall_detection` | hogar |
| `hogar_total` | `people_counting, fall_detection, fire_smoke, face_recognition` | hogar |
| `manual` | explicit `pipeline:` + `sector:` in config.yaml | — |

### Capability implementation

| Capability | Type | Model | Notes |
|-----------|------|-------|-------|
| `people_counting` | PGIE | PeopleNet v2.3.4 | Always active |
| `cross_camera_reid` | SGIE (gie-id=3) | OSNet-x1.0 ONNX | **Always active** (tied to people_counting, if the ONNX exists on disk). Auto-downloaded by setup.sh. Generates 512-dim appearance vectors for cross-camera re-ID directly via DeepStream tensor metadata — no Python worker. |
| `age_gender` | SGIE (gie-id=2) | ResNet-18 Pedestrian Attr | Full-body crop → 6 classes |
| `fall_detection` | Python worker | MoveNet Lightning ONNX | **Hogar only** (hogar_avanzado / hogar_total). Auto-downloaded by setup.sh. |
| `face_recognition` | Python worker | PeopleNet class 2 (face) + InsightFace buffalo_l | No extra model needed — uses face detections already produced by PeopleNet PGIE. comercio/industrial: identity rides along in `positions_snapshot` (`employee_id`/`face_confirmed` fields), no discrete event. hogar: `unknown_person_alert` (discrete event, intrusion alert). |
| `epp_detection` | SGIE | *(pending)* | Helmet/vest/gloves |
| `fire_smoke` | SGIE | *(pending)* | Frame-level classifier |
| `license_plate` | SGIE | *(pending)* | LPD + LPR |

### Where pipeline is resolved (priority order)

| Source | How to set | Use case |
|--------|-----------|----------|
| `NX_PIPELINE` env var | `docker compose run -e NX_PIPELINE=...` | One-off testing |
| `/etc/nx_pipeline` | `setup.sh --package <tier>` or `tee /etc/nx_pipeline` | Production |
| `config.yaml package:` | `setup.sh --package <tier>` or edit and push | Client default (recommended) |
| `config.yaml pipeline:` | Only when `package: manual` | Custom / testing |

Sector follows the same order: `NX_SECTOR` env → `/etc/nx_sector` → derived from `package:` → explicit `sector:` (only when `package: manual`).

### No NGC API key needed

`face_recognition` uses the face detections already produced by PeopleNet (class 2) — no separate face detector model is required. InsightFace `buffalo_l` (auto-downloaded on first run) handles the recognition step. No NGC account or API key needed.

### Adding a new model

When a new model (e.g. EPP) is ready:

1. Drop model files into `deploy/models/<capability>/` (ONNX + `config_infer.txt` + `labels.txt`)
2. Add one line to `SGIE_CONFIGS` in [deploy/pipelines/app.py](deploy/pipelines/app.py) (or `None` for Python workers)
3. Implement the handler class in [deploy/pipelines/probes.py](deploy/pipelines/probes.py) (stub already exists)
4. Set the package on the Jetson: `echo "people_counting,epp_detection" | sudo tee /etc/nx_pipeline`

---

## Adding a new client

1. **On your laptop, create the client folder and commit it:**
   ```bash
   cd deploy/
   mkdir clients/<client_name>
   cp clients/demo/config.yaml   clients/<client_name>/config.yaml
   cp clients/demo/.env.example  clients/<client_name>/.env.example
   # Edit clients/<client_name>/config.yaml as needed
   git add clients/<client_name>/config.yaml clients/<client_name>/.env.example
   git commit -m "feat: add client <client_name>"
   git push
   ```

2. **Edit `deploy/clients/<client_name>/config.yaml`:**
   ```yaml
   client_name: <client_name>
   dvr_port: 554
   rtsp_url_pattern: "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/main/av_stream"
   channels: [1, 2]
   package: comercio_basico   # sets pipeline + sector automatically; use "manual" for custom
   stream_type: main          # main (1080p, ≤6 cams) | sub (960×544, 16+ cams)
   tracker: nvdcf             # nvdcf (≤6 streams) | iou (up to 16 streams)
   entry_exit_channels: []
   ```
   > `setup.sh --package` and `setup.sh --stream-type` write these fields automatically.
   > `identify_dvr.py --update-config` sets `rtsp_url_pattern` and `stream_type` automatically.

3. **On the Jetson, pull and set credentials + package:**
   ```bash
   cd ~/NX-JETSON && git pull
   cd deploy/
   cp clients/<client_name>/.env.example clients/<client_name>/.env
   nano clients/<client_name>/.env     # fill DVR_USER and DVR_PASS
   # Re-run setup to apply the new client config:
   sudo bash setup.sh --client <client_name> --package comercio_basico
   ```
   `.env` is gitignored and stays only on the Jetson.

---

## Config reference

### `deploy/clients/<name>/config.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `client_name` | string | — | Must match the folder name |
| `dvr_port` | int | `554` | RTSP port on the DVR |
| `rtsp_url_pattern` | string | generic | URL template with `{user}`, `{password}`, `{dvr_ip}`, `{port}`, `{ch:02d}`. Set automatically by `identify_dvr.py --update-config`. |
| `channels` | list[int] | `[1]` | DVR channel numbers to ingest. Set automatically by `probe_cameras.py --update-config`. |
| `package` | string | `"manual"` | Contracted package — derives `pipeline` capabilities and `sector` automatically. Set by `setup.sh --package`. Use `"manual"` to specify `pipeline` and `sector` explicitly. |
| `stream_type` | string | `"main"` | `"main"` — 1920×1080, up to 6 cameras. `"sub"` — 960×544, up to 16 cameras (requires sub-stream URL in `rtsp_url_pattern`). Set by `setup.sh --stream-type` and `identify_dvr.py --stream-type`. |
| `tracker` | string | `"nvdcf"` | Tracker algorithm — see table below |
| `entry_exit_channels` | list[int] | `[]` | Subset of `channels` whose cameras cover entrance/exit doors. Used to calculate store dwell time and visit count. Set with `setup.sh --entry-exit-channels "1,2"` or edited manually. |
| `pipeline` | list | `[people_counting]` | Only used when `package: manual`. Valid values: `people_counting`, `age_gender`, `epp_detection`, `fire_smoke`, `license_plate`, `fall_detection`, `face_recognition`. |
| `sector` | string | `"comercio"` | Only used when `package: manual`. `"comercio"` \| `"industrial"` \| `"hogar"`. Controls event types for `face_recognition` and severity of `fall_detected`. |

#### Tracker selection

| Value | Algorithm | Max stable streams (Orin Nano) | Best for |
|-------|-----------|-------------------------------|----------|
| `nvdcf` | NvDCF correlation filter | ~6 | Clients with few cameras, need precise re-ID through occlusions |
| `nvdcf_extended_shadow` | Same as `nvdcf`, `maxShadowTrackingAge` 51→100 frames | ~6 | Fragmented `track_id` from brief occlusion/fast movement, no new model needed |
| `nvdcf_reid` | `nvdcf_extended_shadow` + NvDCF's own ReID/Re-Assoc submodule | ~6 (GPU cost not yet measured) | Fragmented `track_id` surviving longer occlusion than shadow tracking covers — intra-camera only, complements (does not replace) the cross-camera OSNet appearance SGIE. Requires `python3 tools/download_models.py --tracker-reid`. |
| `nvdcf_accuracy` | ⚠️ Broken — NVIDIA stock accuracy profile, missing model was never downloaded (`"TAO model file does not exist"`). Use `nvdcf_reid` instead. | — | — |
| `iou` | IoU bounding-box matching | 16 | Clients with many cameras, people-counting use case |

**Sub-stream tip:** For 16-camera deployments the Jetson Orin Nano NVDEC will overload at 1080p.
Use the DVR sub-stream (960×544) to keep NVDEC within its rated capacity.
`setup.sh --stream-type sub` and `identify_dvr.py --stream-type sub --update-config` handle this automatically — they set the correct sub-stream URL for the detected DVR brand and write `stream_type: sub` to config.yaml.

```yaml
# 16-camera example (Dahua, sub-stream — set automatically by identify_dvr.py)
rtsp_url_pattern: "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=1"
channels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
stream_type: sub
tracker: iou

# 1–6 camera example (main stream, full quality)
rtsp_url_pattern: "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=0"
channels: [1]
stream_type: main
tracker: nvdcf
```

Sub-stream URL paths by brand (set automatically by `identify_dvr.py`):

| Brand | Main stream | Sub-stream |
|-------|-------------|------------|
| Generic / QSee / Swann / Annke | `/ch{ch:02d}/main/av_stream` | `/ch{ch:02d}/sub/av_stream` |
| Hikvision | `/Streaming/Channels/{ch:02d}01` | `/Streaming/Channels/{ch:02d}02` |
| Dahua / Amcrest / Lorex | `?channel={ch}&subtype=0` | `?channel={ch}&subtype=1` |
| Reolink | `/h264Preview_{ch:02d}_main` | `/h264Preview_{ch:02d}_sub` |
| Uniview / UNV | `/media/video{ch}` | `/unicast/c{ch}/s1/live` |
| Axis / Hanwha | device-specific | set manually via DVR web UI |

### `deploy/clients/<name>/.env` (gitignored)

| Key | Description |
|-----|-------------|
| `DVR_USER` | DVR login username |
| `DVR_PASS` | DVR login password |

### `deploy/.env` (gitignored) — Docker Compose level

| Key | Description |
|-----|-------------|
| `API_BASE_URL` | NX backend URL (e.g. `https://api.nxcomputing.ai`) |
| `API_KEY` | API token assigned to this Jetson. Set with `setup.sh --api-key <token>` or edit `.env` manually. |
| `JETSON_ID` | Device identifier sent with every event (e.g. `jetson-mova-001`) |
| `WS_BASE_URL` | WebSocket base URL for position telemetry (e.g. `wss://api.nxcomputing.ai`). Leave empty to disable — the pipeline runs normally without it. |

Copy from `deploy/.env.example` and fill in before first deploy.

### Runtime resolution order

`NX_PIPELINE` env var → `/etc/nx_pipeline` → `config.yaml package:` (derives pipeline) → `config.yaml pipeline:` (only when `package: manual`)  
`NX_SECTOR` env var → `/etc/nx_sector` → derived from `config.yaml package:` → `config.yaml sector:` (only when `package: manual`)  
`NX_CLIENT` env var → `/etc/nx_client` (written by setup.sh)  
`NX_DVR_IP` env var → `/etc/nx_dvr_ip` (written by setup.sh)  
`DVR_USER` / `DVR_PASS` env vars → `clients/<name>/.env`

---

## Face Recognition

### Registering known persons (run on your laptop)

Embeddings are generated on your laptop — photos never leave your machine. Only the
resulting `known_faces.json` (~1 KB/person) is copied to the Jetson.

```bash
# Prerequisites (one-time):
pip install insightface onnxruntime opencv-python

# Option A — register one person from an image:
python deploy/tools/register_face.py --name "Juan Perez" --image foto.jpg --client <client_name>

# Option B — import a full folder at once (recommended for many people):
#   Folder structure: clients/<client>/faces/<Name>/<photo.jpg>
python deploy/tools/register_face.py --import-dir clients/<client_name>/faces/ --client <client_name>

# List registered persons:
python deploy/tools/register_face.py --list --client <client_name>

# Delete a person:
python deploy/tools/register_face.py --delete "Juan Perez" --client <client_name>
```

### Deploying the face database to a Jetson

```bash
# Copy the JSON (not the photos) to the Jetson:
scp deploy/clients/<client_name>/known_faces.json \
    jetson@<ip>:/nx_tech/clients/<client_name>/

# Restart the container to reload the DB:
ssh jetson@<ip> "cd /nx_tech && docker compose restart deepstream"
```

### What gets stored

`known_faces.json` contains only 512-dimensional ArcFace embedding vectors (no photos):
```json
{
  "Juan Perez": [[0.12, -0.34, ...], [0.11, -0.35, ...]],
  "Ana Lopez":  [[0.55,  0.21, ...]]
}
```
Multiple embeddings per person increase recognition robustness across poses/lighting.

### Face recognition — setup by sector

The same `face_recognition` capability behaves differently depending on the client's `sector`:

| Sector | Known person | Unknown person event | Use case |
|--------|--------------------|---------------------|----------|
| `comercio` / `industrial` | Identity rides along in `positions_snapshot` (`employee_id`/`face_confirmed` fields) — no discrete event | — | Employee attendance per zone, derived by the backend from positions |
| `hogar` | (no attendance use case) | `unknown_person_alert` (with face crop) | Intruder alert |

For **comercio/industrial**: register employees with `register_face.py` as usual — once identified, `_FaceRecognitionHandler` tags the person's cross-camera `global_id` and it flows to the backend on every position update, not as a one-off event.
For **hogar**: register household members the same way. Anyone not in the database triggers an `unknown_person_alert` with a face snapshot — note this requires ReID (`global_id`) to be resolved for that track first (see `cross_camera_reid` below), so a Jetson without the OSNet model installed won't trigger it.

---

## Cross-camera person tracking

The pipeline generates a 512-dimensional **appearance embedding** (OSNet-x1.0 ONNX) for every detected person, extracted **directly from the OSNet SGIE** (gie-id=3) via DeepStream tensor metadata — no Python worker, no crop copy. This vector captures clothing/silhouette, not face — it works without `face_recognition` enabled.

**How it works:**

1. Person detected → `person_entry` emission deferred until the SGIE embedding is ready (same frame as detection; up to ~1 s / 30 frames as a safety fallback if the bbox never clears the SGIE's min size, 96×192 px)
2. `ReIdManager.match_or_create()` matches the vector against the local DB (cosine similarity ≥ 0.85, `SIMILARITY_THRESHOLD` — calibrated 2026-07-08 against real client crops) and returns a `global_id`
3. **Retry before creating a new identity:** a track's very first embedding no longer decides its fate permanently. A brand-new `global_id` is only seeded once the view is confidently full-body (`ratio ≥ FULL_BODY_MIN_RATIO=2.2`) or the ~1 s deadline is reached — an ambiguous view (e.g. bent over, only torso visible) that doesn't match anything just waits for a better frame on the same track instead of committing early. Bboxes below `PARTIAL_BODY_MIN_RATIO=1.3` (only legs/feet visible) are skipped entirely, no match attempt at all.
4. `person_entry` event is sent with `global_id` and `entry_type: "new"` | `"return"`, or `person_channel_change` if same person moved cameras within 5 min
5. If no embedding ever clears the SGIE's min-size threshold before the deadline, `person_entry` is sent with `global_id: null`

**Event types emitted:**
- `person_entry` (`entry_type: "new"`) — never seen before
- `person_entry` (`entry_type: "return"`) — same `global_id`, last seen > 5 min ago
- `person_channel_change` — same `global_id`, switched cameras within the presence window

**What this enables:**
- Count unique visitors (not camera views)
- Build cross-camera movement paths
- Calculate true store dwell time (entry cam → exit cam)
- Identify employees in cameras where their face is not visible

The cross-camera matching runs **locally on the Jetson** via `ReIdManager` (no cloud round-trip). Identity DB persists across restarts at `deploy/reid_db.json` with a 1-hour TTL. Each identity keeps a gallery of up to 10 embeddings (angles/poses); a new embedding is added only if it's diverse enough from existing ones (0.71 ≤ similarity < 0.95, `GALLERY_DIVERSITY_THRESHOLD_MIN/MAX`) so near-duplicate frames don't waste a gallery slot.

OSNet model is auto-downloaded by `setup.sh`. To download manually:
```bash
docker compose run --rm deepstream python3 tools/download_models.py --reid
```

---

## WebSocket position stream

The pipeline sends normalized person centroids to the backend every 1 second per camera (`POSITION_SEND_INTERVAL` in `probes.py`) via a persistent WebSocket connection (not REST). This enables real-time heatmaps in the dashboard, and — for comercio/industrial employees — attendance.

```
WS /ws/positions  (Jetson is client, backend is server)
Message: { type: "positions_snapshot", camera_id, timestamp,
           positions: [{global_id, x_norm, y_norm, employee_id, face_confirmed}] }
```

Entries are keyed by `global_id` (cross-camera ReID identity), not `track_id`. `employee_id` is set once face recognition tags that `global_id` as a known employee (see "Face recognition — setup by sector" above); `face_confirmed` is `true` only in the cycle a face was actually re-checked, not a persistent flag.

`x_norm` and `y_norm` are in the range [0, 1] relative to frame dimensions. To map to pixel coordinates:
```
pixel_x = x_norm × frame_width
pixel_y = y_norm × frame_height
```
Overlay on the `reference-frame` image the pipeline sends when the scene is empty (and re-sends when the layout changes significantly) for accurate heatmap rendering.

**Why WebSocket instead of REST:** WebSocket keeps a single TCP connection open. For 6 cameras × 6 snapshots/min = 36 messages/min, REST adds ~300 bytes HTTP overhead per request. WebSocket adds 2-10 bytes. Reconnects automatically with exponential backoff (1s → 2s → ... → 30s).

To enable, set `WS_BASE_URL` in `deploy/.env`:
```
WS_BASE_URL=wss://api.nxcomputing.ai
```
Leave empty to disable — the pipeline runs normally, position data is silently dropped.

---

## Architecture

```
DVR (RTSP) → rtspsrc → nvv4l2decoder → nvstreammux
  → PeopleNet PGIE (gie-id=1, always active) → Tracker (NvDCF or IOU)
      ├── class 0 (person) → age_gender SGIE (gie-id=2), tracker lifecycle, appearance
      └── class 2 (face)   → FaceRecognizer worker (if face_recognition active)
  → [SGIE: age_gender    gie-id=2]  ← one nvinfer per active SGIE capability
  → [SGIE: epp/fire/lpr  gie-id=3+] ← pending models
  → nvmultistreamtiler(640×360) → nvvideoconvert → capsfilter(RGBA)
  → [probe: crops + analytics] → fakesink
  (QA mode: probe also draws overlays + feeds MjpegServer on :8080 + publishes Redis metadata)

Async paths (background threads, never block pipeline):
  people_counting → OSNet SGIE (gie-id=3, in-pipeline, not async) → 512-dim vector → ReIdManager (local match, 0.85 threshold) → POST /api/events (person_entry with global_id | person_channel_change)
  fall_detection  → PoseWorker       → MoveNet ONNX → 17 keypoints → POST /api/events (fall_detected)  # NOTE: pose_worker.py not found in current repo — this line may be stale, not verified in this pass
  face_recognition→ FaceRecognizer   → InsightFace ArcFace → tags global_id (comercio/industrial, rides in positions_snapshot) or POST /api/events unknown_person_alert (hogar)
  all cameras     → NxApiClient      → REST POST /api/events, /api/analytics, /api/crops
  all cameras     → WsPositionClient → WS /ws/positions (positions_snapshot every 1s)
```

The probe dispatcher reads PeopleNet class 0 (person) for tracking and class 2 (face) for
recognition — both come from the same PGIE pass at zero extra cost.
`fall_detection` and `face_recognition` are pure Python worker threads (no extra SGIE element).

The production pipeline has no video output (`fakesink`). To view the live feed with bounding-box
overlays, use the QA Visual app:
```bash
./qa.sh    # opens Streamlit dashboard at http://<jetson-tailscale-ip>:8501
```
