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

# 2. Create client credentials (DVR login)
cp clients/demo/.env.example clients/<client_name>/.env
nano clients/<client_name>/.env     # fill DVR_USER and DVR_PASS

# 3. Create API credentials (NX backend token)
cp .env.example .env
nano .env                           # fill API_KEY and API_BASE_URL

# 4. Run setup — does everything automatically:
#    installs Docker + Tailscale, detects DVR IP, builds image,
#    identifies DVR URL pattern, detects active channels, starts pipeline.
#    --package sets the contracted tier (writes /etc/nx_pipeline).
sudo bash setup.sh \
  --client <client_name> \
  --package comercio_total \
  --authkey <tailscale-key>

# 5. Watch inference from your laptop
vlc --network-caching=300 http://<jetson-tailscale-ip>:8080/stream
```

> **First run**: TensorRT builds engines for active models (~5 min each).  
> Subsequent starts take ~30 seconds.

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

## Pipeline packages

The active models are controlled by the `pipeline` setting. Each capability maps to either a
DeepStream SGIE element or a Python worker thread. The startup sequence validates that all
required model files exist before GStreamer initializes.

### Package → capability mapping

| Package | `pipeline` value | Models loaded |
|---------|-----------------|---------------|
| Comercio Básico / Avanzado | `people_counting` | PeopleNet only |
| Comercio Total | `people_counting, age_gender` | + ResNet-18 age/gender |
| Comercio Enterprise | `people_counting, age_gender, face_recognition` | + FaceDetectIR + InsightFace |
| Industrial Básico | `people_counting` | PeopleNet only |
| Industrial Avanzado | `people_counting, epp_detection` | + PPE SGIE *(model pending)* |
| Industrial Total | `people_counting, epp_detection, license_plate, fire_smoke` | + LPD/LPR + fire *(pending)* |
| Industrial Enterprise | `people_counting, epp_detection, license_plate, fire_smoke, face_recognition` | + face recognition |
| Hogar Básico | `people_counting` | PeopleNet only |
| Hogar Avanzado | `people_counting, fall_detection` | + MoveNet pose (Python worker) |
| Hogar Total | `people_counting, fall_detection, fire_smoke, face_recognition` | + fire + face recognition |

### Capability implementation

| Capability | Type | Model | Notes |
|-----------|------|-------|-------|
| `people_counting` | PGIE | PeopleNet v2.3.4 | Always active |
| `cross_camera_reid` | Python worker | OSNet-x0.25 ONNX | **Always active** (tied to people_counting). Auto-downloaded by setup.sh. Generates 512-dim appearance vectors for cross-camera re-ID. |
| `age_gender` | SGIE (gie-id=2) | ResNet-18 Pedestrian Attr | Full-body crop → 6 classes |
| `fall_detection` | Python worker | MoveNet Lightning ONNX | **Hogar only** (hogar_avanzado / hogar_total). Auto-downloaded by setup.sh. |
| `face_recognition` | SGIE (gie-id=3) + Python worker | FaceDetectIR + InsightFace buffalo_l | Requires NGC API key. Event type depends on `sector`: `employee_seen` (comercio/industrial) or `known_person_seen` + `unknown_person_alert` (hogar). |
| `epp_detection` | SGIE | *(pending)* | Helmet/vest/gloves |
| `fire_smoke` | SGIE | *(pending)* | Frame-level classifier |
| `license_plate` | SGIE | *(pending)* | LPD + LPR |

### Where pipeline is resolved (priority order)

| Source | How to set | Use case |
|--------|-----------|----------|
| `NX_PIPELINE` env var | `docker compose run -e NX_PIPELINE=...` | One-off testing |
| `/etc/nx_pipeline` | `setup.sh --package <tier>` or `tee /etc/nx_pipeline` | Production |
| `config.yaml pipeline:` | Edit client config and push | Client default / fallback |

### NGC API Key (required for face_recognition)

FaceDetectIR is downloaded from NVIDIA GPU Cloud (NGC) and requires a free account:

1. Create account at [ngc.nvidia.com](https://ngc.nvidia.com) (free)
2. Generate an API key at **ngc.nvidia.com/setup/api-key**
3. `setup.sh` will prompt for it automatically when `face_recognition` is in the pipeline  
   — the key is saved to `/etc/nx_ngc_key` (mode 600) for future runs

```bash
# Manual download if needed:
NGC_API_KEY=<your-key>
wget --header="Authorization: ApiKey ${NGC_API_KEY}" \
  -O deploy/models/facedetect_ir/resnet18_facedetectir_pruned_quantized.onnx \
  "https://api.ngc.nvidia.com/v2/models/nvidia/tao/facedetectir/versions/pruned_quantized_v2.0/files/resnet18_facedetectir_pruned_quantized.onnx"
```

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
   channels: [1, 2]        # DVR channel numbers to process
   stream_width: 1920
   stream_height: 1080
   pipeline:               # default fallback — overridden by /etc/nx_pipeline on the Jetson
     - people_counting
   tracker: nvdcf           # nvdcf (≤6 streams) | iou (up to 16 streams)
   ```

3. **On the Jetson, pull and set credentials + package:**
   ```bash
   cd ~/NX-JETSON && git pull
   cd deploy/
   cp clients/<client_name>/.env.example clients/<client_name>/.env
   nano clients/<client_name>/.env     # fill DVR_USER and DVR_PASS
   echo "<client_name>" | sudo tee /etc/nx_client
   # Set contracted package (writes /etc/nx_pipeline):
   echo "people_counting,age_gender" | sudo tee /etc/nx_pipeline
   docker compose restart deepstream
   ```
   `.env` is gitignored and stays only on the Jetson.

---

## Config reference

### `deploy/clients/<name>/config.yaml`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `client_name` | string | — | Must match the folder name |
| `dvr_port` | int | `554` | RTSP port on the DVR |
| `rtsp_url_pattern` | string | generic | URL template with `{user}`, `{password}`, `{dvr_ip}`, `{port}`, `{ch:02d}` |
| `channels` | list[int] | `[1]` | DVR channel numbers to ingest |
| `stream_width` | int | `1920` | DVR main stream width |
| `stream_height` | int | `1080` | DVR main stream height |
| `pipeline` | list | `[people_counting]` | Default capabilities for this client. Overridden by `/etc/nx_pipeline` or `NX_PIPELINE` env var. Valid values: `people_counting`, `age_gender`, `epp_detection`, `fire_smoke`, `license_plate`, `fall_detection`, `face_recognition` |
| `tracker` | string | `nvdcf` | Tracker algorithm — see table below |
| `sector` | string | `"comercio"` | Client vertical: `"comercio"` \| `"industrial"` \| `"hogar"`. Inferred automatically by `setup.sh --package`. Controls event types emitted by `face_recognition` and severity of `fall_detected`. |
| `entry_exit_channels` | list[int] | `[]` | Subset of `channels` whose cameras cover entrance/exit doors. Used to calculate store dwell time and visit count. Leave empty if no camera directly covers the entrance. Can be set with `setup.sh --entry-exit-channels "1,2"` or edited manually later. |
| `stream_width` | int | `1920` | Resolution fed to `nvstreammux` (inference + MJPEG output) |
| `stream_height` | int | `1080` | Same, height |

#### Tracker selection

| Value | Algorithm | Max stable streams (Orin Nano) | Best for |
|-------|-----------|-------------------------------|----------|
| `nvdcf` | NvDCF correlation filter | ~6 | Clients with few cameras, need precise re-ID through occlusions |
| `iou` | IoU bounding-box matching | 16 | Clients with many cameras, people-counting use case |

**Substream tip:** For multi-camera deployments use the DVR substream (`subtype=1` in Dahua URL) and set `stream_width: 960` / `stream_height: 544`. This reduces GPU memory pressure significantly.

```yaml
# Multi-camera example (Dahua, 16 channels, substream)
rtsp_url_pattern: "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=1"
channels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
stream_width: 960
stream_height: 544
tracker: iou

# Single/few cameras example (main stream, full quality)
rtsp_url_pattern: "rtsp://{user}:{password}@{dvr_ip}:{port}/cam/realmonitor?channel={ch}&subtype=0"
channels: [1]
stream_width: 1920
stream_height: 1080
tracker: nvdcf
```

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

`NX_PIPELINE` env var → `/etc/nx_pipeline` (setup.sh `--package`) → `config.yaml pipeline:`  
`NX_SECTOR` env var → `/etc/nx_sector` (inferred by setup.sh from `--package`) → `config.yaml sector:`  
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

The same `face_recognition` capability emits different API events depending on the client's `sector`:

| Sector | Known person event | Unknown person event | Use case |
|--------|--------------------|---------------------|----------|
| `comercio` / `industrial` | `employee_seen` + `employee_presence` + `employee_exit` | — | Employee attendance per zone |
| `hogar` | `known_person_seen` + `known_person_exit` | `unknown_person_alert` (with face crop) | Family member arrival / intruder alert |

For **comercio/industrial**: register employees with `register_face.py` as usual.  
For **hogar**: register household members the same way — they appear as `known_person_seen` events. Anyone not in the database triggers an `unknown_person_alert` with a face snapshot.

---

## Cross-camera person tracking

The pipeline generates a 512-dimensional **appearance embedding** (OSNet-x0.25 ONNX) for every detected person. This vector captures clothing/silhouette, not face — it works without `face_recognition` enabled.

**How it works:**

1. Person detected → `person_entry` event (immediate)
2. AppearanceWorker processes the crop (~1-2s) → `person_appearance` event with 512-dim vector
3. Both share the same `(jetson_id, camera_id, track_id)` triplet — the backend joins them
4. When the same person appears on a different camera within 120s, the backend matches vectors (cosine similarity > 0.65) and assigns a `global_person_id` to link both sightings

**What this enables:**
- Count unique visitors (not camera views)
- Build cross-camera movement paths
- Calculate true store dwell time (entry cam → exit cam)
- Identify employees in cameras where their face is not visible

The cross-camera matching itself runs in the **backend** — the Jetson only generates and sends the vector. This allows matching across multiple Jetsons on the same client site.

OSNet model is auto-downloaded by `setup.sh`. To download manually:
```bash
python3 deploy/tools/download_models.py --reid
```

---

## WebSocket position stream

The pipeline sends normalized person centroids to the backend every 10 seconds per camera via a persistent WebSocket connection (not REST). This enables real-time heatmaps in the dashboard.

```
WS /ws/positions  (Jetson is client, backend is server)
Message: { type: "positions_snapshot", camera_id, timestamp, positions: [{track_id, x_norm, y_norm}] }
```

`x_norm` and `y_norm` are in the range [0, 1] relative to frame dimensions. To map to pixel coordinates:
```
pixel_x = x_norm × frame_width
pixel_y = y_norm × frame_height
```
Overlay on the `reference-frame` image the pipeline sends at startup for accurate heatmap rendering.

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
  → [SGIE: age_gender    gie-id=2]  ← one nvinfer per active SGIE capability
  → [SGIE: facedetectir  gie-id=3]  ← face detection (if face_recognition active)
  → [SGIE: epp/fire/lpr  gie-id=4+] ← pending models
  → nvmultistreamtiler → nvdsosd
  → nvvideoconvert → appsink → HTTP MJPEG server (port 8080)

Async paths (background threads, never block pipeline):
  people_counting → AppearanceWorker → OSNet ONNX → 512-dim vector → POST /api/events (person_appearance)
  fall_detection  → PoseWorker       → MoveNet ONNX → 17 keypoints → POST /api/events (fall_detected)
  face_recognition→ FaceRecognizer   → InsightFace ArcFace → POST /api/events (employee_seen / known_person_seen)
  all cameras     → NxApiClient      → REST POST /api/events, /api/analytics, /api/crops
  all cameras     → WsPositionClient → WS /ws/positions (positions_snapshot every 10s)
```

The probe dispatcher separates PGIE person detections from FaceDetectIR SGIE face detections.
`fall_detection` and `face_recognition` use Python worker threads (same async pattern as the
REST API client) so inference never blocks the GStreamer pipeline.

View the live feed:
```bash
vlc --network-caching=300 http://<jetson-ip>:8080/stream
```
With multiple streams the output is a tiled grid (e.g. 4×4 for 16 cameras) at 1920×1080.
