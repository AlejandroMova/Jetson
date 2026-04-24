# NX Computing AI — Jetson Edge Pipeline

Real-time CCTV analytics on NVIDIA Jetson NX using DeepStream 6.3.  
Detects people, classifies age/gender, and streams inference output via RTSP.

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
│   ├── tools/                      # Utility scripts (run inside container)
│   │   ├── identify_dvr.py         # Auto-detect DVR brand + URL pattern
│   │   ├── probe_cameras.py        # Find which DVR channels have cameras
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

# 2. Create client credentials (on the Jetson — never on laptop)
cp clients/demo/.env.example clients/<client_name>/.env
nano clients/<client_name>/.env     # fill DVR_USER and DVR_PASS

# 3. Run setup — does everything automatically:
#    installs Docker + Tailscale, detects DVR IP, builds image,
#    identifies DVR URL pattern, detects active channels, starts pipeline
sudo bash setup.sh --client <client_name> --authkey <tailscale-key>

# 4. Watch inference from your laptop
vlc rtsp://<jetson-tailscale-ip>:8554/ds-test
```

> **First run**: TensorRT will build engines for both models (~5 min each).  
> Subsequent starts take ~30 seconds.

### Manual mode (if you need to run steps individually)

```bash
# Skip docker in setup, then run each step manually:
sudo bash setup.sh --client <client_name> --authkey <tailscale-key> --no-docker

docker compose build
docker compose run --rm deepstream python3 tools/identify_dvr.py --update-config
docker compose run --rm deepstream python3 tools/probe_cameras.py --update-config
docker compose up -d
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

# SSH into Jetson, then:
cd ~/NX-JETSON/deploy
docker compose run --rm deepstream \
    python3 pipelines/app_video_testing.py test_videos/your_video.mp4
```

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
   pipeline: people_counting
   ```

3. **On the Jetson, pull and set credentials:**
   ```bash
   cd ~/NX-JETSON && git pull
   cd deploy/
   cp clients/<client_name>/.env.example clients/<client_name>/.env
   nano clients/<client_name>/.env     # fill DVR_USER and DVR_PASS
   echo "<client_name>" | sudo tee /etc/nx_client
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
| `pipeline` | string or list | `people_counting` | Which pipeline(s) to run |

### `deploy/clients/<name>/.env` (gitignored)

| Key | Description |
|-----|-------------|
| `DVR_USER` | DVR login username |
| `DVR_PASS` | DVR login password |

### Runtime resolution order

`NX_CLIENT` env var → `/etc/nx_client` (written by setup.sh)  
`NX_DVR_IP` env var → `/etc/nx_dvr_ip` (written by setup.sh)  
`DVR_USER` / `DVR_PASS` env vars → `clients/<name>/.env`

---

## Architecture

```
DVR (RTSP) → rtspsrc → nvv4l2decoder → nvstreammux
  → PeopleNet (PGIE) → NvDCF Tracker
  → ResNet-18 Age/Gender (SGIE)
  → nvdsosd → nvrtspoutsinkbin (port 8554)
              → REST API (probes.py async client)
```
