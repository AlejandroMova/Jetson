"""
config_loader.py — NX Computing AI | Client Configuration Loader

Merges four sources at runtime (highest priority first):
  1. NX_PIPELINE env var  (or /etc/nx_pipeline)  → active pipeline capabilities
  2. NX_CLIENT env var    (or /etc/nx_client)     → which client folder to load
  3. NX_DVR_IP env var    (or /etc/nx_dvr_ip)     → DVR IP discovered by setup.sh
  4. clients/<name>/config.yaml                   → non-sensitive client config
  5. clients/<name>/.env                          → DVR_USER / DVR_PASS (gitignored)

Usage:
    from config_loader import load_config
    cfg = load_config()
    for url in cfg.rtsp_urls():
        print(url)
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Repo root is one level above this file (NX-JETSON/)
_REPO_ROOT = Path(__file__).resolve().parent.parent


TRACKER_CONFIGS = {
    "nvdcf": "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
    "iou":   "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml",
}

# All known pipeline capabilities. Each capability beyond people_counting maps to a SGIE.
# Model files must exist before a capability can be activated (validated at startup).
VALID_CAPABILITIES = {
    "people_counting",   # Always active — PeopleNet PGIE + tracker. No extra model needed.
    "age_gender",        # ResNet-18 SGIE: gender (male/female) + age group (young/adult/senior)
    "epp_detection",     # SGIE: PPE compliance — helmet, vest, gloves on person crops
    "fire_smoke",        # Frame-level classifier: fire and smoke detection
    "license_plate",     # LPD + LPR: detect and read vehicle license plates
    "fall_detection",    # MoveNet ONNX Python worker: detects person fall events
    "face_recognition",  # FaceDetectIR SGIE + InsightFace ArcFace: identifies known persons
}


@dataclass
class ClientConfig:
    client_name: str
    dvr_ip: str
    dvr_port: int
    dvr_user: str
    dvr_pass: str
    rtsp_url_pattern: str
    channels: List[int]
    pipeline: List[str]
    stream_width: int = 1920
    stream_height: int = 1080
    tracker: str = "nvdcf"  # "nvdcf" (precise, ≤6 streams) | "iou" (stable, 16 streams)
    sector: str = "comercio"  # "comercio" | "industrial" | "hogar"
    entry_exit_channels: List[int] = field(default_factory=list)

    def tracker_config_path(self) -> str:
        if self.tracker not in TRACKER_CONFIGS:
            raise RuntimeError(
                f"Unknown tracker '{self.tracker}'. Valid options: {list(TRACKER_CONFIGS)}"
            )
        return TRACKER_CONFIGS[self.tracker]

    def rtsp_urls(self) -> List[str]:
        """Build one RTSP URL per active channel."""
        urls = []
        for ch in self.channels:
            url = (
                self.rtsp_url_pattern
                .replace("{user}",     self.dvr_user)
                .replace("{password}", self.dvr_pass)
                .replace("{dvr_ip}",   self.dvr_ip)
                .replace("{port}",     str(self.dvr_port))
                .replace("{ch:02d}",   f"{ch:02d}")
                .replace("{ch}",       str(ch))
            )
            urls.append(url)
        return urls

    def log_summary(self):
        logger.info("Client     : %s", self.client_name)
        logger.info("DVR        : %s:%d", self.dvr_ip, self.dvr_port)
        logger.info("Channels   : %s", self.channels)
        logger.info("Pipeline(s): %s", self.pipeline)
        logger.info("Resolution : %dx%d", self.stream_width, self.stream_height)
        logger.info("Tracker    : %s", self.tracker)
        for url in self.rtsp_urls():
            masked = url.replace(self.dvr_pass, "***") if self.dvr_pass else url
            logger.info("RTSP URL   : %s", masked)

    def active_sgies(self) -> List[str]:
        """Return capabilities that require a SGIE (everything except people_counting)."""
        return [c for c in self.pipeline if c != "people_counting"]

    def entry_exit_pad_indices(self) -> set:
        """Return the pad indices that correspond to entry/exit cameras."""
        return {
            idx for idx, ch in enumerate(self.channels)
            if ch in self.entry_exit_channels
        }


def _read_etc_file(path: str, env_var: str, label: str) -> str:
    """Read a value from an env var (priority) or /etc file."""
    value = os.environ.get(env_var, "").strip()
    if value:
        logger.debug("%s from env var %s: %s", label, env_var, value)
        return value
    try:
        value = Path(path).read_text().strip()
        logger.debug("%s from %s: %s", label, path, value)
        return value
    except FileNotFoundError:
        raise RuntimeError(
            f"{label} not found. "
            f"Either set {env_var} env var or run setup.sh to populate {path}."
        )


def load_config() -> ClientConfig:
    """
    Load and merge all config sources for the active client.
    Raises RuntimeError if any required value is missing.
    """
    client_name = _read_etc_file("/etc/nx_client", "NX_CLIENT", "Client name")
    dvr_ip      = _read_etc_file("/etc/nx_dvr_ip", "NX_DVR_IP", "DVR IP")

    # Sector resolution: /etc/nx_sector (written by setup.sh) > NX_SECTOR env > config.yaml
    raw_sector = os.environ.get("NX_SECTOR", "").strip()
    if not raw_sector:
        try:
            raw_sector = Path("/etc/nx_sector").read_text().strip()
        except FileNotFoundError:
            raw_sector = ""  # fallback to config.yaml value below

    client_dir = _REPO_ROOT / "clients" / client_name
    config_path = client_dir / "config.yaml"
    env_path    = client_dir / ".env"

    if not config_path.exists():
        raise RuntimeError(
            f"No config found for client '{client_name}'. "
            f"Expected: {config_path}"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = dotenv_values(env_path) if env_path.exists() else {}

    dvr_user = env.get("DVR_USER", os.environ.get("DVR_USER", ""))
    dvr_pass = env.get("DVR_PASS", os.environ.get("DVR_PASS", ""))

    if not dvr_user or not dvr_pass:
        raise RuntimeError(
            f"DVR credentials missing for client '{client_name}'. "
            f"Copy {client_dir}/.env.example to {client_dir}/.env and fill in the values."
        )

    # Pipeline resolution order:
    #   1. NX_PIPELINE env var  (e.g. docker compose run -e NX_PIPELINE=people_counting,age_gender)
    #   2. /etc/nx_pipeline     (written by setup.sh --package)
    #   3. config.yaml pipeline field (client default)
    raw_pipeline = os.environ.get("NX_PIPELINE", "").strip()
    if raw_pipeline:
        pipeline = [c.strip() for c in raw_pipeline.split(",") if c.strip()]
        logger.debug("Pipeline from NX_PIPELINE env var: %s", pipeline)
    else:
        try:
            file_val = Path("/etc/nx_pipeline").read_text().strip()
            pipeline = [c.strip() for c in file_val.split(",") if c.strip()]
            logger.debug("Pipeline from /etc/nx_pipeline: %s", pipeline)
        except FileNotFoundError:
            pipeline = cfg.get("pipeline", ["people_counting"])
            if isinstance(pipeline, str):
                pipeline = [p.strip() for p in pipeline.split(",") if p.strip()]
            logger.debug("Pipeline from config.yaml: %s", pipeline)

    if "people_counting" not in pipeline:
        pipeline = ["people_counting"] + pipeline

    unknown = set(pipeline) - VALID_CAPABILITIES
    if unknown:
        raise RuntimeError(
            f"Unknown pipeline capabilities: {unknown}\n"
            f"Valid options: {sorted(VALID_CAPABILITIES)}"
        )

    sector = raw_sector or cfg.get("sector", "comercio")

    entry_exit_channels = cfg.get("entry_exit_channels", [])
    if isinstance(entry_exit_channels, str):
        entry_exit_channels = [int(x.strip()) for x in entry_exit_channels.split(",") if x.strip()]

    config = ClientConfig(
        client_name=client_name,
        dvr_ip=dvr_ip,
        dvr_port=int(cfg.get("dvr_port", 554)),
        dvr_user=dvr_user,
        dvr_pass=dvr_pass,
        rtsp_url_pattern=cfg.get(
            "rtsp_url_pattern",
            "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/main/av_stream",
        ),
        channels=cfg.get("channels", [1]),
        pipeline=pipeline,
        stream_width=int(cfg.get("stream_width", 1920)),
        stream_height=int(cfg.get("stream_height", 1080)),
        tracker=cfg.get("tracker", "nvdcf"),
        sector=sector,
        entry_exit_channels=entry_exit_channels,
    )
    _warn_decoder_load(config)
    return config


# Jetson Orin Nano: 1 NVDEC engine rated at 8× 1080p30 H.264.
# Beyond 2× that limit (~16× 1080p30), expect cudaErrorIllegalAddress crashes.
_NVDEC_SAFE_MPIX  = 8  * 1920 * 1080   # 8× 1080p — rated capacity
_NVDEC_HARD_MPIX  = 16 * 1920 * 1080   # 16× 1080p — known crash threshold


def _warn_decoder_load(cfg: "ClientConfig") -> None:
    total = len(cfg.channels) * cfg.stream_width * cfg.stream_height
    if total > _NVDEC_HARD_MPIX:
        logger.error(
            "NVDEC OVERLOAD: %d streams × %dx%d = %.1f× the Orin Nano decoder capacity. "
            "Expect cudaErrorIllegalAddress crashes. "
            "Use sub-streams (subtype=1, 960×544) or distribute across multiple Jetsons.",
            len(cfg.channels), cfg.stream_width, cfg.stream_height,
            total / _NVDEC_SAFE_MPIX,
        )
    elif total > _NVDEC_SAFE_MPIX:
        logger.warning(
            "NVDEC near capacity: %d streams × %dx%d = %.1f× the Orin Nano rated load. "
            "Monitor for instability; use sub-streams (subtype=1) if crashes occur.",
            len(cfg.channels), cfg.stream_width, cfg.stream_height,
            total / _NVDEC_SAFE_MPIX,
        )
