"""
config_loader.py — NX Computing AI | Client Configuration Loader

Merges four sources at runtime (highest priority first):
  1. NX_PIPELINE env var  (or /etc/nx_pipeline)  → active pipeline capabilities
  2. NX_CLIENT env var    (or /etc/nx_client)     → which client folder to load
  3. NX_DVR_IP env var    (or /etc/nx_dvr_ip)     → DVR IP discovered by setup.sh
  4. clients/<name>/config.yaml                   → non-sensitive client config
  5. clients/<name>/.env                          → DVR_USER / DVR_PASS (gitignored)

Pipeline + sector are normally derived from the `package` field in config.yaml.
Set `package: manual` to specify them explicitly (for testing or custom deployments).

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

# Stream type → default resolution fed to nvstreammux.
# "main" — full-resolution main stream (1920×1080). OK for ≤6 cameras on Orin Nano.
# "sub"  — DVR sub-stream (960×544). Required for 16-camera deployments; also update
#           rtsp_url_pattern (subtype=1 for Dahua, Channels/{ch:02d}02 for Hikvision).
#           PeopleNet natively targets 960×544 — zero accuracy loss on detection.
#           SGIEs/workers see smaller crops; distant people may fall below min-size filters.
STREAM_TYPES = {
    "main": {"width": 1920, "height": 1080},
    "sub":  {"width": 960,  "height": 544},
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

# Package → pipeline capabilities + sector.
# Mirrors PACKAGE_CAPABILITIES in setup.sh. Set package: manual in config.yaml to
# specify pipeline and sector explicitly instead of deriving them from a package.
PACKAGE_DEFINITIONS = {
    "comercio_basico":      {"pipeline": ["people_counting"],                                                                "sector": "comercio"},
    "comercio_avanzado":    {"pipeline": ["people_counting"],                                                                "sector": "comercio"},
    "comercio_total":       {"pipeline": ["people_counting", "age_gender", "face_recognition"],                              "sector": "comercio"},
    "comercio_enterprise":  {"pipeline": ["people_counting", "age_gender", "face_recognition"],                              "sector": "comercio"},
    "industrial_basico":    {"pipeline": ["people_counting"],                                                                "sector": "industrial"},
    "industrial_avanzado":  {"pipeline": ["people_counting", "epp_detection"],                                               "sector": "industrial"},
    "industrial_total":     {"pipeline": ["people_counting", "epp_detection", "license_plate", "fire_smoke", "face_recognition"], "sector": "industrial"},
    "industrial_enterprise":{"pipeline": ["people_counting", "epp_detection", "license_plate", "fire_smoke", "face_recognition"], "sector": "industrial"},
    "hogar_basico":         {"pipeline": ["people_counting"],                                                                "sector": "hogar"},
    "hogar_avanzado":       {"pipeline": ["people_counting", "fall_detection"],                                              "sector": "hogar"},
    "hogar_total":          {"pipeline": ["people_counting", "fall_detection", "fire_smoke", "face_recognition"],            "sector": "hogar"},
    "manual":               None,  # pipeline and sector must be set explicitly in config.yaml
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
    tracker: str = "nvdcf"      # "nvdcf" (precise, ≤6 streams) | "iou" (stable, 16 streams)
    stream_type: str = "main"   # "main" (1920×1080, ≤6 cams) | "sub" (960×544, ≤16 cams)
    sector: str = "comercio"    # "comercio" | "industrial" | "hogar"
    package: str = "manual"     # contracted package; "manual" = custom/testing
    entry_exit_channels: List[int] = field(default_factory=list)
    pgie_batch_size: int = 0   # >0 = override nvinfer_config.txt at runtime; 0 = use file value
    pgie_interval: int = -1    # ≥0 = override nvinfer_config.txt at runtime; -1 = use file value

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
        logger.info("Package    : %s", self.package)
        logger.info("Sector     : %s", self.sector)
        logger.info("DVR        : %s:%d", self.dvr_ip, self.dvr_port)
        logger.info("Channels   : %s", self.channels)
        logger.info("Pipeline(s): %s", self.pipeline)
        logger.info("Resolution : %dx%d (%s)", self.stream_width, self.stream_height, self.stream_type)
        logger.info("Tracker    : %s", self.tracker)
        logger.info("PGIE batch : %s", self.pgie_batch_size or "(from nvinfer_config.txt)")
        logger.info("PGIE interval: %s", self.pgie_interval if self.pgie_interval >= 0 else "(from nvinfer_config.txt)")
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

    # ── Package resolution ────────────────────────────────────────────────────
    # Reads the contracted package from config.yaml and derives pipeline + sector.
    # "manual" skips derivation — pipeline and sector must be set explicitly.
    package = str(cfg.get("package", "manual")).strip()
    if package not in PACKAGE_DEFINITIONS:
        raise RuntimeError(
            f"Unknown package '{package}' in config.yaml.\n"
            f"Valid options: {sorted(PACKAGE_DEFINITIONS)}"
        )

    # ── Pipeline resolution order ─────────────────────────────────────────────
    #   1. NX_PIPELINE env var  (docker compose run -e NX_PIPELINE=... for one-off overrides)
    #   2. /etc/nx_pipeline     (written by setup.sh --package, used in production)
    #   3. package field        (derives pipeline from contracted package — recommended)
    #   4. pipeline field       (explicit list, only used when package: manual)
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
            if package != "manual":
                pipeline = list(PACKAGE_DEFINITIONS[package]["pipeline"])
                logger.debug("Pipeline from package '%s': %s", package, pipeline)
            else:
                raw = cfg.get("pipeline", ["people_counting"])
                if isinstance(raw, str):
                    raw = [p.strip() for p in raw.split(",") if p.strip()]
                pipeline = raw
                logger.debug("Pipeline from config.yaml (manual): %s", pipeline)

    if "people_counting" not in pipeline:
        pipeline = ["people_counting"] + pipeline

    unknown = set(pipeline) - VALID_CAPABILITIES
    if unknown:
        raise RuntimeError(
            f"Unknown pipeline capabilities: {unknown}\n"
            f"Valid options: {sorted(VALID_CAPABILITIES)}"
        )

    # ── Sector resolution order ───────────────────────────────────────────────
    #   1. NX_SECTOR env var  /  /etc/nx_sector  (written by setup.sh --package)
    #   2. package field (derives sector automatically)
    #   3. sector field in config.yaml (only when package: manual)
    raw_sector = os.environ.get("NX_SECTOR", "").strip()
    if not raw_sector:
        try:
            raw_sector = Path("/etc/nx_sector").read_text().strip()
        except FileNotFoundError:
            raw_sector = ""

    if raw_sector:
        sector = raw_sector
    elif package != "manual":
        sector = PACKAGE_DEFINITIONS[package]["sector"]
    else:
        sector = cfg.get("sector", "comercio")

    entry_exit_channels = cfg.get("entry_exit_channels", [])
    if isinstance(entry_exit_channels, str):
        entry_exit_channels = [int(x.strip()) for x in entry_exit_channels.split(",") if x.strip()]

    stream_type = cfg.get("stream_type", "main")
    if stream_type not in STREAM_TYPES:
        raise RuntimeError(
            f"Unknown stream_type '{stream_type}'. Valid options: {list(STREAM_TYPES)}"
        )
    _res = STREAM_TYPES[stream_type]

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
        stream_width=int(cfg.get("stream_width", _res["width"])),
        stream_height=int(cfg.get("stream_height", _res["height"])),
        tracker=cfg.get("tracker", "nvdcf"),
        stream_type=stream_type,
        sector=sector,
        package=package,
        entry_exit_channels=entry_exit_channels,
        pgie_batch_size=int(cfg.get("pgie_batch_size", 0)),
        pgie_interval=int(cfg.get("pgie_interval", -1)),
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
            "Set stream_type: sub in config.yaml (960×544, also update rtsp_url_pattern) "
            "or distribute across multiple Jetsons.",
            len(cfg.channels), cfg.stream_width, cfg.stream_height,
            total / _NVDEC_SAFE_MPIX,
        )
    elif total > _NVDEC_SAFE_MPIX:
        logger.warning(
            "NVDEC near capacity: %d streams × %dx%d = %.1f× the Orin Nano rated load. "
            "Set stream_type: sub in config.yaml if crashes occur.",
            len(cfg.channels), cfg.stream_width, cfg.stream_height,
            total / _NVDEC_SAFE_MPIX,
        )
