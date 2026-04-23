"""
config_loader.py — NX Computing AI | Client Configuration Loader

Merges three sources at runtime:
  1. /etc/nx_client   (or NX_CLIENT env var)   → which client folder to load
  2. /etc/nx_dvr_ip   (or NX_DVR_IP env var)   → DVR IP discovered by setup.sh
  3. clients/<name>/config.yaml                 → non-sensitive client config
  4. clients/<name>/.env                        → DVR_USER / DVR_PASS (gitignored)

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
            )
            urls.append(url)
        return urls

    def log_summary(self):
        logger.info("Client     : %s", self.client_name)
        logger.info("DVR        : %s:%d", self.dvr_ip, self.dvr_port)
        logger.info("Channels   : %s", self.channels)
        logger.info("Pipeline(s): %s", self.pipeline)
        logger.info("Resolution : %dx%d", self.stream_width, self.stream_height)
        # Log URLs but mask the password
        for url in self.rtsp_urls():
            masked = url.replace(self.dvr_pass, "***") if self.dvr_pass else url
            logger.info("RTSP URL   : %s", masked)


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

    pipeline = cfg.get("pipeline", ["people_counting"])
    if isinstance(pipeline, str):
        pipeline = [pipeline]

    return ClientConfig(
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
    )
