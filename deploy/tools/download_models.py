#!/usr/bin/env python3
"""
download_models.py — NX Computing AI
Downloads optional model files that are not tracked in git.

Usage:
  python tools/download_models.py --fall-detection
  python tools/download_models.py --all
"""
import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "models"


def _download(url: str, dest: Path, label: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("%s already exists — skipping.", dest.name)
        return
    logger.info("Downloading %s ...", label)
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        logger.info("Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except Exception as e:
        logger.error("Download failed: %s", e)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def _progress(block, block_size, total):
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)


def download_movenet(dest_dir: Path):
    """
    MoveNet SinglePose Lightning — ONNX, 192×192 input.
    Source: PINTO0309 model zoo (community ONNX export of Google MoveNet).
    Model is ~2 MB, runs at ~30 FPS on Jetson Orin Nano CPU / ~60 FPS with GPU EP.

    Alternative sources if the URL below becomes unavailable:
      - https://www.kaggle.com/models/google/movenet/frameworks/tfLite (convert to ONNX)
      - https://github.com/PINTO0309/PINTO_model_zoo/tree/main/115_MoveNet
    """
    dest = dest_dir / "movenet" / "movenet_singlepose_lightning_192.onnx"
    url = (
        "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/115_MoveNet/"
        "movenet_singlepose_lightning_192/movenet_singlepose_lightning_192.onnx"
    )
    _download(url, dest, "MoveNet SinglePose Lightning (ONNX 192×192)")


def main():
    parser = argparse.ArgumentParser(description="Download NX optional model files")
    parser.add_argument("--fall-detection", action="store_true",
                        help="Download MoveNet ONNX for fall detection")
    parser.add_argument("--all", action="store_true",
                        help="Download all optional models")
    args = parser.parse_args()

    if not any([args.fall_detection, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.fall_detection or args.all:
        download_movenet(_MODELS_DIR)

    logger.info("Done.")


if __name__ == "__main__":
    main()
