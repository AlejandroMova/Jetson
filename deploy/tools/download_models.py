#!/usr/bin/env python3
"""
download_models.py — NX Computing AI
Downloads optional model files that are not tracked in git.

Usage:
  python tools/download_models.py --fall-detection
  python tools/download_models.py --reid --github-token <token>
  python tools/download_models.py --tracker-reid
  python tools/download_models.py --all --github-token <token>

The GitHub token is extracted automatically by setup.sh from the git remote URL
(the same token used to clone the repo). For manual runs, pass it with --github-token.
"""
import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "models"


def _download(url: str, dest: Path, label: str, token: str = ""):
    """Download a file from `url` to `dest`, showing progress.

    Idempotent: skips if the file already exists.
    Exits with code 1 on failure so setup.sh does not continue with a corrupt model.

    If `token` is provided it is sent as a GitHub Bearer token — required for
    assets hosted in private GitHub Releases. The token is the same one used to
    clone the repo (extracted automatically from the git remote URL in setup.sh).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        logger.info("%s already exists — skipping.", dest.name)
        return

    logger.info("Downloading %s ...", label)
    try:
        if token:
            # Private GitHub Release: send auth header, then follow redirect to S3
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/octet-stream",
                },
            )
            with urllib.request.urlopen(req) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(dest, "wb") as f:
                    while chunk := resp.read(65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = min(100, downloaded * 100 // total)
                            print(f"\r  {pct}%", end="", flush=True)
            print()
        else:
            # Public URL — use urlretrieve for simpler progress reporting
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print()

        logger.info("Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except Exception as e:
        logger.error("Download failed: %s", e)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def _progress(block, block_size, total):
    """Progress callback for urlretrieve — prints percentage on the same line."""
    if total > 0:
        pct = min(100, block * block_size * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)


def download_osnet(dest_dir: Path, token: str = ""):
    """
    OSNet-x1.0 (torchreid / KaiyangZhou), Apache 2.0.
    Input: NCHW float32 RGB ImageNet-normalized, 3×256×128.
    Output: (batch, 512) float32 embedding (AppearanceWorker L2-normalizes it).

    OSNet-x1.0 achieves ~94% Rank-1 on Market-1501 vs ~82% for the x0.25 variant.
    At 2.2M params it remains lightweight and fits on GPU alongside PeopleNet.

    Downloaded from GitHub Releases — no torch or torchreid required on the Jetson.
    Requires a GitHub token because the repo is private (passed via --github-token).
    To regenerate the ONNX file: deploy/tools/export_osnet_colab.ipynb
    """
    dest = dest_dir / "osnet" / "osnet_x1_0_market1501.onnx"
    url = (
        "https://github.com/AlejandroMova/NX-JETSON/releases/download/models-v1/"
        "osnet_x1_0_market1501.onnx"
    )
    _download(url, dest, "OSNet-x1.0 ONNX (cross-camera re-ID)", token=token)


def download_tracker_reid(dest_dir: Path):
    """
    ReIdentificationNet (NVIDIA TAO Toolkit, resnet50_market1501), deployable_v1.0.
    TAO-encoded model (.etlt, key "nvidia_tao") — used by the NvDCF tracker's own
    ReID/Re-Assoc submodule (intra-camera track_id recovery after occlusion), NOT
    by our cross-camera OSNet SGIE. See config_tracker_NvDCF_reid.yml for the
    matching preprocessing values (inferDims, offsets, netScaleFactor) confirmed
    from NVIDIA's stock config_tracker_NvDCF_accuracy.yml.

    Public NGC download — no API key/token required (verified: plain GET redirects
    to a signed URL, ~92 MB).
    """
    dest = dest_dir / "tracker" / "resnet50_market1501.etlt"
    url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/"
        "versions/deployable_v1.0/files/resnet50_market1501.etlt"
    )
    _download(url, dest, "ReIdentificationNet resnet50 (NvDCF tracker ReID)")


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
    """CLI entry point — parses args and calls the corresponding download functions.

    Typical usage from setup.sh:
      python3 tools/download_models.py --reid --github-token <token>
      python3 tools/download_models.py --all  --github-token <token>
    """
    parser = argparse.ArgumentParser(description="Download NX optional model files")
    parser.add_argument("--reid", action="store_true",
                        help="Download OSNet-x1.0 ONNX for cross-camera re-ID")
    parser.add_argument("--tracker-reid", action="store_true",
                        help="Download ReIdentificationNet .etlt for NvDCF tracker ReID (intra-camera)")
    parser.add_argument("--fall-detection", action="store_true",
                        help="Download MoveNet ONNX for fall detection (Hogar only)")
    parser.add_argument("--all", action="store_true",
                        help="Download all optional models")
    parser.add_argument("--github-token", default="",
                        help="GitHub personal access token for private release assets")
    args = parser.parse_args()

    if not any([args.reid, args.tracker_reid, args.fall_detection, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.reid or args.all:
        download_osnet(_MODELS_DIR, token=args.github_token)

    if args.tracker_reid or args.all:
        download_tracker_reid(_MODELS_DIR)

    if args.fall_detection or args.all:
        download_movenet(_MODELS_DIR)

    logger.info("Done.")


if __name__ == "__main__":
    main()
