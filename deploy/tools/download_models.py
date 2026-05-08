#!/usr/bin/env python3
"""
download_models.py — NX Computing AI
Downloads optional model files that are not tracked in git.

Usage:
  python tools/download_models.py --fall-detection
  python tools/download_models.py --reid
  python tools/download_models.py --face-recognition --ngc-key <key>
  python tools/download_models.py --all --ngc-key <key>

NGC API key (required for --face-recognition):
  Get a free key at https://ngc.nvidia.com/setup/api-key
  Pass via --ngc-key or the NGC_API_KEY environment variable.
"""
import argparse
import logging
import os
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


def download_osnet(dest_dir: Path):
    """
    OSNet-x0.25 (torchreid / KaiyangZhou), Apache 2.0.
    Input: NCHW float32 RGB ImageNet-normalized, 3×256×128.
    Output: (batch, 512) float32 embedding (AppearanceWorker L2-normalizes it).

    Exported on-the-fly from torchreid pretrained Market-1501 weights.
    Run this script natively on the Jetson (not inside Docker):
      pip3 install torchreid
      python3 tools/download_models.py --reid
    """
    dest = dest_dir / "osnet" / "osnet_x0_25_market1501.onnx"
    if dest.exists():
        logger.info("OSNet already exists — skipping.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torchreid  # noqa: F401
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Install with:  pip3 install torchreid")
        logger.error("Then re-run outside Docker: python3 tools/download_models.py --reid")
        sys.exit(1)

    logger.info("Exporting OSNet-x0.25 from torchreid pretrained weights (Market-1501)...")
    try:
        model = torchreid.models.build_model("osnet_x0_25", num_classes=1, pretrained=True)
        model.eval()
        dummy = torch.randn(1, 3, 256, 128)
        torch.onnx.export(
            model, dummy, str(dest),
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        logger.info("Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


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


def download_facedetectir(dest_dir: Path, ngc_key: str):
    """
    FaceDetectIR — ResNet-18 pruned+quantized ONNX from NVIDIA NGC.
    Input: 3×136×240 (BGR). Used as secondary GIE for face detection on person crops.
    Source: nvidia/tao/facedetectir, version pruned_quantized_v2.0 (NGC, requires API key).
    Get a free key at https://ngc.nvidia.com/setup/api-key
    """
    dest = dest_dir / "facedetect_ir" / "resnet18_facedetectir_pruned_quantized.onnx"
    if dest.exists():
        logger.info("FaceDetectIR already exists — skipping.")
        return

    if not ngc_key:
        logger.error("NGC API key required to download FaceDetectIR.")
        logger.error("Get a free key at: https://ngc.nvidia.com/setup/api-key")
        logger.error("Then run:  python3 tools/download_models.py --face-recognition --ngc-key <key>")
        logger.error("Or set:    NGC_API_KEY=<key> python3 tools/download_models.py --face-recognition")
        sys.exit(1)

    url = (
        "https://api.ngc.nvidia.com/v2/models/nvidia/tao/facedetectir"
        "/versions/pruned_quantized_v2.0/files/resnet18_facedetectir_pruned_quantized.onnx"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FaceDetectIR from NGC...")
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"ApiKey {ngc_key}"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            block_size = 65536
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(100, downloaded * 100 // total)
                        print(f"\r  {pct}%", end="", flush=True)
        print()
        logger.info("Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except urllib.error.HTTPError as e:
        logger.error("Download failed (HTTP %d): %s", e.code, e.reason)
        if e.code == 401:
            logger.error("Invalid NGC API key — verify at https://ngc.nvidia.com/setup/api-key")
        if dest.exists():
            dest.unlink()
        sys.exit(1)
    except Exception as e:
        logger.error("Download failed: %s", e)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download NX optional model files")
    parser.add_argument("--reid", action="store_true",
                        help="Download OSNet-x0.25 ONNX for cross-camera re-ID")
    parser.add_argument("--fall-detection", action="store_true",
                        help="Download MoveNet ONNX for fall detection (Hogar only)")
    parser.add_argument("--face-recognition", action="store_true",
                        help="Download FaceDetectIR ONNX from NGC (requires --ngc-key or NGC_API_KEY)")
    parser.add_argument("--ngc-key", default=None,
                        help="NGC API key for downloading FaceDetectIR "
                             "(also read from NGC_API_KEY env var)")
    parser.add_argument("--all", action="store_true",
                        help="Download all optional models (--ngc-key required for face-recognition)")
    args = parser.parse_args()

    if not any([args.reid, args.fall_detection, args.face_recognition, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.reid or args.all:
        download_osnet(_MODELS_DIR)

    if args.fall_detection or args.all:
        download_movenet(_MODELS_DIR)

    if args.face_recognition or args.all:
        ngc_key = args.ngc_key or os.environ.get("NGC_API_KEY", "")
        download_facedetectir(_MODELS_DIR, ngc_key)

    logger.info("Done.")


if __name__ == "__main__":
    main()
