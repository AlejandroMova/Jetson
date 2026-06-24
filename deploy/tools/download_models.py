#!/usr/bin/env python3
"""
download_models.py — NX Computing AI
Downloads optional model files that are not tracked in git.

Usage:
  python tools/download_models.py --fall-detection
  python tools/download_models.py --reid
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
    """Descarga un archivo desde `url` a `dest`, mostrando progreso en consola.

    Idempotente: si el archivo ya existe, lo omite sin error.
    Si la descarga falla, elimina el archivo parcial y sale con código 1
    para que setup.sh detecte el fallo y no continúe con un modelo corrupto.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)  # crear directorios intermedios si no existen

    # Si ya descargamos este modelo antes, no volver a descargar
    if dest.exists():
        logger.info("%s already exists — skipping.", dest.name)
        return

    logger.info("Downloading %s ...", label)
    try:
        # urlretrieve llama a _progress en cada bloque recibido
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # nueva línea tras la barra de progreso inline
        logger.info("Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except Exception as e:
        logger.error("Download failed: %s", e)
        # Eliminar archivo parcial para evitar que el pipeline arranque con un modelo corrupto
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def _progress(block, block_size, total):
    """Callback de progreso para urlretrieve — imprime porcentaje en la misma línea."""
    if total > 0:
        # Calcular porcentaje sin superar 100% (el último bloque puede exceder ligeramente)
        pct = min(100, block * block_size * 100 // total)
        print(f"\r  {pct}%", end="", flush=True)


def download_osnet(dest_dir: Path):
    """
    OSNet-x1.0 (torchreid / KaiyangZhou), Apache 2.0.
    Input: NCHW float32 RGB ImageNet-normalized, 3×256×128.
    Output: (batch, 512) float32 embedding (AppearanceWorker L2-normalizes it).

    OSNet-x1.0 achieves ~94% Rank-1 on Market-1501 vs ~82% for the x0.25 variant.
    At 2.2M params it remains lightweight and fits on GPU alongside PeopleNet.

    Must be exported inside the Docker container where torch/torchreid are installed:
      docker compose run --rm deepstream python3 tools/download_models.py --reid
    """
    dest = dest_dir / "osnet" / "osnet_x1_0_market1501.onnx"
    if dest.exists():
        logger.info("OSNet already exists — skipping.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torchreid  # noqa: F401
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        logger.error("Run inside container: docker compose run --rm deepstream python3 tools/download_models.py --reid")
        sys.exit(1)

    logger.info("Exporting OSNet-x1.0 from torchreid pretrained weights (Market-1501)...")
    try:
        model = torchreid.models.build_model("osnet_x1_0", num_classes=1, pretrained=True)
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


def main():
    """CLI principal — parsea argumentos y llama a las funciones de descarga correspondientes.

    Ejemplo de uso típico desde setup.sh:
      python3 tools/download_models.py --all
    """
    parser = argparse.ArgumentParser(description="Download NX optional model files")
    parser.add_argument("--reid", action="store_true",
                        help="Export OSNet-x1.0 ONNX for cross-camera re-ID")
    parser.add_argument("--fall-detection", action="store_true",
                        help="Download MoveNet ONNX for fall detection (Hogar only)")
    parser.add_argument("--all", action="store_true",
                        help="Download all optional models")
    args = parser.parse_args()

    if not any([args.reid, args.fall_detection, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.reid or args.all:
        download_osnet(_MODELS_DIR)

    if args.fall_detection or args.all:
        download_movenet(_MODELS_DIR)

    logger.info("Done.")


if __name__ == "__main__":
    main()
