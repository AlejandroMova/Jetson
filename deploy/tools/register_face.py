#!/usr/bin/env python3
"""
register_face.py — NX Computing AI | Face Registration CLI

Generates ArcFace embeddings from photos and saves them to known_faces.json.
Run on your laptop — photos never leave your machine.

Usage:
  # Register from a single image:
  python tools/register_face.py --name "Juan Perez" --image foto.jpg

  # Register from a video (extracts N frames):
  python tools/register_face.py --name "Juan Perez" --video clip.mp4 --n 5

  # Import a full folder structure at once:
  #   clients/demo/faces/
  #     Juan Perez/foto1.jpg
  #     Ana Lopez/selfie.jpg
  python tools/register_face.py --import-dir clients/demo/faces/ --client demo

  # List registered persons:
  python tools/register_face.py --list --client demo

  # Delete a person:
  python tools/register_face.py --delete "Juan Perez" --client demo
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INSIGHTFACE_ROOT = str(_REPO_ROOT / "models" / "insightface")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_insightface():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            root=_INSIGHTFACE_ROOT,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except ImportError:
        logger.error("insightface not installed. Run: pip install insightface onnxruntime")
        sys.exit(1)


def _db_path(client: str) -> Path:
    return _REPO_ROOT / "clients" / client / "known_faces.json"


def _load_db(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_db(path: Path, db: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db, indent=2))
    logger.info("Saved: %s (%d person(s))", path, len(db))


def _extract_embedding(app, img_bgr: np.ndarray) -> np.ndarray:
    faces = app.get(img_bgr)
    if not faces:
        return None
    return faces[0].normed_embedding.tolist()


def cmd_register_image(app, name: str, image_path: str, client: str):
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Cannot read image: %s", image_path)
        sys.exit(1)
    emb = _extract_embedding(app, img)
    if emb is None:
        logger.error("No face detected in: %s", image_path)
        sys.exit(1)
    path = _db_path(client)
    db = _load_db(path)
    db.setdefault(name, []).append(emb)
    _save_db(path, db)
    logger.info("Registered '%s' from %s (%d embedding(s) total)", name, image_path, len(db[name]))


def cmd_register_video(app, name: str, video_path: str, n: int, client: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // n)
    embeddings = []
    frame_idx = 0
    while len(embeddings) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        emb = _extract_embedding(app, frame)
        if emb is not None:
            embeddings.append(emb)
            logger.info("  Frame %d: embedding extracted", frame_idx)
        frame_idx += step
    cap.release()
    if not embeddings:
        logger.error("No faces found in video: %s", video_path)
        sys.exit(1)
    path = _db_path(client)
    db = _load_db(path)
    db.setdefault(name, []).extend(embeddings)
    _save_db(path, db)
    logger.info("Registered '%s' from video: %d embedding(s) added", name, len(embeddings))


def cmd_import_dir(app, import_dir: str, client: str):
    root = Path(import_dir)
    if not root.is_dir():
        logger.error("Directory not found: %s", import_dir)
        sys.exit(1)
    path = _db_path(client)
    db = _load_db(path)
    total_added = 0
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        person_embeddings = []
        for img_file in sorted(person_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning("  Cannot read %s — skipping", img_file.name)
                continue
            emb = _extract_embedding(app, img)
            if emb is not None:
                person_embeddings.append(emb)
            else:
                logger.warning("  No face in %s — skipping", img_file.name)
        if person_embeddings:
            db.setdefault(name, []).extend(person_embeddings)
            total_added += len(person_embeddings)
            logger.info("  '%s': %d embedding(s)", name, len(person_embeddings))
        else:
            logger.warning("  '%s': no valid faces found — skipping", name)
    _save_db(path, db)
    logger.info("Import complete: %d embedding(s) added across %d person(s)",
                total_added, len(db))


def cmd_list(client: str):
    path = _db_path(client)
    db = _load_db(path)
    if not db:
        print(f"No persons registered in {path}")
        return
    print(f"\nRegistered persons in {path}:")
    for name, embeddings in sorted(db.items()):
        print(f"  {name}: {len(embeddings)} embedding(s)")
    print()


def cmd_delete(name: str, client: str):
    path = _db_path(client)
    db = _load_db(path)
    if name not in db:
        logger.error("'%s' not found in DB", name)
        sys.exit(1)
    del db[name]
    _save_db(path, db)
    logger.info("Deleted '%s'", name)


def main():
    parser = argparse.ArgumentParser(description="NX face registration tool")
    parser.add_argument("--client", default="demo",
                        help="Client name (default: demo)")
    parser.add_argument("--name", help="Person name")
    parser.add_argument("--image", help="Image file path")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--n", type=int, default=5,
                        help="Frames to sample from video (default: 5)")
    parser.add_argument("--import-dir",
                        help="Import all subfolders (folder name = person name)")
    parser.add_argument("--list", action="store_true",
                        help="List registered persons")
    parser.add_argument("--delete", metavar="NAME",
                        help="Delete a person from the DB")
    args = parser.parse_args()

    if args.list:
        cmd_list(args.client)
        return

    if args.delete:
        cmd_delete(args.delete, args.client)
        return

    if args.import_dir:
        app = _load_insightface()
        cmd_import_dir(app, args.import_dir, args.client)
        return

    if args.image:
        if not args.name:
            parser.error("--name is required with --image")
        app = _load_insightface()
        cmd_register_image(app, args.name, args.image, args.client)
        return

    if args.video:
        if not args.name:
            parser.error("--name is required with --video")
        app = _load_insightface()
        cmd_register_video(app, args.name, args.video, args.n, args.client)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
