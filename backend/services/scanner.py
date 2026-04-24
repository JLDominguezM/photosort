import os
from pathlib import Path

import exifread
from PIL import Image
import pillow_heif

from services.logging_config import get_logger

pillow_heif.register_heif_opener()

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/photos")


def walk_photos(base_dir: str | None = None) -> list[dict]:
    base = Path(base_dir or PHOTOS_DIR)
    results = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
            continue
        rel = str(path.relative_to(base))
        results.append({
            "filepath": rel,
            "filename": path.name,
            "filesize": path.stat().st_size,
            "absolute": str(path),
            "is_video": ext in VIDEO_EXTENSIONS,
        })
    return results


def extract_exif(filepath: str) -> dict:
    info = {"width": None, "height": None, "taken_at": None}
    try:
        with open(filepath, "rb") as f:
            tags = exifread.process_file(f, stop_tag="DateTimeOriginal", details=False)
        dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if dt:
            info["taken_at"] = str(dt).replace(":", "-", 2)
    except Exception as e:
        log.debug("EXIF read failed for %s: %s", filepath, e)
    try:
        with Image.open(filepath) as img:
            info["width"], info["height"] = img.size
    except Exception as e:
        log.debug("Pillow open failed for %s: %s", filepath, e)
    return info


def get_new_photos(db_conn, base_dir: str | None = None) -> list[dict]:
    all_files = walk_photos(base_dir)
    existing = {row[0] for row in db_conn.execute("SELECT filepath FROM photos").fetchall()}
    return [f for f in all_files if f["filepath"] not in existing]
