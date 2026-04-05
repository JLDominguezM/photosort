import os
from pathlib import Path

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

THUMB_DIR = os.path.join(os.getenv("DATA_DIR", "data"), "thumbnails")
THUMB_SIZE = 300


def ensure_thumbnail(photo_id: int, source_path: str) -> str:
    os.makedirs(THUMB_DIR, exist_ok=True)
    thumb_path = os.path.join(THUMB_DIR, f"{photo_id}.jpg")
    if os.path.exists(thumb_path):
        return thumb_path
    try:
        with Image.open(source_path) as img:
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img = img.convert("RGB")
            img.save(thumb_path, "JPEG", quality=80)
    except Exception:
        return ""
    return thumb_path


def get_thumbnail_path(photo_id: int) -> str | None:
    path = os.path.join(THUMB_DIR, f"{photo_id}.jpg")
    return path if os.path.exists(path) else None
