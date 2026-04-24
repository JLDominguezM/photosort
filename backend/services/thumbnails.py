import os
from pathlib import Path

from PIL import Image
import pillow_heif

from services.logging_config import get_logger

pillow_heif.register_heif_opener()

log = get_logger(__name__)

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
    except Exception as e:
        log.warning("thumbnail failed for photo_id=%s path=%s: %s", photo_id, source_path, e)
        return ""
    return thumb_path


def get_thumbnail_path(photo_id: int) -> str | None:
    path = os.path.join(THUMB_DIR, f"{photo_id}.jpg")
    return path if os.path.exists(path) else None
