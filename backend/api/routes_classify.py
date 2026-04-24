import os
import shutil
import threading
import time

import numpy as np
import yaml
from fastapi import APIRouter, Depends, Request, Body, HTTPException

from api.deps import get_app_db
from models.schemas import ClassifyResult, CategoryOut, CategoriesConfig
from services.jobs import tracker
from services.logging_config import get_logger
from services.paths import safe_photo_path

router = APIRouter(prefix="/api", tags=["Classification"])

log = get_logger(__name__)

PHOTOS_BASE = os.path.realpath(os.getenv("PHOTOS_DIR", "/photos"))
CONFIG_DIR = os.getenv("CONFIG_DIR", "config")

# Serializes /classify batch runs — concurrent calls would race on writes
# to photos.clip_embedding and classifications. Acquired non-blockingly so
# a second caller gets a clean 409 instead of stacking up.
_classify_lock = threading.Lock()


@router.post("/classify", summary="Classify unclassified (or all) photos with CLIP")
def classify_photos(request: Request, force: bool = False, db=Depends(get_app_db)):
    """Run CLIP classification as an async job. With `force=true`,
    reclassifies every photo (deletes prior automatic classifications).
    Returns 409 if a classify job is already running."""
    if not _classify_lock.acquire(blocking=False):
        raise HTTPException(409, "A classify job is already running")

    clip = request.app.state.clip
    if force:
        rows = db.execute("SELECT id, filepath, clip_embedding FROM photos").fetchall()
    else:
        rows = db.execute(
            "SELECT id, filepath, clip_embedding FROM photos WHERE id NOT IN (SELECT photo_id FROM classifications)"
        ).fetchall()

    if not rows:
        _classify_lock.release()
        return {"message": "No photos to classify", "count": 0}

    job_id = tracker.create("classify")
    total = len(rows)
    tracker.update(job_id, 0, total)
    log.info("classify job %s started: total=%d force=%s", job_id, total, force)

    def _classify():
        try:
            from services.database import get_db
            conn = get_db()
            batch_size = clip.batch_size
            count = 0

            for i in range(0, total, batch_size):
                batch = rows[i:i + batch_size]
                for row in batch:
                    photo_id = row["id"]
                    filepath = row["filepath"]
                    embedding_blob = row["clip_embedding"]

                    if embedding_blob:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    else:
                        abs_path = safe_photo_path(PHOTOS_BASE, filepath)
                        if abs_path is None:
                            log.warning("skip classify, unsafe path: %r", filepath)
                            continue
                        is_video = filepath.lower().endswith(
                            (".mov", ".mp4", ".avi", ".mkv")
                        )
                        embedding = (
                            clip.encode_video(abs_path) if is_video
                            else clip.encode_image(abs_path)
                        )
                        if embedding is None:
                            continue
                        conn.execute(
                            "UPDATE photos SET clip_embedding = ? WHERE id = ?",
                            (embedding.tobytes(), photo_id),
                        )

                    category, confidence = clip.classify(embedding)
                    if force:
                        conn.execute(
                            "DELETE FROM classifications WHERE photo_id = ? AND is_manual = 0",
                            (photo_id,),
                        )
                    conn.execute(
                        """INSERT OR REPLACE INTO classifications (photo_id, category, confidence, is_manual)
                           VALUES (?, ?, ?, 0)""",
                        (photo_id, category, confidence),
                    )
                    count += 1

                conn.commit()
                tracker.update(job_id, min(i + batch_size, total), total)

            tracker.complete(job_id, {"classified": count})
            log.info("classify job %s finished: classified=%d/%d", job_id, count, total)
            conn.close()
        except Exception as e:
            log.exception("classify job %s failed: %s", job_id, e)
            tracker.fail(job_id, str(e))
        finally:
            _classify_lock.release()

    threading.Thread(target=_classify, daemon=True).start()
    return {"job_id": job_id, "total": total}


@router.post("/classify/{photo_id}", response_model=ClassifyResult, summary="Classify a single photo")
def classify_single(photo_id: int, request: Request, db=Depends(get_app_db)):
    """Classify one photo synchronously and return its top category."""
    clip = request.app.state.clip
    row = db.execute("SELECT filepath, clip_embedding FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise HTTPException(404)

    if row["clip_embedding"]:
        embedding = np.frombuffer(row["clip_embedding"], dtype=np.float32)
    else:
        abs_path = safe_photo_path(PHOTOS_BASE, row["filepath"])
        if abs_path is None:
            raise HTTPException(403, "Invalid path")
        embedding = clip.encode_image(abs_path)
        if embedding is None:
            raise HTTPException(500, "Could not encode image")
        db.execute("UPDATE photos SET clip_embedding = ? WHERE id = ?", (embedding.tobytes(), photo_id))
        db.commit()

    category, confidence = clip.classify(embedding)
    db.execute(
        "INSERT OR REPLACE INTO classifications (photo_id, category, confidence) VALUES (?, ?, ?)",
        (photo_id, category, confidence),
    )
    db.commit()
    return ClassifyResult(photo_id=photo_id, category=category, confidence=confidence)


@router.put("/photos/{photo_id}/category", summary="Manually set a photo's category")
def set_category(photo_id: int, category: str = Body(..., embed=True), db=Depends(get_app_db)):
    """Override automatic classification for a photo. The manual assignment
    has confidence 1.0 and is preserved across future `classify --force` runs."""
    row = db.execute("SELECT id FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    db.execute("DELETE FROM classifications WHERE photo_id = ?", (photo_id,))
    db.execute(
        "INSERT INTO classifications (photo_id, category, confidence, is_manual) VALUES (?, ?, 1.0, 1)",
        (photo_id, category),
    )
    db.commit()
    return {"photo_id": photo_id, "category": category}


@router.get("/categories", response_model=list[CategoryOut], summary="List categories with photo counts")
def list_categories(db=Depends(get_app_db)):
    """Return each category currently used in the library and how many photos
    have been assigned to it."""
    rows = db.execute(
        """SELECT category, COUNT(*) as count FROM classifications
           GROUP BY category ORDER BY count DESC"""
    ).fetchall()
    return [CategoryOut(name=r["category"], count=r["count"]) for r in rows]


@router.put("/categories", summary="Replace the categories configuration")
def update_categories(request: Request, payload: CategoriesConfig):
    """Validate and persist a new `categories.yml`. The previous file is
    backed up to `categories.yml.bak.<timestamp>` and CLIP text embeddings
    are recomputed for the new categories. Run `classify --force` afterwards
    to reapply to existing photos."""
    config_path = os.path.join(CONFIG_DIR, "categories.yml")
    if os.path.exists(config_path):
        backup = f"{config_path}.bak.{int(time.time())}"
        try:
            shutil.copy2(config_path, backup)
            log.info("Backed up previous categories.yml to %s", backup)
        except OSError as e:
            log.warning("Could not backup categories.yml: %s", e)
    data = payload.model_dump()
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    except OSError as e:
        log.error("Failed to write categories.yml: %s", e)
        raise HTTPException(500, "Could not persist categories")
    request.app.state.clip.load_categories(config_path)
    log.info("Categories updated: %d entries", len(payload.categories))
    return {"message": "Categories updated", "count": len(payload.categories)}
