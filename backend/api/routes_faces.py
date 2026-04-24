import os
import threading

import numpy as np
from fastapi import APIRouter, Depends, Request, HTTPException, Body

from api.deps import get_app_db
from models.schemas import PersonOut
from services.jobs import tracker
from services.logging_config import get_logger
from services.paths import safe_photo_path

router = APIRouter(prefix="/api")

log = get_logger(__name__)

PHOTOS_BASE = os.path.realpath(os.getenv("PHOTOS_DIR", "/photos"))


def _safe_path(filepath: str) -> str | None:
    resolved = safe_photo_path(PHOTOS_BASE, filepath)
    if resolved is None:
        log.warning("Rejected path traversal in faces: %r", filepath)
    return resolved


@router.post("/faces/detect")
def detect_faces(request: Request, db=Depends(get_app_db)):
    face_engine = request.app.state.faces
    rows = db.execute(
        "SELECT id, filepath FROM photos WHERE id NOT IN (SELECT DISTINCT photo_id FROM faces) AND filepath NOT LIKE '%.mov' AND filepath NOT LIKE '%.mp4' AND filepath NOT LIKE '%.avi'"
    ).fetchall()

    if not rows:
        return {"message": "No new photos to process", "count": 0}

    job_id = tracker.create("face_detect")
    total = len(rows)
    tracker.update(job_id, 0, total)

    def _detect():
        from services.database import get_db
        conn = get_db()
        count = 0
        for i, row in enumerate(rows):
            abs_path = _safe_path(row["filepath"])
            if abs_path is None:
                tracker.update(job_id, i + 1, total)
                continue
            try:
                faces = face_engine.detect_faces(abs_path)
                for face in faces:
                    conn.execute(
                        """INSERT INTO faces (photo_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (row["id"], face["bbox_x"], face["bbox_y"],
                         face["bbox_w"], face["bbox_h"],
                         face["embedding"].astype(np.float32).tobytes()),
                    )
                    count += 1
                conn.commit()
            except Exception as e:
                log.warning("face detect failed for %s: %s", row["filepath"], e)
            tracker.update(job_id, i + 1, total)
        tracker.complete(job_id, {"faces_detected": count})
        log.info("face_detect job %s finished: faces=%d photos=%d", job_id, count, total)
        conn.close()

    threading.Thread(target=_detect, daemon=True).start()
    return {"job_id": job_id, "total": total}


@router.post("/faces/cluster")
def cluster_faces(request: Request, db=Depends(get_app_db)):
    face_engine = request.app.state.faces
    rows = db.execute("SELECT id, embedding FROM faces WHERE person_id IS NULL").fetchall()
    if not rows:
        return {"message": "No unclustered faces"}

    face_ids = [r["id"] for r in rows]
    embeddings = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    labels = face_engine.cluster_faces(embeddings)

    label_to_person = {}
    for face_id, label in zip(face_ids, labels):
        if label == -1:
            continue
        if label not in label_to_person:
            db.execute("INSERT INTO persons (name) VALUES (NULL)")
            db.commit()
            person_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            label_to_person[label] = person_id
        db.execute("UPDATE faces SET person_id = ? WHERE id = ?", (label_to_person[label], face_id))

    db.commit()
    return {"clusters_created": len(label_to_person), "faces_assigned": sum(1 for l in labels if l != -1)}


@router.get("/faces/persons", response_model=list[PersonOut])
def list_persons(db=Depends(get_app_db)):
    rows = db.execute(
        """SELECT p.id, p.name,
                  COUNT(f.id) as face_count,
                  COUNT(DISTINCT f.photo_id) as photo_count
           FROM persons p
           LEFT JOIN faces f ON f.person_id = p.id
           GROUP BY p.id
           ORDER BY face_count DESC"""
    ).fetchall()
    return [PersonOut(**dict(r)) for r in rows]


@router.get("/faces/persons/{person_id}")
def get_person_photos(person_id: int, db=Depends(get_app_db)):
    person = db.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    if not person:
        raise HTTPException(404)

    photos = db.execute(
        """SELECT DISTINCT p.*, c.category, c.confidence
           FROM photos p
           JOIN faces f ON f.photo_id = p.id
           LEFT JOIN classifications c ON c.photo_id = p.id
               AND c.confidence = (SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = p.id)
           WHERE f.person_id = ?
           ORDER BY p.taken_at DESC""",
        (person_id,),
    ).fetchall()

    return {
        "person": {"id": person["id"], "name": person["name"]},
        "photos": [dict(p) for p in photos],
    }


@router.put("/faces/persons/{person_id}")
def name_person(person_id: int, name: str = Body(..., embed=True), db=Depends(get_app_db)):
    db.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
    db.commit()
    return {"person_id": person_id, "name": name}


@router.post("/faces/persons/merge")
def merge_persons(person_a: int = Body(...), person_b: int = Body(...), db=Depends(get_app_db)):
    if person_a == person_b:
        raise HTTPException(400, "Cannot merge a person with themselves")
    existing = {r["id"] for r in db.execute(
        "SELECT id FROM persons WHERE id IN (?, ?)", (person_a, person_b)
    ).fetchall()}
    if person_a not in existing or person_b not in existing:
        raise HTTPException(404, "Person not found")
    db.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (person_a, person_b))
    db.execute("DELETE FROM persons WHERE id = ?", (person_b,))
    db.commit()
    log.info("merged person %s into %s", person_b, person_a)
    return {"merged_into": person_a, "deleted": person_b}


@router.get("/faces/{photo_id}/crops")
def get_face_crops(photo_id: int, db=Depends(get_app_db)):
    faces = db.execute(
        "SELECT id, bbox_x, bbox_y, bbox_w, bbox_h, person_id FROM faces WHERE photo_id = ?",
        (photo_id,),
    ).fetchall()
    return [dict(f) for f in faces]
