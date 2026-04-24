import os
import threading

import numpy as np
from fastapi import APIRouter, Depends, Request, HTTPException, Body

from api.deps import get_app_db
from models.schemas import PersonOut
from services.face_engine import FaceEngine
from services.jobs import tracker
from services.logging_config import get_logger
from services.paths import safe_photo_path

router = APIRouter(prefix="/api", tags=["Faces"])

log = get_logger(__name__)

PHOTOS_BASE = os.path.realpath(os.getenv("PHOTOS_DIR", "/photos"))


def _safe_path(filepath: str) -> str | None:
    resolved = safe_photo_path(PHOTOS_BASE, filepath)
    if resolved is None:
        log.warning("Rejected path traversal in faces: %r", filepath)
    return resolved


def _recompute_centroid(conn, person_id: int) -> None:
    """Recompute a person's centroid as the L2-normalized mean of their face
    embeddings. Called after cluster creation and after merges."""
    rows = conn.execute(
        "SELECT embedding FROM faces WHERE person_id = ?", (person_id,)
    ).fetchall()
    if not rows:
        conn.execute("UPDATE persons SET centroid = NULL WHERE id = ?", (person_id,))
        return
    embeddings = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    centroid = FaceEngine.compute_centroid(embeddings)
    conn.execute(
        "UPDATE persons SET centroid = ? WHERE id = ?",
        (centroid.tobytes() if centroid is not None else None, person_id),
    )


@router.post("/faces/detect", summary="Detect faces in photos not yet processed")
def detect_faces(request: Request, db=Depends(get_app_db)):
    """Run InsightFace detection + embedding on every photo without existing
    face rows. Videos are skipped. Runs asynchronously; poll
    /api/jobs/{job_id}."""
    face_engine = request.app.state.faces
    rows = db.execute(
        "SELECT id, filepath FROM photos WHERE id NOT IN (SELECT DISTINCT photo_id FROM faces) AND filepath NOT LIKE '%.mov' AND filepath NOT LIKE '%.mp4' AND filepath NOT LIKE '%.avi'"
    ).fetchall()

    if not rows:
        return {"message": "No new photos to process", "count": 0}

    job_id = tracker.create("face_detect")
    total = len(rows)
    tracker.update(job_id, 0, total)
    log.info("face_detect job %s started: total=%d", job_id, total)

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


@router.post("/faces/cluster", summary="Group unassigned faces into persons using DBSCAN")
def cluster_faces(request: Request, db=Depends(get_app_db)):
    """Cluster every face with person_id IS NULL by cosine distance on the
    face embedding. Each cluster becomes a new `persons` row with its
    centroid pre-computed."""
    face_engine = request.app.state.faces
    rows = db.execute("SELECT id, embedding FROM faces WHERE person_id IS NULL").fetchall()
    if not rows:
        return {"message": "No unclustered faces"}

    face_ids = [r["id"] for r in rows]
    embeddings = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    labels = face_engine.cluster_faces(embeddings)

    label_to_person: dict[int, int] = {}
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

    for person_id in label_to_person.values():
        _recompute_centroid(db, person_id)
    db.commit()

    assigned = sum(1 for l in labels if l != -1)
    log.info("cluster_faces: created %d persons, assigned %d faces", len(label_to_person), assigned)
    return {"clusters_created": len(label_to_person), "faces_assigned": assigned}


@router.get("/faces/persons", response_model=list[PersonOut], summary="List detected persons")
def list_persons(db=Depends(get_app_db)):
    """Return every person with counts of faces and distinct photos."""
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


@router.get("/faces/persons/{person_id}", summary="List photos of a person")
def get_person_photos(person_id: int, db=Depends(get_app_db)):
    """Return every photo that has at least one face assigned to this person."""
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


@router.put("/faces/persons/{person_id}", summary="Rename a person")
def name_person(person_id: int, name: str = Body(..., embed=True), db=Depends(get_app_db)):
    """Assign a human-readable name to a person cluster."""
    db.execute("UPDATE persons SET name = ? WHERE id = ?", (name, person_id))
    db.commit()
    return {"person_id": person_id, "name": name}


@router.post("/faces/persons/merge", summary="Merge two person clusters")
def merge_persons(person_a: int = Body(...), person_b: int = Body(...), db=Depends(get_app_db)):
    """Reassign every face from `person_b` to `person_a`, then delete
    `person_b` and recompute `person_a`'s centroid over the combined set."""
    if person_a == person_b:
        raise HTTPException(400, "Cannot merge a person with themselves")
    existing = {r["id"] for r in db.execute(
        "SELECT id FROM persons WHERE id IN (?, ?)", (person_a, person_b)
    ).fetchall()}
    if person_a not in existing or person_b not in existing:
        raise HTTPException(404, "Person not found")
    db.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (person_a, person_b))
    db.execute("DELETE FROM persons WHERE id = ?", (person_b,))
    _recompute_centroid(db, person_a)
    db.commit()
    log.info("merged person %s into %s (centroid recomputed)", person_b, person_a)
    return {"merged_into": person_a, "deleted": person_b}


@router.get("/faces/{photo_id}/crops", summary="Face bounding boxes for a photo")
def get_face_crops(photo_id: int, db=Depends(get_app_db)):
    """Return each face's bbox and the person_id it was assigned to."""
    faces = db.execute(
        "SELECT id, bbox_x, bbox_y, bbox_w, bbox_h, person_id FROM faces WHERE photo_id = ?",
        (photo_id,),
    ).fetchall()
    return [dict(f) for f in faces]
