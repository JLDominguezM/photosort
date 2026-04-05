import os
import threading

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import FileResponse

from api.deps import get_app_db
from models.schemas import PhotoOut, PhotoList, StatsOut
from services.scanner import get_new_photos, extract_exif, PHOTOS_DIR
from services.thumbnails import ensure_thumbnail, get_thumbnail_path
from services.jobs import tracker

router = APIRouter(prefix="/api")

PHOTOS_BASE = os.getenv("PHOTOS_DIR", "/photos")


@router.post("/scan")
def scan_photos(db=Depends(get_app_db)):
    new_files = get_new_photos(db)
    if not new_files:
        return {"message": "No new photos found", "count": 0}

    job_id = tracker.create("scan")
    total = len(new_files)
    tracker.update(job_id, 0, total)

    def _scan():
        conn_inner = __import__("services.database", fromlist=["get_db"]).get_db()
        count = 0
        for i, f in enumerate(new_files):
            abs_path = os.path.join(PHOTOS_BASE, f["filepath"])
            exif = extract_exif(abs_path) if not f["is_video"] else {"width": None, "height": None, "taken_at": None}
            try:
                conn_inner.execute(
                    """INSERT INTO photos (filepath, filename, filesize, width, height, taken_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (f["filepath"], f["filename"], f["filesize"],
                     exif["width"], exif["height"], exif["taken_at"]),
                )
                conn_inner.commit()
                row = conn_inner.execute(
                    "SELECT id FROM photos WHERE filepath = ?", (f["filepath"],)
                ).fetchone()
                if row and not f["is_video"]:
                    ensure_thumbnail(row[0], abs_path)
                count += 1
            except Exception:
                pass
            tracker.update(job_id, i + 1, total)
        tracker.complete(job_id, {"imported": count})
        conn_inner.close()

    threading.Thread(target=_scan, daemon=True).start()
    return {"job_id": job_id, "total": total}


@router.get("/photos", response_model=PhotoList)
def list_photos(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    person_id: int | None = None,
    db=Depends(get_app_db),
):
    where = []
    params = []

    if category == "Uncategorized":
        where.append("p.id NOT IN (SELECT photo_id FROM classifications)")
    elif category:
        where.append("""p.id IN (
            SELECT photo_id FROM classifications
            WHERE category = ? AND (is_manual = 1 OR confidence = (
                SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = classifications.photo_id
            ))
        )""")
        params.append(category)
    if date_from:
        where.append("p.taken_at >= ?")
        params.append(date_from)
    if date_to:
        where.append("p.taken_at <= ?")
        params.append(date_to)
    if person_id:
        where.append("p.id IN (SELECT photo_id FROM faces WHERE person_id = ?)")
        params.append(person_id)

    where_clause = (" WHERE " + " AND ".join(where)) if where else ""

    total = db.execute(f"SELECT COUNT(*) FROM photos p{where_clause}", params).fetchone()[0]

    offset = (page - 1) * per_page
    rows = db.execute(
        f"""SELECT p.*, c.category, c.confidence
            FROM photos p
            LEFT JOIN classifications c ON c.photo_id = p.id
                AND c.confidence = (SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = p.id)
            {where_clause}
            ORDER BY p.taken_at DESC NULLS LAST, p.id DESC
            LIMIT ? OFFSET ?""",
        params + [per_page, offset],
    ).fetchall()

    photos = [PhotoOut(**dict(r)) for r in rows]
    return PhotoList(photos=photos, total=total, page=page, per_page=per_page)


@router.get("/photos/{photo_id}")
def get_photo(photo_id: int, db=Depends(get_app_db)):
    row = db.execute(
        """SELECT p.*, c.category, c.confidence
           FROM photos p
           LEFT JOIN classifications c ON c.photo_id = p.id
               AND c.confidence = (SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = p.id)
           WHERE p.id = ?""",
        (photo_id,),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Photo not found")
    return PhotoOut(**dict(row))


@router.get("/photos/{photo_id}/thumbnail")
def get_thumbnail(photo_id: int, db=Depends(get_app_db)):
    row = db.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    thumb = get_thumbnail_path(photo_id)
    if not thumb:
        abs_path = os.path.join(PHOTOS_BASE, row["filepath"])
        thumb = ensure_thumbnail(photo_id, abs_path)
    if not thumb:
        raise HTTPException(404, "Could not generate thumbnail")
    return FileResponse(thumb, media_type="image/jpeg")


@router.get("/photos/{photo_id}/full")
def get_full_photo(photo_id: int, db=Depends(get_app_db)):
    row = db.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row:
        raise HTTPException(404)
    abs_path = os.path.join(PHOTOS_BASE, row["filepath"])
    if not os.path.exists(abs_path):
        raise HTTPException(404)
    return FileResponse(abs_path)


@router.get("/stats", response_model=StatsOut)
def get_stats(db=Depends(get_app_db)):
    total = db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    classified = db.execute("SELECT COUNT(DISTINCT photo_id) FROM classifications").fetchone()[0]
    faces_detected = db.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    persons = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    dup_groups = db.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0]
    return StatsOut(
        total_photos=total,
        classified=classified,
        uncategorized=total - classified,
        faces_detected=faces_detected,
        persons=persons,
        duplicate_groups=dup_groups,
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = tracker.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **job}


@router.get("/jobs")
def list_jobs():
    return tracker.list_all()
