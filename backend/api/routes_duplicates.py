import os
import threading

from fastapi import APIRouter, Depends, Request, HTTPException

from api.deps import get_app_db
from services.jobs import tracker

router = APIRouter(prefix="/api")

PHOTOS_BASE = os.getenv("PHOTOS_DIR", "/photos")


@router.post("/duplicates/scan")
def scan_duplicates(request: Request, db=Depends(get_app_db)):
    hasher = request.app.state.hasher

    rows = db.execute("SELECT id, filepath, phash FROM photos").fetchall()
    unhashed = [r for r in rows if not r["phash"]]

    job_id = tracker.create("duplicates_scan")
    total = len(unhashed) + 1  # +1 for comparison step
    tracker.update(job_id, 0, total)

    def _scan():
        from services.database import get_db
        conn = get_db()

        # Compute missing hashes
        for i, row in enumerate(unhashed):
            abs_path = os.path.join(PHOTOS_BASE, row["filepath"])
            phash = hasher.compute_phash(abs_path)
            if phash:
                conn.execute("UPDATE photos SET phash = ? WHERE id = ?", (phash, row["id"]))
            if (i + 1) % 50 == 0:
                conn.commit()
            tracker.update(job_id, i + 1, total)
        conn.commit()

        # Clear old groups
        conn.execute("DELETE FROM duplicate_members")
        conn.execute("DELETE FROM duplicate_groups")
        conn.commit()

        # Find duplicates
        all_hashes = conn.execute("SELECT id, phash FROM photos WHERE phash IS NOT NULL").fetchall()
        hash_pairs = [(r["id"], r["phash"]) for r in all_hashes]
        groups = hasher.find_duplicates(hash_pairs)

        for group in groups:
            conn.execute("INSERT INTO duplicate_groups DEFAULT VALUES")
            group_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            for photo_id in group:
                conn.execute(
                    "INSERT INTO duplicate_members (group_id, photo_id) VALUES (?, ?)",
                    (group_id, photo_id),
                )
        conn.commit()

        tracker.update(job_id, total, total)
        tracker.complete(job_id, {"groups_found": len(groups)})
        conn.close()

    threading.Thread(target=_scan, daemon=True).start()
    return {"job_id": job_id}


@router.get("/duplicates")
def list_duplicates(db=Depends(get_app_db)):
    groups = db.execute("SELECT id FROM duplicate_groups ORDER BY id").fetchall()
    result = []
    for g in groups:
        photos = db.execute(
            """SELECT p.*, dm.is_kept
               FROM photos p
               JOIN duplicate_members dm ON dm.photo_id = p.id
               WHERE dm.group_id = ?
               ORDER BY p.filesize DESC""",
            (g["id"],),
        ).fetchall()
        result.append({
            "group_id": g["id"],
            "photos": [dict(p) for p in photos],
        })
    return result


@router.post("/duplicates/{group_id}/keep/{photo_id}")
def keep_photo(group_id: int, photo_id: int, db=Depends(get_app_db)):
    db.execute("UPDATE duplicate_members SET is_kept = 0 WHERE group_id = ?", (group_id,))
    db.execute(
        "UPDATE duplicate_members SET is_kept = 1 WHERE group_id = ? AND photo_id = ?",
        (group_id, photo_id),
    )
    db.commit()
    return {"group_id": group_id, "kept": photo_id}
