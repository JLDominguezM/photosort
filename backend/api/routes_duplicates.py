import os
import threading

from fastapi import APIRouter, Depends, Query, Request

from api.deps import get_app_db
from services.jobs import tracker
from services.logging_config import get_logger
from services.paths import safe_photo_path

router = APIRouter(prefix="/api", tags=["Duplicates"])

log = get_logger(__name__)

PHOTOS_BASE = os.path.realpath(os.getenv("PHOTOS_DIR", "/photos"))
THUMB_DIR = os.path.join(os.getenv("DATA_DIR", "data"), "thumbnails")


@router.post("/duplicates/scan", summary="Scan library for duplicate photos")
def scan_duplicates(request: Request, db=Depends(get_app_db)):
    """Compute perceptual hashes for any photo missing one, then group
    near-identical photos by Hamming distance. Replaces any existing groups.
    Runs asynchronously; poll /api/jobs/{job_id} for progress."""
    hasher = request.app.state.hasher

    rows = db.execute("SELECT id, filepath, phash FROM photos").fetchall()
    unhashed = [r for r in rows if not r["phash"]]

    job_id = tracker.create("duplicates_scan")
    total = len(unhashed) + 1
    tracker.update(job_id, 0, total)
    log.info("duplicates_scan job %s started: unhashed=%d", job_id, len(unhashed))

    def _scan():
        from services.database import get_db
        conn = get_db()

        for i, row in enumerate(unhashed):
            abs_path = safe_photo_path(PHOTOS_BASE, row["filepath"])
            if abs_path is None:
                tracker.update(job_id, i + 1, total)
                continue
            phash = hasher.compute_phash(abs_path)
            if phash:
                conn.execute("UPDATE photos SET phash = ? WHERE id = ?", (phash, row["id"]))
            if (i + 1) % 50 == 0:
                conn.commit()
            tracker.update(job_id, i + 1, total)
        conn.commit()

        conn.execute("DELETE FROM duplicate_members")
        conn.execute("DELETE FROM duplicate_groups")
        conn.commit()

        all_hashes = conn.execute("SELECT id, phash FROM photos WHERE phash IS NOT NULL").fetchall()
        groups = hasher.find_duplicates([(r["id"], r["phash"]) for r in all_hashes])

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
        log.info("duplicates_scan job %s finished: groups=%d", job_id, len(groups))
        conn.close()

    threading.Thread(target=_scan, daemon=True).start()
    return {"job_id": job_id}


@router.get("/duplicates", summary="List duplicate groups with their photos")
def list_duplicates(db=Depends(get_app_db)):
    """Return every duplicate group with member photos sorted by filesize
    (largest first so the biggest/best quality is easy to spot)."""
    groups = db.execute("SELECT id FROM duplicate_groups ORDER BY id").fetchall()
    result = []
    for g in groups:
        # Explicit columns: SELECT p.* would pull clip_embedding (BLOB) which
        # FastAPI's encoder then tries to UTF-8 decode and 500s on.
        photos = db.execute(
            """SELECT p.id, p.filepath, p.filename, p.filesize,
                      p.width, p.height, p.taken_at, p.imported_at,
                      dm.is_kept
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


@router.post("/duplicates/{group_id}/keep/{photo_id}", summary="Mark a photo as the one to keep in its group")
def keep_photo(group_id: int, photo_id: int, db=Depends(get_app_db)):
    """Exactly one photo per group can be marked as kept. This clears any
    previous keep in the group before setting the new one."""
    db.execute("UPDATE duplicate_members SET is_kept = 0 WHERE group_id = ?", (group_id,))
    db.execute(
        "UPDATE duplicate_members SET is_kept = 1 WHERE group_id = ? AND photo_id = ?",
        (group_id, photo_id),
    )
    db.commit()
    return {"group_id": group_id, "kept": photo_id}


@router.post("/duplicates/cleanup", summary="Delete DB entries for non-kept duplicates")
def cleanup_duplicates(
    dry_run: bool = Query(True, description="If true, only report what would be deleted"),
    db=Depends(get_app_db),
):
    """For each duplicate group that has exactly one photo marked as kept,
    delete the database rows (and thumbnails) of the other members. Groups
    with no keep mark are skipped.

    Original files on disk are **never** touched — PHOTOS_DIR is mounted
    read-only. Re-running a scan will reindex any photos whose files still
    exist, so this is a safe way to clean the library view without data loss.
    """
    groups = db.execute("SELECT id FROM duplicate_groups").fetchall()

    to_delete: list[int] = []
    skipped_groups = 0
    processed_groups = 0

    for g in groups:
        members = db.execute(
            "SELECT photo_id, is_kept FROM duplicate_members WHERE group_id = ?",
            (g["id"],),
        ).fetchall()
        kept = [m["photo_id"] for m in members if m["is_kept"]]
        if len(kept) != 1:
            skipped_groups += 1
            continue
        processed_groups += 1
        for m in members:
            if not m["is_kept"]:
                to_delete.append(m["photo_id"])

    if dry_run:
        log.info(
            "cleanup dry_run: would delete %d photos across %d groups (%d skipped)",
            len(to_delete), processed_groups, skipped_groups,
        )
        return {
            "dry_run": True,
            "would_delete": len(to_delete),
            "processed_groups": processed_groups,
            "skipped_groups": skipped_groups,
        }

    for pid in to_delete:
        thumb = os.path.join(THUMB_DIR, f"{pid}.jpg")
        try:
            if os.path.exists(thumb):
                os.remove(thumb)
        except OSError as e:
            log.warning("Could not remove thumbnail %s: %s", thumb, e)
        db.execute("DELETE FROM photos WHERE id = ?", (pid,))

    db.execute(
        """DELETE FROM duplicate_groups WHERE id IN (
               SELECT dg.id FROM duplicate_groups dg
               LEFT JOIN duplicate_members dm ON dm.group_id = dg.id
               GROUP BY dg.id
               HAVING COUNT(dm.photo_id) <= 1
           )"""
    )
    db.commit()
    log.info(
        "cleanup: deleted %d photos across %d groups (%d skipped)",
        len(to_delete), processed_groups, skipped_groups,
    )
    return {
        "dry_run": False,
        "deleted": len(to_delete),
        "processed_groups": processed_groups,
        "skipped_groups": skipped_groups,
    }
