import os
import sys

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

app = typer.Typer(name="photosort", help="Local AI photo classifier")
console = Console()

PHOTOS_DIR = os.getenv("PHOTOS_DIR", "/photos")
CONFIG_DIR = os.getenv("CONFIG_DIR", "config")
DEVICE = os.getenv("DEVICE", "cuda")


def _init():
    from services.database import init_db
    init_db()


@app.command()
def scan():
    """Scan photo directory for new files."""
    import numpy as np
    _init()
    from services.database import get_db
    from services.scanner import get_new_photos, extract_exif
    from services.thumbnails import ensure_thumbnail

    db = get_db()
    new_files = get_new_photos(db)
    if not new_files:
        console.print("[green]No new photos found.[/green]")
        return

    console.print(f"Found {len(new_files)} new files")
    with Progress() as progress:
        task = progress.add_task("Importing...", total=len(new_files))
        for f in new_files:
            abs_path = os.path.join(PHOTOS_DIR, f["filepath"])
            exif = extract_exif(abs_path) if not f["is_video"] else {"width": None, "height": None, "taken_at": None}
            try:
                db.execute(
                    "INSERT INTO photos (filepath, filename, filesize, width, height, taken_at) VALUES (?,?,?,?,?,?)",
                    (f["filepath"], f["filename"], f["filesize"], exif["width"], exif["height"], exif["taken_at"]),
                )
                db.commit()
                row = db.execute("SELECT id FROM photos WHERE filepath = ?", (f["filepath"],)).fetchone()
                if row and not f["is_video"]:
                    ensure_thumbnail(row[0], abs_path)
            except Exception:
                pass
            progress.advance(task)
    console.print(f"[green]Imported {len(new_files)} files.[/green]")
    db.close()


@app.command()
def classify(force: bool = typer.Option(False, "--force", help="Reclassify all photos")):
    """Classify photos using CLIP."""
    import numpy as np
    _init()
    from services.database import get_db
    from services.clip_engine import CLIPEngine

    db = get_db()
    clip = CLIPEngine(device=DEVICE)
    clip.load_categories(os.path.join(CONFIG_DIR, "categories.yml"))

    if force:
        rows = db.execute("SELECT id, filepath, clip_embedding FROM photos").fetchall()
    else:
        rows = db.execute(
            "SELECT id, filepath, clip_embedding FROM photos WHERE id NOT IN (SELECT photo_id FROM classifications)"
        ).fetchall()

    if not rows:
        console.print("[green]No photos to classify.[/green]")
        return

    console.print(f"Classifying {len(rows)} photos...")
    with Progress() as progress:
        task = progress.add_task("Classifying...", total=len(rows))
        for row in rows:
            blob = row["clip_embedding"]
            if blob:
                emb = np.frombuffer(blob, dtype=np.float32)
            else:
                abs_path = os.path.join(PHOTOS_DIR, row["filepath"])
                emb = clip.encode_image(abs_path)
                if emb is None:
                    progress.advance(task)
                    continue
                db.execute("UPDATE photos SET clip_embedding = ? WHERE id = ?", (emb.tobytes(), row["id"]))

            cat, conf = clip.classify(emb)
            if force:
                db.execute("DELETE FROM classifications WHERE photo_id = ? AND is_manual = 0", (row["id"],))
            db.execute(
                "INSERT OR REPLACE INTO classifications (photo_id, category, confidence) VALUES (?,?,?)",
                (row["id"], cat, conf),
            )
            progress.advance(task)
        db.commit()
    console.print("[green]Classification complete.[/green]")
    db.close()


@app.command()
def search(query: str = typer.Argument(..., help="Text to search for")):
    """Search photos by text description."""
    import numpy as np
    _init()
    from services.database import get_db
    from services.clip_engine import CLIPEngine

    db = get_db()
    clip = CLIPEngine(device=DEVICE)
    clip.load_categories(os.path.join(CONFIG_DIR, "categories.yml"))

    rows = db.execute("SELECT id, filepath, clip_embedding FROM photos WHERE clip_embedding IS NOT NULL").fetchall()
    if not rows:
        console.print("[yellow]No embedded photos. Run classify first.[/yellow]")
        return

    ids = [r["id"] for r in rows]
    paths = {r["id"]: r["filepath"] for r in rows}
    embeddings = np.stack([np.frombuffer(r["clip_embedding"], dtype=np.float32) for r in rows])

    results = clip.search(query, embeddings, top_k=10)
    table = Table(title=f'Search: "{query}"')
    table.add_column("Rank")
    table.add_column("Score")
    table.add_column("File")

    for rank, (idx, score) in enumerate(results, 1):
        pid = ids[idx]
        table.add_row(str(rank), f"{score:.3f}", paths[pid])
    console.print(table)
    db.close()


@app.command()
def faces(
    action: str = typer.Argument("detect", help="detect or cluster"),
):
    """Detect or cluster faces."""
    import numpy as np
    _init()
    from services.database import get_db
    from services.face_engine import FaceEngine

    db = get_db()
    engine = FaceEngine(device=DEVICE)

    if action == "detect":
        rows = db.execute(
            "SELECT id, filepath FROM photos WHERE id NOT IN (SELECT DISTINCT photo_id FROM faces) AND filepath NOT LIKE '%.mov' AND filepath NOT LIKE '%.mp4'"
        ).fetchall()
        if not rows:
            console.print("[green]All photos processed.[/green]")
            return
        console.print(f"Detecting faces in {len(rows)} photos...")
        with Progress() as progress:
            task = progress.add_task("Detecting...", total=len(rows))
            count = 0
            for row in rows:
                abs_path = os.path.join(PHOTOS_DIR, row["filepath"])
                try:
                    detected = engine.detect_faces(abs_path)
                    for f in detected:
                        db.execute(
                            "INSERT INTO faces (photo_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding) VALUES (?,?,?,?,?,?)",
                            (row["id"], f["bbox_x"], f["bbox_y"], f["bbox_w"], f["bbox_h"],
                             f["embedding"].astype(np.float32).tobytes()),
                        )
                        count += 1
                    db.commit()
                except Exception:
                    pass
                progress.advance(task)
        console.print(f"[green]Detected {count} faces.[/green]")

    elif action == "cluster":
        rows = db.execute("SELECT id, embedding FROM faces WHERE person_id IS NULL").fetchall()
        if not rows:
            console.print("[green]No unclustered faces.[/green]")
            return
        face_ids = [r["id"] for r in rows]
        embeddings = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        labels = engine.cluster_faces(embeddings)

        label_to_person = {}
        for fid, label in zip(face_ids, labels):
            if label == -1:
                continue
            if label not in label_to_person:
                db.execute("INSERT INTO persons (name) VALUES (NULL)")
                db.commit()
                pid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                label_to_person[label] = pid
            db.execute("UPDATE faces SET person_id = ? WHERE id = ?", (label_to_person[label], fid))
        db.commit()
        console.print(f"[green]Created {len(label_to_person)} person clusters.[/green]")

    db.close()


@app.command()
def duplicates():
    """Find duplicate photos."""
    _init()
    from services.database import get_db
    from services.hash_engine import HashEngine

    db = get_db()
    hasher = HashEngine()

    # Compute missing hashes
    rows = db.execute("SELECT id, filepath FROM photos WHERE phash IS NULL").fetchall()
    if rows:
        console.print(f"Computing hashes for {len(rows)} photos...")
        with Progress() as progress:
            task = progress.add_task("Hashing...", total=len(rows))
            for row in rows:
                abs_path = os.path.join(PHOTOS_DIR, row["filepath"])
                h = hasher.compute_phash(abs_path)
                if h:
                    db.execute("UPDATE photos SET phash = ? WHERE id = ?", (h, row["id"]))
                progress.advance(task)
            db.commit()

    # Find duplicates
    all_h = db.execute("SELECT id, phash FROM photos WHERE phash IS NOT NULL").fetchall()
    groups = hasher.find_duplicates([(r["id"], r["phash"]) for r in all_h])

    db.execute("DELETE FROM duplicate_members")
    db.execute("DELETE FROM duplicate_groups")
    for group in groups:
        db.execute("INSERT INTO duplicate_groups DEFAULT VALUES")
        gid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        for pid in group:
            db.execute("INSERT INTO duplicate_members (group_id, photo_id) VALUES (?,?)", (gid, pid))
    db.commit()

    console.print(f"[green]Found {len(groups)} duplicate groups.[/green]")
    db.close()


@app.command()
def stats():
    """Show library statistics."""
    _init()
    from services.database import get_db
    db = get_db()

    table = Table(title="PhotoSort Stats")
    table.add_column("Metric")
    table.add_column("Value")

    total = db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    classified = db.execute("SELECT COUNT(DISTINCT photo_id) FROM classifications").fetchone()[0]
    faces_count = db.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    persons = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    dups = db.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0]

    table.add_row("Total Photos", str(total))
    table.add_row("Classified", str(classified))
    table.add_row("Uncategorized", str(total - classified))
    table.add_row("Faces Detected", str(faces_count))
    table.add_row("Persons", str(persons))
    table.add_row("Duplicate Groups", str(dups))

    console.print(table)
    db.close()


if __name__ == "__main__":
    app()
