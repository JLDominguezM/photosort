def test_init_db_creates_expected_tables(isolated_db) -> None:
    conn = isolated_db.get_db()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    conn.close()

    expected = {
        "photos",
        "classifications",
        "persons",
        "faces",
        "duplicate_groups",
        "duplicate_members",
    }
    assert expected.issubset(names)


def test_photos_has_unique_filepath(isolated_db) -> None:
    import sqlite3

    conn = isolated_db.get_db()
    conn.execute(
        "INSERT INTO photos (filepath, filename, filesize) VALUES ('a.jpg', 'a.jpg', 100)"
    )
    conn.commit()
    try:
        conn.execute(
            "INSERT INTO photos (filepath, filename, filesize) VALUES ('a.jpg', 'a.jpg', 200)"
        )
        conn.commit()
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    finally:
        conn.close()
    assert raised, "Duplicate filepath should violate UNIQUE constraint"


def test_classifications_cascade_on_photo_delete(isolated_db) -> None:
    conn = isolated_db.get_db()
    conn.execute(
        "INSERT INTO photos (filepath, filename, filesize) VALUES ('b.jpg', 'b.jpg', 10)"
    )
    pid = conn.execute("SELECT id FROM photos WHERE filepath='b.jpg'").fetchone()["id"]
    conn.execute(
        "INSERT INTO classifications (photo_id, category, confidence) VALUES (?, 'Food', 0.9)",
        (pid,),
    )
    conn.commit()

    conn.execute("DELETE FROM photos WHERE id = ?", (pid,))
    conn.commit()

    remaining = conn.execute(
        "SELECT COUNT(*) AS c FROM classifications WHERE photo_id = ?", (pid,)
    ).fetchone()["c"]
    conn.close()
    assert remaining == 0


def test_wal_mode_enabled(isolated_db) -> None:
    conn = isolated_db.get_db()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"
