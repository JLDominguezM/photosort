import os
import sqlite3
import threading

DB_PATH = os.path.join(os.getenv("DATA_DIR", "data"), "photosort.db")
_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_SCHEMA = """
    CREATE TABLE IF NOT EXISTS photos (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        filepath    TEXT NOT NULL UNIQUE,
        filename    TEXT NOT NULL,
        filesize    INTEGER NOT NULL,
        width       INTEGER,
        height      INTEGER,
        taken_at    TEXT,
        imported_at TEXT NOT NULL DEFAULT (datetime('now')),
        phash       TEXT,
        clip_embedding BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_photos_filepath ON photos(filepath);
    CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
    CREATE INDEX IF NOT EXISTS idx_photos_taken_at ON photos(taken_at);

    CREATE TABLE IF NOT EXISTS classifications (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
        category    TEXT NOT NULL,
        confidence  REAL NOT NULL,
        is_manual   INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(photo_id, category)
    );
    CREATE INDEX IF NOT EXISTS idx_class_photo ON classifications(photo_id);
    CREATE INDEX IF NOT EXISTS idx_class_category ON classifications(category);

    CREATE TABLE IF NOT EXISTS persons (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT,
        centroid    BLOB,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS faces (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
        bbox_x      INTEGER NOT NULL,
        bbox_y      INTEGER NOT NULL,
        bbox_w      INTEGER NOT NULL,
        bbox_h      INTEGER NOT NULL,
        embedding   BLOB NOT NULL,
        person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL
    );
    CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
    CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);

    CREATE TABLE IF NOT EXISTS duplicate_groups (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS duplicate_members (
        group_id    INTEGER NOT NULL REFERENCES duplicate_groups(id) ON DELETE CASCADE,
        photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
        is_kept     INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (group_id, photo_id)
    );
    CREATE INDEX IF NOT EXISTS idx_dup_members_kept ON duplicate_members(group_id, is_kept);
"""


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in cols)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    # Add persons.centroid to pre-existing databases that predate it.
    if not _column_exists(conn, "persons", "centroid"):
        conn.execute("ALTER TABLE persons ADD COLUMN centroid BLOB")


def init_db() -> None:
    with _lock:
        conn = get_db()
        conn.executescript(_SCHEMA)
        _apply_migrations(conn)
        conn.commit()
        conn.close()
