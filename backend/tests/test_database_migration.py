import os
import sqlite3
from pathlib import Path

import pytest


def _make_db_without_centroid(path: str) -> None:
    """Simulate a pre-migration DB that has the old persons schema."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO persons (name) VALUES ('Alice'), ('Bob');
        """
    )
    conn.commit()
    conn.close()


def test_migration_adds_centroid_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "photosort.db"

    _make_db_without_centroid(str(db_path))
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    import importlib
    import services.database as database_module
    importlib.reload(database_module)
    database_module.init_db()

    conn = database_module.get_db()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(persons)").fetchall()]
    data = conn.execute("SELECT name FROM persons ORDER BY id").fetchall()
    conn.close()

    assert "centroid" in cols
    assert [r["name"] for r in data] == ["Alice", "Bob"]


def test_migration_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    import importlib
    import services.database as database_module
    importlib.reload(database_module)

    database_module.init_db()
    database_module.init_db()  # second call must not raise

    conn = database_module.get_db()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(persons)").fetchall()]
    conn.close()
    assert cols.count("centroid") == 1
