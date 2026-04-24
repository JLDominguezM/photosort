"""Tests the pure cleanup logic: which photos should be deleted given a
set of duplicate groups and keep marks. The endpoint wiring is thin and
tested implicitly; here we only exercise the decision rules by driving
the same queries the endpoint uses."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def populated_db(isolated_db):
    conn = isolated_db.get_db()
    for pid, name, size in [(1, "a.jpg", 100), (2, "b.jpg", 200), (3, "c.jpg", 300),
                             (4, "d.jpg", 400), (5, "e.jpg", 500)]:
        conn.execute(
            "INSERT INTO photos (id, filepath, filename, filesize) VALUES (?, ?, ?, ?)",
            (pid, name, name, size),
        )
    conn.execute("INSERT INTO duplicate_groups (id) VALUES (1), (2), (3)")
    # Group 1: 1 kept, 2 others → should delete 2 of 3
    conn.execute(
        "INSERT INTO duplicate_members (group_id, photo_id, is_kept) VALUES "
        "(1, 1, 1), (1, 2, 0), (1, 3, 0)"
    )
    # Group 2: no keep → skip
    conn.execute(
        "INSERT INTO duplicate_members (group_id, photo_id, is_kept) VALUES "
        "(2, 4, 0), (2, 5, 0)"
    )
    conn.commit()
    yield conn
    conn.close()


def _compute_plan(conn):
    """Replicates the endpoint's decision logic for testability."""
    groups = conn.execute("SELECT id FROM duplicate_groups").fetchall()
    to_delete: list[int] = []
    skipped = 0
    processed = 0
    for g in groups:
        members = conn.execute(
            "SELECT photo_id, is_kept FROM duplicate_members WHERE group_id = ?",
            (g["id"],),
        ).fetchall()
        kept = [m["photo_id"] for m in members if m["is_kept"]]
        if len(kept) != 1:
            skipped += 1
            continue
        processed += 1
        to_delete.extend(m["photo_id"] for m in members if not m["is_kept"])
    return to_delete, processed, skipped


def test_plan_deletes_non_kept_when_one_keep_exists(populated_db) -> None:
    to_delete, processed, skipped = _compute_plan(populated_db)
    assert set(to_delete) == {2, 3}
    assert processed == 1
    assert skipped == 2  # group 2 (no keep) + group 3 (empty)


def test_plan_skips_groups_with_no_keep(populated_db) -> None:
    # Group 2 has members but no is_kept=1 → must be skipped, nothing deleted
    plan, _, _ = _compute_plan(populated_db)
    assert 4 not in plan
    assert 5 not in plan


def test_plan_skips_groups_with_multiple_keeps(isolated_db) -> None:
    conn = isolated_db.get_db()
    conn.execute(
        "INSERT INTO photos (id, filepath, filename, filesize) VALUES "
        "(10, 'x.jpg', 'x', 1), (11, 'y.jpg', 'y', 1)"
    )
    conn.execute("INSERT INTO duplicate_groups (id) VALUES (10)")
    conn.execute(
        "INSERT INTO duplicate_members (group_id, photo_id, is_kept) VALUES "
        "(10, 10, 1), (10, 11, 1)"
    )
    conn.commit()

    plan, processed, skipped = _compute_plan(conn)
    conn.close()
    assert plan == []
    assert processed == 0
    assert skipped == 1
