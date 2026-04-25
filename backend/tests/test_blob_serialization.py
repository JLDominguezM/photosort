"""Regression: SELECT p.* on the photos table pulls clip_embedding (BLOB)
which crashes FastAPI's default encoder when the row is dict-ified and
returned. The fix is to enumerate columns explicitly. These tests scan
the route source files to make sure no future change reintroduces
SELECT p.* in an endpoint that returns dicts.

Comment lines are skipped so that documentation about this very bug
(which legitimately mentions `SELECT p.*`) doesn't trigger a false
positive."""

import re
from pathlib import Path

ROUTES_DIR = Path(__file__).resolve().parents[1] / "api"

# Match `p.*` or `photos.*` only when preceded by SELECT or a comma —
# i.e. actual SQL column lists, not just text containing those characters.
_BAD_PATTERN = re.compile(r"(SELECT|,)\s+p(hotos)?\.\*", re.IGNORECASE)


def _non_comment_lines(src: str) -> list[str]:
    return [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]


def _offending_lines(path: Path) -> list[str]:
    src = path.read_text()
    return [ln for ln in _non_comment_lines(src) if _BAD_PATTERN.search(ln)]


def test_routes_duplicates_does_not_select_star_from_photos() -> None:
    offending = _offending_lines(ROUTES_DIR / "routes_duplicates.py")
    assert not offending, (
        "routes_duplicates must not SELECT p.* — it pulls clip_embedding BLOB "
        f"which breaks JSON serialization. Offending lines: {offending}"
    )


def test_routes_faces_does_not_select_star_from_photos() -> None:
    offending = _offending_lines(ROUTES_DIR / "routes_faces.py")
    assert not offending, (
        "routes_faces must not SELECT p.* — it pulls clip_embedding BLOB "
        f"which breaks JSON serialization. Offending lines: {offending}"
    )
