import os
from pathlib import Path

from services.paths import safe_photo_path


def test_normal_relative_path(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").touch()
    assert safe_photo_path(str(tmp_path), "a.jpg") == str(tmp_path / "a.jpg")


def test_nested_relative_path(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)
    (nested / "x.jpg").touch()
    assert safe_photo_path(str(tmp_path), "sub/dir/x.jpg") == str(nested / "x.jpg")


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    assert safe_photo_path(str(tmp_path), "../escaped.jpg") is None


def test_rejects_deep_parent_traversal(tmp_path: Path) -> None:
    assert safe_photo_path(str(tmp_path), "a/../../escape.jpg") is None


def test_rejects_absolute_path(tmp_path: Path) -> None:
    assert safe_photo_path(str(tmp_path), "/etc/passwd") is None


def test_allows_internal_traversal_that_stays_inside(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b.jpg").touch()
    assert safe_photo_path(str(tmp_path), "a/../b.jpg") == str(tmp_path / "b.jpg")


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir(exist_ok=True)
    target = outside / "secret.jpg"
    target.write_text("secret")

    base = tmp_path / "photos"
    base.mkdir()
    link = base / "link.jpg"
    os.symlink(target, link)

    assert safe_photo_path(str(base), "link.jpg") is None


def test_base_itself_is_accepted_but_rel_must_be_inside(tmp_path: Path) -> None:
    # empty rel resolves to base itself, which is inside — accepted
    assert safe_photo_path(str(tmp_path), "") == str(tmp_path)
