import sqlite3
from pathlib import Path

from PIL import Image

from services.scanner import walk_photos, get_new_photos, extract_exif


def _make_image(path: Path, size=(32, 32), color=(200, 100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, "JPEG")


def test_walk_photos_finds_supported_extensions(tmp_photos_dir: Path) -> None:
    _make_image(tmp_photos_dir / "a.jpg")
    _make_image(tmp_photos_dir / "sub" / "b.jpeg")
    (tmp_photos_dir / "c.png").write_bytes(b"not really a png but has extension")
    (tmp_photos_dir / "video.mp4").write_bytes(b"fake video")
    (tmp_photos_dir / "README.txt").write_text("ignore me")

    found = walk_photos(str(tmp_photos_dir))
    paths = {f["filepath"] for f in found}

    assert "a.jpg" in paths
    assert "sub/b.jpeg" in paths or "sub\\b.jpeg" in paths
    assert "c.png" in paths
    assert "video.mp4" in paths
    assert "README.txt" not in paths


def test_walk_photos_flags_videos(tmp_photos_dir: Path) -> None:
    _make_image(tmp_photos_dir / "photo.jpg")
    (tmp_photos_dir / "clip.mov").write_bytes(b"fake")

    found = {f["filename"]: f for f in walk_photos(str(tmp_photos_dir))}
    assert found["photo.jpg"]["is_video"] is False
    assert found["clip.mov"]["is_video"] is True


def test_walk_photos_reports_filesize(tmp_photos_dir: Path) -> None:
    _make_image(tmp_photos_dir / "a.jpg")
    found = walk_photos(str(tmp_photos_dir))
    assert found[0]["filesize"] > 0


def test_get_new_photos_excludes_already_imported(tmp_photos_dir: Path, tmp_path: Path) -> None:
    _make_image(tmp_photos_dir / "one.jpg")
    _make_image(tmp_photos_dir / "two.jpg")

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, filepath TEXT)")
    db.execute("INSERT INTO photos (filepath) VALUES (?)", ("one.jpg",))
    db.commit()

    new = get_new_photos(db, base_dir=str(tmp_photos_dir))
    names = {n["filename"] for n in new}
    assert names == {"two.jpg"}
    db.close()


def test_extract_exif_on_plain_jpeg_returns_dimensions(tmp_photos_dir: Path) -> None:
    path = tmp_photos_dir / "p.jpg"
    _make_image(path, size=(100, 50))
    info = extract_exif(str(path))
    assert info["width"] == 100
    assert info["height"] == 50


def test_extract_exif_handles_missing_file(tmp_path: Path) -> None:
    info = extract_exif(str(tmp_path / "missing.jpg"))
    assert info == {"width": None, "height": None, "taken_at": None}
