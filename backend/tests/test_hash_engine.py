from pathlib import Path

from services.hash_engine import HashEngine


def test_compute_phash_returns_string(sample_jpeg: Path) -> None:
    h = HashEngine.compute_phash(str(sample_jpeg))
    assert isinstance(h, str)
    assert len(h) == 16  # phash is 64 bits = 16 hex chars


def test_compute_phash_deterministic(sample_jpeg: Path) -> None:
    assert HashEngine.compute_phash(str(sample_jpeg)) == HashEngine.compute_phash(str(sample_jpeg))


def test_compute_phash_returns_none_for_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "not_an_image.jpg"
    bad.write_text("not an image")
    assert HashEngine.compute_phash(str(bad)) is None


def test_compute_phash_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert HashEngine.compute_phash(str(tmp_path / "missing.jpg")) is None


def test_find_duplicates_groups_similar(sample_jpeg: Path, sample_jpeg_variant: Path, sample_jpeg_different: Path) -> None:
    h1 = HashEngine.compute_phash(str(sample_jpeg))
    h2 = HashEngine.compute_phash(str(sample_jpeg_variant))
    h3 = HashEngine.compute_phash(str(sample_jpeg_different))

    groups = HashEngine.find_duplicates([(1, h1), (2, h2), (3, h3)], threshold=8)

    assert len(groups) == 1
    group = groups[0]
    assert 1 in group and 2 in group
    assert 3 not in group


def test_find_duplicates_empty_when_all_distinct(sample_jpeg: Path, sample_jpeg_different: Path) -> None:
    h1 = HashEngine.compute_phash(str(sample_jpeg))
    h2 = HashEngine.compute_phash(str(sample_jpeg_different))

    groups = HashEngine.find_duplicates([(1, h1), (2, h2)], threshold=0)
    assert groups == []


def test_find_duplicates_skips_none_hashes() -> None:
    groups = HashEngine.find_duplicates([(1, None), (2, None)], threshold=8)
    assert groups == []
