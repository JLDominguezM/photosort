import os
import sys
from pathlib import Path

import pytest
from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def tmp_photos_dir(tmp_path: Path) -> Path:
    d = tmp_path / "photos"
    d.mkdir()
    return d


@pytest.fixture
def sample_jpeg(tmp_photos_dir: Path) -> Path:
    path = tmp_photos_dir / "sample.jpg"
    Image.new("RGB", (64, 64), color=(128, 64, 32)).save(path, "JPEG")
    return path


@pytest.fixture
def sample_jpeg_variant(tmp_photos_dir: Path) -> Path:
    path = tmp_photos_dir / "variant.jpg"
    Image.new("RGB", (64, 64), color=(130, 66, 34)).save(path, "JPEG")
    return path


@pytest.fixture
def sample_jpeg_different(tmp_photos_dir: Path) -> Path:
    path = tmp_photos_dir / "different.jpg"
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    for x in range(64):
        for y in range(64):
            img.putpixel((x, y), ((x * 4) % 256, (y * 4) % 256, (x + y) % 256))
    img.save(path, "JPEG")
    return path


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATA_DIR", str(data_dir))

    import importlib
    import services.database as database_module
    importlib.reload(database_module)
    database_module.init_db()
    yield database_module
    database_module = importlib.reload(database_module)
