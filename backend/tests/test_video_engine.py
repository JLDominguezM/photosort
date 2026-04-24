from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from services.video_engine import extract_keyframes


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (32, 32))
    if not writer.isOpened():
        pytest.skip("no mp4v codec available in this environment")
    for i in range(30):  # 3 seconds at 10 fps
        frame = np.full((32, 32, 3), (i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_keyframes_returns_requested_count(tiny_video: Path) -> None:
    frames = extract_keyframes(str(tiny_video), num_frames=3)
    assert len(frames) == 3
    for f in frames:
        assert f.size == (32, 32)
        assert f.mode == "RGB"


def test_extract_keyframes_num_one(tiny_video: Path) -> None:
    frames = extract_keyframes(str(tiny_video), num_frames=1)
    assert len(frames) == 1


def test_extract_keyframes_bad_path_returns_empty(tmp_path: Path) -> None:
    assert extract_keyframes(str(tmp_path / "missing.mp4")) == []


def test_extract_keyframes_non_video_returns_empty(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video")
    assert extract_keyframes(str(fake)) == []


def test_extract_keyframes_zero_count(tiny_video: Path) -> None:
    assert extract_keyframes(str(tiny_video), num_frames=0) == []
