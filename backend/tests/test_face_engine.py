import numpy as np

from services.face_engine import FaceEngine


def test_compute_centroid_empty_returns_none() -> None:
    assert FaceEngine.compute_centroid(np.array([])) is None
    assert FaceEngine.compute_centroid(np.empty((0, 512))) is None


def test_compute_centroid_single_vector_is_normalized() -> None:
    v = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    c = FaceEngine.compute_centroid(v)
    assert c is not None
    assert c.dtype == np.float32
    assert np.isclose(np.linalg.norm(c), 1.0)


def test_compute_centroid_is_mean_then_normalized() -> None:
    vecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    c = FaceEngine.compute_centroid(vecs)
    expected_mean = vecs.mean(axis=0)
    expected = expected_mean / np.linalg.norm(expected_mean)
    assert np.allclose(c, expected.astype(np.float32))


def test_compute_centroid_all_zero_stays_zero() -> None:
    vecs = np.zeros((3, 5), dtype=np.float32)
    c = FaceEngine.compute_centroid(vecs)
    assert c is not None
    assert np.allclose(c, 0.0)
