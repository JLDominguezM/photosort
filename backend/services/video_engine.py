import cv2
from PIL import Image

from services.logging_config import get_logger

log = get_logger(__name__)


def extract_keyframes(video_path: str, num_frames: int = 3) -> list[Image.Image]:
    """Sample `num_frames` evenly-spaced frames from a video and return them
    as PIL RGB images. Returns an empty list if the video can't be opened or
    has no readable frames.

    The sampling points are the fractions i/(num_frames+1) of total frames,
    which avoids the very first/last frames (often black) while still
    covering the full duration.
    """
    if num_frames < 1:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("could not open video %s", video_path)
        return []

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        frames: list[Image.Image] = []
        for i in range(1, num_frames + 1):
            target = int(total * i / (num_frames + 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        return frames
    finally:
        cap.release()
