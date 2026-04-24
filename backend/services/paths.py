import os


def safe_photo_path(base: str, rel: str) -> str | None:
    """Resolve `rel` under `base` and reject any result that escapes `base`.

    Returns the real absolute path on success, or None if the candidate
    path points outside `base` (path traversal, absolute path, or a symlink
    that resolves outside the base directory).
    """
    base_real = os.path.realpath(base)
    candidate = os.path.realpath(os.path.join(base_real, rel))
    if candidate == base_real or candidate.startswith(base_real + os.sep):
        return candidate
    return None
