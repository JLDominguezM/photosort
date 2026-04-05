from pydantic import BaseModel


class PhotoOut(BaseModel):
    id: int
    filepath: str
    filename: str
    filesize: int
    width: int | None = None
    height: int | None = None
    taken_at: str | None = None
    imported_at: str
    category: str | None = None
    confidence: float | None = None


class PhotoList(BaseModel):
    photos: list[PhotoOut]
    total: int
    page: int
    per_page: int


class ClassifyResult(BaseModel):
    photo_id: int
    category: str
    confidence: float


class CategoryOut(BaseModel):
    name: str
    count: int


class FaceOut(BaseModel):
    id: int
    photo_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    person_id: int | None = None


class PersonOut(BaseModel):
    id: int
    name: str | None = None
    face_count: int
    photo_count: int


class DuplicateGroup(BaseModel):
    group_id: int
    photos: list[PhotoOut]


class SearchResult(BaseModel):
    photo: PhotoOut
    score: float


class JobStatus(BaseModel):
    job_id: str
    name: str
    status: str
    progress: int
    total: int
    result: dict | None = None


class StatsOut(BaseModel):
    total_photos: int
    classified: int
    uncategorized: int
    faces_detected: int
    persons: int
    duplicate_groups: int
