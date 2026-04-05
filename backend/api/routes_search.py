import os

import numpy as np
from fastapi import APIRouter, Depends, Query, Request, HTTPException

from api.deps import get_app_db
from models.schemas import PhotoOut, SearchResult

router = APIRouter(prefix="/api")


def _load_embeddings(db) -> tuple[list[int], np.ndarray]:
    rows = db.execute(
        "SELECT id, clip_embedding FROM photos WHERE clip_embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return [], np.array([])
    ids = [r["id"] for r in rows]
    embeddings = np.stack([np.frombuffer(r["clip_embedding"], dtype=np.float32) for r in rows])
    return ids, embeddings


@router.get("/search")
def text_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=200),
    request: Request = None,
    db=Depends(get_app_db),
):
    clip = request.app.state.clip
    ids, embeddings = _load_embeddings(db)
    if len(ids) == 0:
        return []

    results = clip.search(q, embeddings, top_k=top_k)

    photo_results = []
    for idx, score in results:
        photo_id = ids[idx]
        row = db.execute(
            """SELECT p.*, c.category, c.confidence
               FROM photos p
               LEFT JOIN classifications c ON c.photo_id = p.id
                   AND c.confidence = (SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = p.id)
               WHERE p.id = ?""",
            (photo_id,),
        ).fetchone()
        if row:
            photo_results.append(SearchResult(
                photo=PhotoOut(**dict(row)),
                score=score,
            ))
    return photo_results


@router.get("/search/similar/{photo_id}")
def find_similar(
    photo_id: int,
    top_k: int = Query(20, ge=1, le=200),
    request: Request = None,
    db=Depends(get_app_db),
):
    clip = request.app.state.clip
    row = db.execute("SELECT clip_embedding FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row or not row["clip_embedding"]:
        raise HTTPException(404, "Photo not found or not embedded")

    target = np.frombuffer(row["clip_embedding"], dtype=np.float32)
    ids, embeddings = _load_embeddings(db)
    if len(ids) == 0:
        return []

    results = clip.find_similar(target, embeddings, top_k=top_k + 1)

    photo_results = []
    for idx, score in results:
        pid = ids[idx]
        if pid == photo_id:
            continue
        r = db.execute(
            """SELECT p.*, c.category, c.confidence
               FROM photos p
               LEFT JOIN classifications c ON c.photo_id = p.id
                   AND c.confidence = (SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = p.id)
               WHERE p.id = ?""",
            (pid,),
        ).fetchone()
        if r:
            photo_results.append(SearchResult(photo=PhotoOut(**dict(r)), score=score))
    return photo_results[:top_k]
