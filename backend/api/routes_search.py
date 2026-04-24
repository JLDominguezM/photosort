import numpy as np
from fastapi import APIRouter, Depends, Query, Request, HTTPException

from api.deps import get_app_db
from models.schemas import PhotoOut, SearchResult, SearchResultPage
from services.logging_config import get_logger

router = APIRouter(prefix="/api", tags=["Search"])

log = get_logger(__name__)


def _load_embeddings(db) -> tuple[list[int], np.ndarray]:
    rows = db.execute(
        "SELECT id, clip_embedding FROM photos WHERE clip_embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return [], np.array([])
    ids = [r["id"] for r in rows]
    embeddings = np.stack([np.frombuffer(r["clip_embedding"], dtype=np.float32) for r in rows])
    return ids, embeddings


def _photo_row(db, photo_id: int):
    return db.execute(
        """SELECT p.*, c.category, c.confidence
           FROM photos p
           LEFT JOIN classifications c ON c.photo_id = p.id
               AND c.confidence = (SELECT MAX(c2.confidence) FROM classifications c2 WHERE c2.photo_id = p.id)
           WHERE p.id = ?""",
        (photo_id,),
    ).fetchone()


@router.get("/search", response_model=SearchResultPage, summary="Semantic text search")
def text_search(
    q: str = Query(..., min_length=1, description="Natural-language query"),
    page: int = Query(1, ge=1),
    per_page: int = Query(40, ge=1, le=200),
    request: Request = None,
    db=Depends(get_app_db),
):
    """Rank all embedded photos by cosine similarity to the query text, then
    return a paginated slice. Scoring is done over the full library; only the
    current page is materialized into response objects."""
    clip = request.app.state.clip
    ids, embeddings = _load_embeddings(db)
    if len(ids) == 0:
        return SearchResultPage(results=[], total=0, page=page, per_page=per_page)

    window = page * per_page
    ranked = clip.search(q, embeddings, top_k=window)
    slice_start = (page - 1) * per_page
    page_slice = ranked[slice_start:slice_start + per_page]

    log.info("search q=%r page=%d results=%d/%d", q, page, len(page_slice), len(ids))

    results = []
    for idx, score in page_slice:
        row = _photo_row(db, ids[idx])
        if row:
            results.append(SearchResult(photo=PhotoOut(**dict(row)), score=score))

    return SearchResultPage(
        results=results,
        total=len(ids),
        page=page,
        per_page=per_page,
    )


@router.get("/search/similar/{photo_id}", response_model=SearchResultPage, summary="Find visually similar photos")
def find_similar(
    photo_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(40, ge=1, le=200),
    request: Request = None,
    db=Depends(get_app_db),
):
    """Return photos most similar to the given photo's CLIP embedding."""
    clip = request.app.state.clip
    row = db.execute("SELECT clip_embedding FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if not row or not row["clip_embedding"]:
        raise HTTPException(404, "Photo not found or not embedded")

    target = np.frombuffer(row["clip_embedding"], dtype=np.float32)
    ids, embeddings = _load_embeddings(db)
    if len(ids) == 0:
        return SearchResultPage(results=[], total=0, page=page, per_page=per_page)

    window = page * per_page + 1  # +1 so dropping self still fills the page
    ranked = clip.find_similar(target, embeddings, top_k=window)
    filtered = [(idx, s) for idx, s in ranked if ids[idx] != photo_id]

    slice_start = (page - 1) * per_page
    page_slice = filtered[slice_start:slice_start + per_page]

    results = []
    for idx, score in page_slice:
        r = _photo_row(db, ids[idx])
        if r:
            results.append(SearchResult(photo=PhotoOut(**dict(r)), score=score))

    return SearchResultPage(
        results=results,
        total=max(len(ids) - 1, 0),
        page=page,
        per_page=per_page,
    )
