import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.database import init_db
from services.logging_config import get_logger

log = get_logger("photosort.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    device = os.getenv("DEVICE", "cuda")
    log.info("Starting PhotoSort (device=%s)", device)

    from services.clip_engine import CLIPEngine
    from services.face_engine import FaceEngine
    from services.hash_engine import HashEngine

    app.state.clip = CLIPEngine(device=device)
    app.state.clip.load_categories(os.path.join(os.getenv("CONFIG_DIR", "config"), "categories.yml"))
    app.state.faces = FaceEngine(device=device)
    app.state.hasher = HashEngine()
    log.info("Engines loaded")

    yield
    log.info("Shutting down")


app = FastAPI(title="PhotoSort", lifespan=lifespan)


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

from api.routes_photos import router as photos_router
from api.routes_classify import router as classify_router
from api.routes_search import router as search_router
from api.routes_faces import router as faces_router
from api.routes_duplicates import router as duplicates_router

app.include_router(photos_router)
app.include_router(classify_router)
app.include_router(search_router)
app.include_router(faces_router)
app.include_router(duplicates_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
