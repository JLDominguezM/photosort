import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    device = os.getenv("DEVICE", "cuda")

    # Lazy imports to avoid slow startup if models aren't needed yet
    from services.clip_engine import CLIPEngine
    from services.face_engine import FaceEngine
    from services.hash_engine import HashEngine

    app.state.clip = CLIPEngine(device=device)
    app.state.clip.load_categories(os.path.join(os.getenv("CONFIG_DIR", "config"), "categories.yml"))
    app.state.faces = FaceEngine(device=device)
    app.state.hasher = HashEngine()

    yield


app = FastAPI(title="PhotoSort", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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
