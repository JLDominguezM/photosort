# PhotoSort

Local AI-powered photo classifier. Runs 100% on your machine — no cloud, no data leaves your computer.

PhotoSort indexes a folder of photos, classifies them into customizable categories with CLIP, lets you search by natural-language description, groups photos by detected faces, and finds near-duplicates using perceptual hashing. It exposes both a web UI and a CLI.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start (Docker)](#quick-start-docker)
- [CPU-only mode](#cpu-only-mode)
- [Running without Docker](#running-without-docker)
- [Environment variables](#environment-variables)
- [Usage](#usage)
  - [Web UI](#web-ui)
  - [CLI](#cli)
- [Custom Categories](#custom-categories)
- [API Reference](#api-reference)
- [Project Layout](#project-layout)
- [Data & Storage](#data--storage)
- [Logs](#logs)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [Roadmap](#roadmap)

---

## Features

- **Smart Classification** — Open CLIP (ViT-B/32, LAION-2B) assigns each photo to user-defined categories with a configurable confidence threshold.
- **Semantic Text Search** — Find photos by describing them (e.g. `"sunset at the beach"`).
- **Similarity Search** — Given a photo, find visually similar ones.
- **Face Detection & Grouping** — InsightFace (`buffalo_s`) + DBSCAN clustering. Rename persons and merge clusters manually.
- **Duplicate Detection** — Perceptual hashing (`phash`) groups near-identical photos; you pick which to keep.
- **HEIC / HEIF support** — Natively reads iPhone photos.
- **Background Jobs** — Long operations (scan, classify, detect, dedupe) run async with progress tracking.
- **Web UI + CLI** — Use whichever you prefer.
- **GPU-accelerated** (optional) — CUDA 12.6 runtime; falls back to CPU.

---

## Architecture

```
┌──────────────┐      ┌────────────────────┐      ┌──────────────────┐
│  Browser UI  │─────▶│  nginx (port 8080) │─────▶│  FastAPI backend │
│  vanilla JS  │      │  reverse proxy     │      │  (port 8000)     │
└──────────────┘      └────────────────────┘      └──────┬───────────┘
                                                         │
                             ┌───────────────────────────┼───────────────────────────┐
                             ▼                           ▼                           ▼
                     ┌───────────────┐          ┌────────────────┐         ┌──────────────────┐
                     │ SQLite (WAL)  │          │ OpenCLIP       │         │ InsightFace      │
                     │ photos, faces │          │ ViT-B/32       │         │ buffalo_s        │
                     │ embeddings    │          │ classification │         │ detect + cluster │
                     └───────────────┘          └────────────────┘         └──────────────────┘
```

- **Backend:** Python 3.11, FastAPI, Uvicorn, SQLite (WAL + foreign keys), OpenCLIP, InsightFace + ONNX Runtime, imagehash, scikit-learn (DBSCAN).
- **Frontend:** Vanilla JS (no build step), served by nginx Alpine which proxies `/api/` to the backend.
- **Container:** `nvidia/cuda:12.6.3-runtime-ubuntu22.04` with GPU passthrough.

---

## Requirements

### With Docker (recommended)

- Docker Engine 24+ and Docker Compose v2
- For GPU mode: an NVIDIA GPU, recent drivers (supporting CUDA 12.6), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host
- ~8 GB disk for the backend image (models are pre-downloaded into the image)

### Without Docker

- Python 3.11
- ~4 GB disk for models and dependencies
- For GPU: CUDA 12.x drivers + matching `torch` / `onnxruntime-gpu` wheels

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/JLDominguezM/photosort.git
cd photosort

# 2. Configure
cp .env.example .env
# Open .env and set PHOTOS_DIR to the absolute path of your photo folder
#   e.g. PHOTOS_DIR=/home/you/Pictures

# 3. Build and launch
docker compose up --build
```

First build takes ~10–15 minutes (downloads CLIP + InsightFace weights into the image). Subsequent `up` runs are near-instant.

Open:

- **Web UI:** http://localhost:8080
- **API docs (Swagger):** http://localhost:8000/docs
- **Health:** http://localhost:8000/api/health

On the Gallery page, click **Scan Photos** to import the folder, then **Classify** to run CLIP.

---

## CPU-only mode

If you don't have an NVIDIA GPU:

1. Set `DEVICE=cpu` in `.env`.
2. Edit `docker-compose.yml` and remove both the `runtime: nvidia` line and the whole `deploy:` block under `backend:`.
3. Install `onnxruntime` (CPU) instead of `onnxruntime-gpu` in `backend/requirements.txt` if you want to shave image size.

CPU classification of ~1000 photos takes a few minutes; GPU typically <30 s.

---

## Running without Docker

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Point to your photos and config
export PHOTOS_DIR=/absolute/path/to/photos
export CONFIG_DIR=../config
export DATA_DIR=./data
export DEVICE=cpu    # or cuda

uvicorn main:app --host 0.0.0.0 --port 8000
```

Then serve `frontend/` with any static server (or just open `frontend/index.html` — it assumes `/api` is proxied, so for a raw file-open you'd need to change `fetch('/api/…')` calls in `app.js` to `http://localhost:8000/api/…`).

---

## Environment variables

All variables live in `.env` (copied from `.env.example`). Everything is optional except `PHOTOS_DIR`.

| Variable | Default | Purpose |
|---|---|---|
| `PHOTOS_DIR` | — **(required)** | Absolute host path to your photo folder. Mounted read-only. |
| `DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only. |
| `CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | Comma-separated allowed origins for the API. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `DATA_DIR` | `/app/data` (Docker) / `./data` | Where SQLite DB and thumbnails are written. |
| `CONFIG_DIR` | `/app/config` (Docker) / `./config` | Where `categories.yml` is read from. |

---

## Usage

### Web UI

Five pages, hash-routed (`#gallery`, `#search`, `#faces`, `#duplicates`, `#settings`):

- **Gallery** — Paginated grid. Filter by category, date range, or person.
- **Search** — Type a natural-language description; results ranked by CLIP similarity.
- **Faces** — List of auto-discovered persons. Click one to see their photos. Rename inline; merge two clusters via the modal.
- **Duplicates** — Review each group side-by-side and mark the one to keep.
- **Settings** — Library stats, buttons to trigger Scan / Classify / Detect Faces / Cluster / Find Duplicates.

Click any photo to open the detail modal (full image, EXIF, assign category, see detected faces).

### CLI

The CLI hits the same data as the UI and is useful for scripting or initial bulk imports:

```bash
# Inside the backend container:
docker compose exec backend python cli.py scan           # import new files
docker compose exec backend python cli.py classify       # CLIP classify unclassified
docker compose exec backend python cli.py classify --force   # reclassify all
docker compose exec backend python cli.py search "beach sunset"
docker compose exec backend python cli.py faces detect
docker compose exec backend python cli.py faces cluster
docker compose exec backend python cli.py duplicates
docker compose exec backend python cli.py stats
```

---

## Custom Categories

Edit `config/categories.yml`:

```yaml
categories:
  - name: "Vacations"
    prompts:
      - "a vacation photo at a tourist destination"
      - "a travel photo, a sightseeing picture"
  - name: "Food"
    prompts:
      - "a photo of food on a plate"
      - "a close-up of a meal at a restaurant"
  - name: "Pets"
    prompts:
      - "a photo of a pet dog or cat"

threshold: 0.22   # minimum cosine similarity to assign a category
batch_size: 32    # CLIP encoding batch size
```

Rules:

- `name` — 1–64 chars, displayed in the UI.
- `prompts` — 1–20 strings. Text embeddings are averaged per category, so more varied prompts → more robust matching.
- `threshold` — 0.0–1.0. Photos below this score become `Uncategorized`.
- `batch_size` — 1–256. Higher uses more VRAM.

After editing:

1. Restart the backend (`docker compose restart backend`) — or call `PUT /api/categories` with the same payload to hot-reload.
2. Run `classify --force` (CLI) or click **Reclassify** (UI) to re-apply to existing photos.

The API also validates and **backs up** `categories.yml` before overwriting (`categories.yml.bak.<timestamp>`).

---

## API Reference

Full interactive docs at `http://localhost:8000/docs`.

**Photos & Library**
- `POST /api/scan` — import new files (async job)
- `GET  /api/photos?page=&per_page=&category=&date_from=&date_to=&person_id=`
- `GET  /api/photos/{id}` — details
- `GET  /api/photos/{id}/thumbnail` — 300 px JPEG
- `GET  /api/photos/{id}/full` — original
- `GET  /api/stats`

**Classification**
- `POST /api/classify?force=false` — async
- `POST /api/classify/{photo_id}` — single
- `PUT  /api/photos/{photo_id}/category` — manual assignment
- `GET  /api/categories`
- `PUT  /api/categories` — validated + backed-up

**Search**
- `GET /api/search?q=<text>&top_k=20`
- `GET /api/search/similar/{photo_id}?top_k=20`

**Faces**
- `POST /api/faces/detect` — async
- `POST /api/faces/cluster`
- `GET  /api/faces/persons`
- `GET  /api/faces/persons/{id}`
- `PUT  /api/faces/persons/{id}` — rename
- `POST /api/faces/persons/merge` — body: `{"person_a": int, "person_b": int}`
- `GET  /api/faces/{photo_id}/crops`

**Duplicates**
- `POST /api/duplicates/scan`
- `GET  /api/duplicates`
- `POST /api/duplicates/{group_id}/keep/{photo_id}`

**Jobs & Health**
- `GET /api/jobs`, `GET /api/jobs/{job_id}`
- `GET /api/health`

---

## Project Layout

```
photosort/
├── README.md
├── .env.example
├── docker-compose.yml
├── config/
│   └── categories.yml          # User-editable categories
├── backend/
│   ├── Dockerfile
│   ├── main.py                 # FastAPI entry + lifespan
│   ├── cli.py                  # Typer CLI
│   ├── requirements.txt
│   ├── api/                    # HTTP routes
│   │   ├── deps.py
│   │   ├── routes_photos.py
│   │   ├── routes_classify.py
│   │   ├── routes_search.py
│   │   ├── routes_faces.py
│   │   └── routes_duplicates.py
│   ├── services/               # Core logic
│   │   ├── database.py         # SQLite schema
│   │   ├── clip_engine.py
│   │   ├── face_engine.py
│   │   ├── hash_engine.py
│   │   ├── scanner.py
│   │   ├── thumbnails.py
│   │   ├── jobs.py             # Async job tracker
│   │   └── logging_config.py
│   └── models/
│       └── schemas.py          # Pydantic models
└── frontend/
    ├── index.html
    ├── app.js
    ├── style.css
    └── nginx.conf              # Proxies /api/ → backend:8000
```

---

## Data & Storage

All state lives under `backend/data/` (bind-mounted from host):

- `photosort.db` — SQLite database (photos, classifications, faces, persons, duplicate groups, embeddings).
- `thumbnails/{photo_id}.jpg` — 300 px thumbnails, generated on first request.

Your original photos are **never modified** — the `PHOTOS_DIR` volume is mounted read-only.

CLIP embeddings (512 × float32 = 2 KB each) and face embeddings are stored as BLOBs inside SQLite. Expect ~250 MB of DB per 100 k photos.

---

## Logs

With Docker:

```bash
docker compose logs -f backend
docker compose logs -f frontend
```

Backend log format:

```
2026-04-24T10:51:02 INFO [photosort.main] Engines loaded
2026-04-24T10:51:18 WARNING [services.clip_engine] CLIP encode_image failed for /photos/broken.jpg: cannot identify image file
```

Raise verbosity with `LOG_LEVEL=DEBUG` in `.env`.

---

## Troubleshooting

**`docker compose up` fails with `could not select device driver "nvidia"`**
You need the NVIDIA Container Toolkit. Install it and restart Docker, or switch to CPU mode (see above).

**Gallery is empty after scan**
Check `PHOTOS_DIR` really contains supported extensions (`.jpg .jpeg .png .heic .heif .webp .tiff .bmp` — videos `.mov .mp4 .avi .mkv` are indexed but not classified). Look at `docker compose logs backend` for per-file warnings.

**Classification assigns everything to `Uncategorized`**
Lower `threshold` in `config/categories.yml` (try `0.18`) and reclassify.

**HEIC files fail to decode**
`pillow-heif` is included. If it still fails, the file may be HEVC-only (not HEIF); convert with `heif-convert` or ImageMagick.

**Out-of-memory on GPU during classify**
Lower `batch_size` in `categories.yml` (try `8` or `4`).

**Port 8080 or 8000 already in use**
Edit the `ports:` section in `docker-compose.yml` — e.g. `"9080:80"` for the frontend.

**"CORS error" when opening frontend from another host**
Add that origin to `CORS_ORIGINS` in `.env` and restart the backend.

---

## Security Notes

PhotoSort is designed for **trusted local networks**. Before exposing it to the internet you should at minimum:

- Put it behind a reverse proxy with HTTPS and authentication (Caddy / Traefik / nginx + basic auth or OIDC).
- Tighten `CORS_ORIGINS` to the exact origin you serve from.
- Keep the backend off a public port — only expose the frontend.

Path traversal: the backend refuses to serve any file whose resolved path escapes `PHOTOS_DIR`. Symlinks inside `PHOTOS_DIR` that point outside it will also be rejected.

---

## Roadmap

- Batch-delete API for duplicates marked "not keep"
- Video keyframe classification
- Hot-reload of `categories.yml` on file change
- Optional user authentication
- Export of classified/filtered sets (zip, symlink tree)
- Unit + integration test suite
