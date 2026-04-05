# PhotoSort

Local AI-powered photo classifier. Runs 100% on your machine — no cloud, no data leaves your computer.

## Features

- **Smart Classification** — CLIP model classifies photos into custom categories
- **Text Search** — Search your photos by describing what you're looking for
- **Face Recognition** — Detect and group photos by person
- **Duplicate Detection** — Find and manage duplicate photos
- **Web UI + CLI** — Visual interface and command-line tools

## Quick Start

```bash
# Clone
git clone https://github.com/JLDominguezM/photosort.git
cd photosort

# Configure
cp .env.example .env
# Edit .env and set PHOTOS_DIR to your photos folder

# Launch
docker compose up --build
```

Open `http://localhost:8080` and click **Scan Photos**.

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with CUDA drivers (for GPU acceleration)
- `nvidia-container-toolkit` installed

For CPU-only mode, set `DEVICE=cpu` in `.env` and remove the `runtime: nvidia` and `deploy` sections from `docker-compose.yml`.

## CLI

```bash
docker compose exec backend python cli.py scan          # Import photos
docker compose exec backend python cli.py classify      # Classify photos
docker compose exec backend python cli.py search "beach" # Search by text
docker compose exec backend python cli.py faces detect   # Detect faces
docker compose exec backend python cli.py faces cluster  # Group faces
docker compose exec backend python cli.py duplicates     # Find duplicates
docker compose exec backend python cli.py stats          # Show stats
```

## Custom Categories

Edit `config/categories.yml` to define your own categories:

```yaml
categories:
  - name: "Vacations"
    prompts:
      - "a vacation photo at a tourist destination"
      - "a travel photo"
  - name: "Concerts"
    prompts:
      - "a photo of a live concert"

threshold: 0.22
batch_size: 32
```

## Architecture

- **Backend**: FastAPI + OpenCLIP (ViT-B/32) + InsightFace + SQLite
- **Frontend**: Vanilla JS + nginx
- **Container**: NVIDIA CUDA runtime with GPU passthrough
