FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model caches live under /models, mounted from a docker volume so the
# ~500 MB of weights are downloaded once and persist across rebuilds.
ENV HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    INSIGHTFACE_HOME=/models/insightface

COPY . .
RUN mkdir -p data/thumbnails /models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
