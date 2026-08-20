# --- Stage 1: build the React frontend -------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python backend + serve the built frontend ---------------------
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch FIRST, so the torch dep pulled by sentence-transformers is
# already satisfied and never triggers a CUDA download on a CPU container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download ALL models at BUILD time so boot needs no network access
# (avoids runtime Hugging Face failures/rate-limits and slow cold starts).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('ncbi/MedCPT-Cross-Encoder')"
# Second (fallback) reranker: loaded only if MedCPT fails at runtime, so the
# app still has a working reranking path with no network available.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY rag_core/ rag_core/
COPY backend/app/ backend/app/
COPY backend/tests/ backend/tests/

# Corpus + persisted index (paths must match rag_core/config.py).
COPY 2026-guidelines-for-the-early-detection-of-prostate-cancer.pdf ./
COPY index_store/ index_store/

# Built UI
COPY --from=frontend /app/frontend/dist frontend/dist

# Runtime: all models are already in the build cache, so force offline mode to
# avoid network hangs/timeouts on boot. (Set AFTER the model pre-download RUNs,
# which need the network.)
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]