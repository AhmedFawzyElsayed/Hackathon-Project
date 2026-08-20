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

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small enough for container platforms.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download the models at BUILD time so the container starts without
# hitting Hugging Face at runtime (avoids network/rate-limit failures on boot).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('ncbi/MedCPT-Cross-Encoder')"
# Second model: the MiniLM fallback reranker, so a MedCPT load failure still
# leaves a working (degraded) reranking path without any network access.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY rag_core/ rag_core/
COPY backend/app/ backend/app/
COPY backend/tests/ backend/tests/

# Corpus + persisted index (built with the config.py defaults / BGE-M3).
COPY 2026-guidelines-for-the-early-detection-of-prostate-cancer.pdf ./
COPY index_store/ index_store/

# Built UI
COPY --from=frontend /app/frontend/dist frontend/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]