"""Vercel serverless entrypoint for the Clinical RAG API.

NOTE: this function cannot actually run the full RAG stack on Vercel
(torch + sentence-transformers + BGE-M3 far exceed the 250 MB bundle / 1 GB
RAM limits). It exists so the frontend static build + API route wiring deploy
cleanly. Run the real backend on Railway (see Dockerfile / railway.json) and
point the frontend at it via VITE_API_BASE_URL.
"""
import sys
from pathlib import Path

# backend/ contains the `app` package; the project root contains rag_core.
_backend = str(Path(__file__).resolve().parent.parent)
_root = str(Path(__file__).resolve().parent.parent.parent)
for _p in (_backend, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from mangum import Mangum

    from app.main import app

    handler = Mangum(app, lifespan="off")
except Exception as exc:  # pragma: no cover - surfaces import errors in Vercel logs
    def handler(_scope, _receive, _send):  # type: ignore
        raise exc