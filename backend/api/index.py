"""Vercel serverless entrypoint for the Clinical RAG API.

This is a graceful stub: Vercel's Python runtime cannot host the full RAG
stack (torch + sentence-transformers + BGE-M3 exceed the 250 MB bundle and
1 GB RAM limits). The real backend runs on Railway — see Dockerfile /
railway.json. Point the Vercel frontend at it via VITE_API_BASE_URL and allow
the origin with CORS_ALLOW_ORIGINS on the backend.
"""
import json

try:
    import sys
    from pathlib import Path

    # backend/ contains the `app` package; the project root contains rag_core.
    _backend = str(Path(__file__).resolve().parent.parent)
    _root = str(Path(__file__).resolve().parent.parent.parent)
    for _p in (_backend, _root):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from mangum import Mangum
    from app.main import app

    handler = Mangum(app, lifespan="off")
except Exception as exc:  # pragma: no cover - surfaces import errors in Vercel logs
    _reason = f"RAG backend unavailable on Vercel: {exc}"

    def handler(_scope, _receive, _send):
        body = json.dumps({"detail": _reason}).encode()
        _send(
            {
                "type": "http.response.start",
                "status": 503,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        _send({"type": "http.response.body", "body": body})