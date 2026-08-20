"""Vercel serverless entrypoint for the Clinical RAG API.

The real backend cannot run on Vercel (torch + sentence-transformers +
BGE-M3 exceed the 250 MB bundle and 1 GB RAM limits), so this function is a
graceful stub. The full stack runs on Railway (see Dockerfile / railway.json);
point the Vercel frontend at it via VITE_API_BASE_URL and allow the origin
with CORS_ALLOW_ORIGINS on the backend.
"""
import json


async def handler(scope, receive, send):
    detail = {
        "detail": "Backend runs on Railway. Set VITE_API_BASE_URL on this "
        "frontend (and CORS_ALLOW_ORIGINS on the backend) to connect.",
    }
    body = json.dumps(detail).encode()
    send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    send({"type": "http.response.body", "body": body})