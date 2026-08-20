import sys
from contextlib import asynccontextmanager
from pathlib import Path

# The project root lives one level above backend/, and rag_core lives there.
# Add it so `from rag_core import ...` resolves regardless of CWD.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.chat import router as chat_router
from app.core.config import settings
from rag_core import load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the persisted index once at startup �?" never per-request. If
    # index_store/ doesn't exist yet, log it and let requests fail with a
    # clear 503 rather than crashing the whole app on boot.
    try:
        print("Loading rag_core index (BGE-M3 embedder) ...", flush=True)
        load_index()
        print("rag_core index loaded.", flush=True)
    except FileNotFoundError as e:
        print(f"WARNING: {e}", flush=True)
    yield


app = FastAPI(title="Clinical RAG Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

# Serve the built React frontend (if present) so a single container can run
# both the API and the UI. Mounted last so /health and /ask take precedence.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists() and (_FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        target = _FRONTEND_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(_FRONTEND_DIST / "index.html")
