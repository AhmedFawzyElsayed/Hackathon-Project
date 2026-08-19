from fastapi import APIRouter, HTTPException

from app.schemas.chat import AskRequest, AskResponse, HealthResponse
from app.services import rag_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")
    try:
        return rag_service.ask(question)
    except FileNotFoundError as e:
        # Index artifacts missing — a config problem, not a bad request.
        raise HTTPException(status_code=503, detail=str(e)) from e
