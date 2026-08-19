from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    document_id: str
    section: str
    page: int
    chunk_id: str


class Claim(BaseModel):
    claim_text: str
    citation: Citation | None


class AskResponse(BaseModel):
    status: str
    answer_summary: str
    claims: list[Claim]
    citation_coverage: float
    unverified_citations: list[str]
    confidence_label: str


class HealthResponse(BaseModel):
    status: str
