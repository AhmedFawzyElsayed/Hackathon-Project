from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class Citation(BaseModel):
    document_id: str
    section: str
    page: int
    chunk_id: str


class Claim(BaseModel):
    claim_text: str
    citation: Citation | None


class GenerationMetrics(BaseModel):
    faithfulness: float
    answer_relevance: float
    hallucination_rate: float
    context_utilization: float


class AskResponse(BaseModel):
    status: str
    answer_summary: str
    claims: list[Claim]
    citation_coverage: float
    unverified_citations: list[str]
    confidence_label: str
    conversation_id: str
    metrics: GenerationMetrics


class HealthResponse(BaseModel):
    status: str
