"""
Backend tests for the Clinical RAG Assistant.

Phase 3 step 6 tests (supported question, patient-specific refusal, empty
input) plus new tests for conversation memory and generation metrics.

All tests mock rag_core.answer_question() — no index build, no LLM key needed.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — canonical mock response shapes
# ---------------------------------------------------------------------------
_FAKE_ANSWERED = {
    "status": "answered",
    "answer_summary": "Total PSA of 3 or greater leads to repeating the test.",
    "claims": [
        {
            "claim_text": "Total PSA of 3 or greater leads to repeating the test.",
            "citation": {
                "document_id": "PCFA-EDPC-2026-001",
                "section": "Section D: Early detection",
                "page": 142,
                "chunk_id": "PCFA-EDPC-2026-001-CH-0142",
            },
        }
    ],
    "citation_coverage": 1.0,
    "unverified_citations": [],
    "confidence_label": "high",
    "metrics": {
        "faithfulness": 0.85,
        "answer_relevance": 0.75,
        "hallucination_rate": 0.0,
        "context_utilization": 0.2,
    },
}

_FAKE_REFUSAL = {
    "status": "patient_specific_refusal",
    "answer_summary": "I can't give advice about your individual results...",
    "claims": [],
    "citation_coverage": 0.0,
    "unverified_citations": [],
    "confidence_label": "n/a",
    "metrics": {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "hallucination_rate": 0.0,
        "context_utilization": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Original tests (updated to include new fields)
# ---------------------------------------------------------------------------
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_supported_question_returns_answered(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_ANSWERED
    )
    resp = client.post(
        "/ask", json={"question": "What PSA threshold triggers a repeat test?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["claims"][0]["citation"]["page"] == 142
    assert body["metrics"]["hallucination_rate"] == 0.0


def test_ask_patient_specific_question_returns_refusal(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_REFUSAL
    )
    resp = client.post("/ask", json={"question": "I'm 55, is my PSA of 4.2 normal?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "patient_specific_refusal"


def test_ask_rejects_empty_question():
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------
def test_ask_returns_conversation_id(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_ANSWERED
    )
    resp = client.post("/ask", json={"question": "What is PSA?"})
    body = resp.json()
    assert "conversation_id" in body
    assert isinstance(body["conversation_id"], str)
    assert len(body["conversation_id"]) > 0


def test_ask_with_conversation_id_returns_same_id(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_ANSWERED
    )
    cid = "test-session-123"
    resp1 = client.post(
        "/ask", json={"question": "What is PSA?", "conversation_id": cid}
    )
    assert resp1.json()["conversation_id"] == cid

    resp2 = client.post(
        "/ask", json={"question": "What about PSAD?", "conversation_id": cid}
    )
    assert resp2.json()["conversation_id"] == cid


def test_ask_without_conversation_id_generates_one(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_ANSWERED
    )
    resp1 = client.post("/ask", json={"question": "Question A"})
    resp2 = client.post("/ask", json={"question": "Question B"})
    assert resp1.json()["conversation_id"] != resp2.json()["conversation_id"]


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------
def test_ask_response_includes_metrics(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_ANSWERED
    )
    resp = client.post("/ask", json={"question": "What is PSA?"})
    metrics = resp.json()["metrics"]
    assert "faithfulness" in metrics
    assert "answer_relevance" in metrics
    assert "hallucination_rate" in metrics
    assert "context_utilization" in metrics


def test_metrics_values_are_floats_between_0_and_1(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_ANSWERED
    )
    resp = client.post("/ask", json={"question": "What is PSA?"})
    for key, val in resp.json()["metrics"].items():
        assert isinstance(val, float), f"{key} should be float"
        assert 0.0 <= val <= 1.0, f"{key} should be in [0, 1], got {val}"


def test_refusal_has_zero_metrics(monkeypatch):
    monkeypatch.setattr(
        "app.services.rag_service.answer_question", lambda q, **kw: _FAKE_REFUSAL
    )
    resp = client.post("/ask", json={"question": "Should I get tested?"})
    metrics = resp.json()["metrics"]
    assert metrics == {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "hallucination_rate": 0.0,
        "context_utilization": 0.0,
    }


# ---------------------------------------------------------------------------
# Rag_core metrics unit tests
# ---------------------------------------------------------------------------
def test_compute_faithfulness():
    from rag_core.metrics import compute_faithfulness

    claims = [{"claim_text": "PSA testing is recommended every two years"}]
    evidence = [
        {"text": "For males aged 50 to 69, offer PSA testing every 2 years subject to clinical assessment."}
    ]
    score = compute_faithfulness(claims, evidence)
    assert 0.0 <= score <= 1.0
    assert score > 0.15  # should be nonzero — shares "psa", "testing", "every", "years"


def test_compute_faithfulness_empty():
    from rag_core.metrics import compute_faithfulness

    assert compute_faithfulness([], []) == 0.0
    assert compute_faithfulness([{"claim_text": "hello"}], []) == 0.0


def test_compute_answer_relevance():
    from rag_core.metrics import compute_answer_relevance

    score = compute_answer_relevance(
        "What PSA threshold triggers a repeat test?",
        "PSA of 3 or greater triggers a repeat test within 1 to 3 months.",
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.3  # shares "psa", "threshold"/"triggers", "repeat", "test"


def test_compute_hallucination_rate():
    from rag_core.metrics import compute_hallucination_rate

    cited = [
        {"claim_text": "A", "citation": {"chunk_id": "c1"}},
        {"claim_text": "B", "citation": {"chunk_id": "c2"}},
    ]
    assert compute_hallucination_rate(cited) == 0.0

    mixed = [
        {"claim_text": "A", "citation": {"chunk_id": "c1"}},
        {"claim_text": "B", "citation": None},
    ]
    assert compute_hallucination_rate(mixed) == 0.5

    uncited = [{"claim_text": "A", "citation": None}]
    assert compute_hallucination_rate(uncited) == 1.0

    assert compute_hallucination_rate([]) == 0.0


def test_compute_context_utilization():
    from rag_core.metrics import compute_context_utilization

    claims = [{"citation": {"chunk_id": "c1"}}]
    evidence = [
        {"chunk_id": "c1", "text": "a"},
        {"chunk_id": "c2", "text": "b"},
        {"chunk_id": "c3", "text": "c"},
    ]
    assert compute_context_utilization(claims, evidence) == round(1 / 3, 4)


def test_compute_context_utilization_empty():
    from rag_core.metrics import compute_context_utilization

    assert compute_context_utilization([], []) == 0.0
