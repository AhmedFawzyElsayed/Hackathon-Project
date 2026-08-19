"""
Phase 3 step 6: at least one supported-question test and one patient-specific
refusal test.

These mock rag_core.answer_question() rather than calling it for real, so
the suite runs fast and offline (no index build, no LLM key needed) in CI.
For a real end-to-end check, build the index (`python -m rag_core.build_index`)
and hit `/ask` manually or add a separate `@pytest.mark.integration` test.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_supported_question_returns_answered(monkeypatch):
    fake_result = {
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
    }
    monkeypatch.setattr("app.services.rag_service.answer_question", lambda q: fake_result)

    resp = client.post("/ask", json={"question": "What PSA threshold triggers a repeat test?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["claims"][0]["citation"]["page"] == 142


def test_ask_patient_specific_question_returns_refusal(monkeypatch):
    fake_result = {
        "status": "patient_specific_refusal",
        "answer_summary": "I can't give advice about your individual results...",
        "claims": [],
        "citation_coverage": 0.0,
        "unverified_citations": [],
        "confidence_label": "n/a",
    }
    monkeypatch.setattr("app.services.rag_service.answer_question", lambda q: fake_result)

    resp = client.post("/ask", json={"question": "I'm 55, is my PSA of 4.2 normal?"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "patient_specific_refusal"


def test_ask_rejects_empty_question():
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 422
