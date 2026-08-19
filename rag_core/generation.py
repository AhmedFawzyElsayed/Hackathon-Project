"""
rag_core.generation — answer_question(), the one function everything
downstream (backend, frontend) depends on. Lifted from the notebook's
Sections 17-18 and Tasks 4-10.

THE FROZEN CONTRACT — keep this shape stable across every future change:

    def answer_question(question: str) -> dict:
        {
            "status": "answered" | "insufficient_evidence" | "patient_specific_refusal",
            "answer_summary": str,
            "claims": [
                {
                    "claim_text": str,
                    "citation": {"document_id": str, "section": str, "page": int, "chunk_id": str} | None,
                }
            ],
            "citation_coverage": float,
            "unverified_citations": list,
            "confidence_label": str,
        }

If a change needs to *add* a field, that's fine — add it here and update the
Pydantic schema in backend/app/schemas/chat.py and the TS types in
frontend/src/types/chat.ts in the same commit. Never remove or repurpose an
existing field without doing the same.
"""
import json
import os
import re
import time

from . import config
from .indexing import load_index as _load_index_from_disk
from .indexing import tokenize
from .retrieval import retrieve, retrieval_confidence

# ---------------------------------------------------------------------------
# Index singleton — loaded once via load_index(), not per-request.
# ---------------------------------------------------------------------------
_DB = None
_BM25 = None
_CHUNKS = None


def is_index_loaded() -> bool:
    return _DB is not None


def load_index_once(force: bool = False):
    """Loads the persisted index into module-level singletons. Call this
    explicitly from the backend's FastAPI lifespan at startup. Safe to call
    more than once — it's a no-op after the first successful load unless
    force=True."""
    global _DB, _BM25, _CHUNKS
    if is_index_loaded() and not force:
        return _DB, _BM25, _CHUNKS
    _DB, _BM25, _CHUNKS = _load_index_from_disk()
    return _DB, _BM25, _CHUNKS


# Public name used by rag_core/__init__.py and the backend lifespan — call
# this once at startup (e.g. from FastAPI's lifespan) to prime the index.
load_index = load_index_once


def _ensure_index_loaded():
    if not is_index_loaded():
        load_index_once()


# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------
class GroqLLM:
    """Thin wrapper so .invoke(prompt) matches the interface used below."""

    def __init__(self, model: str = config.GROQ_MODEL):
        from groq import Groq as _GroqClient

        # 30s timeout: a stalled/slow network call should fail loudly, not
        # hang the request indefinitely.
        self.client = _GroqClient(api_key=os.environ.get("GROQ_API_KEY"), timeout=30.0)
        self.model = model

    def invoke(self, prompt: str, json_mode: bool = False) -> str:
        """json_mode=True sets Groq's JSON Object Mode, which constrains the
        model to emit syntactically valid JSON. It doesn't guarantee the JSON
        matches our schema (parse_structured_claims still checks that), but
        it stops the model wrapping JSON in prose."""
        kwargs = dict(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful clinical assistant answering questions based on provided medical guidelines.",
                },
                {"role": "user", "content": prompt},
            ],
            model=self.model,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if json_mode:
                # Some models reject response_format outright — retry once
                # in plain mode rather than losing the whole generation.
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                # One blind retry for transient timeouts/connection errors
                # instead of failing a whole request over a single flaky call.
                time.sleep(2)
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                except Exception:
                    raise e
        return resp.choices[0].message.content


_BACKEND_NAME = None
_LLM = None
_backend_checked = False


def get_llm_backend():
    global _BACKEND_NAME, _LLM, _backend_checked
    if _backend_checked:
        return _BACKEND_NAME, _LLM
    _backend_checked = True

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0)
            llm.invoke("test")
            print(f"LLM backend: OpenAI ({config.OPENAI_MODEL})")
            _BACKEND_NAME, _LLM = "openai", llm
            return _BACKEND_NAME, _LLM
        except Exception as e:  # pragma: no cover
            print(f"OpenAI backend unavailable ({e})")

    if os.environ.get("GROQ_API_KEY"):
        try:
            llm = GroqLLM(model=config.GROQ_MODEL)
            llm.invoke("test")
            print(f"LLM backend: Groq ({config.GROQ_MODEL})")
            _BACKEND_NAME, _LLM = "groq", llm
            return _BACKEND_NAME, _LLM
        except Exception as e:  # pragma: no cover
            print(f"Groq backend unavailable ({e})")

    try:
        from langchain_ollama import OllamaLLM

        llm = OllamaLLM(model="llama3")
        llm.invoke("test")
        print("LLM backend: Ollama (llama3, local)")
        _BACKEND_NAME, _LLM = "ollama", llm
        return _BACKEND_NAME, _LLM
    except Exception as e:  # pragma: no cover
        print(f"Ollama backend unavailable ({e})")

    print(
        "LLM backend: none available -- using deterministic extractive fallback. "
        "Set GROQ_API_KEY or OPENAI_API_KEY, or run Ollama locally, for real generation."
    )
    _BACKEND_NAME, _LLM = "extractive", None
    return _BACKEND_NAME, _LLM


# ---------------------------------------------------------------------------
# Grounding rules + structured prompt
# ---------------------------------------------------------------------------
GROUNDING_RULES = [
    "Use ONLY the provided evidence chunks. Do not use outside medical knowledge.",
    "Every claim must be traceable to exactly one evidence chunk_id provided in the context.",
    "If the evidence does not clearly support an answer, return status='insufficient_evidence' "
    "instead of guessing.",
    "Never answer a question that asks for advice about a specific patient's own results, risk, "
    "or next steps -- refuse and direct them to a clinician (handled before generation, via the "
    "patient-specific safety check).",
    "Do not invent a chunk_id, page, or section that was not in the provided evidence.",
    "Output must be valid JSON matching the schema below -- no prose outside the JSON.",
]

_STRUCTURED_ANSWER_SCHEMA_EXAMPLE = {
    "status": "answered",
    "claims": [
        {
            "claim_text": "Total PSA of 3 micrograms per litre or greater leads to repeating the test.",
            "chunk_id": "PCFA-EDPC-2026-001-CH-0142",
        }
    ],
}

GROUNDING_SYSTEM_PROMPT = (
    "You are a clinical guideline assistant answering questions about the prostate cancer "
    "early-detection guideline. Follow these rules strictly:\n- "
    + "\n- ".join(GROUNDING_RULES)
    + "\n\nRespond with JSON only, matching this shape:\n"
    + json.dumps(_STRUCTURED_ANSWER_SCHEMA_EXAMPLE, indent=2)
)


def is_patient_specific_query(question: str) -> bool:
    """True for questions asking for advice about the asker's own case,
    rather than what the guideline says in general."""
    q = question.lower()
    patterns = [
        r"\bshould i\b",
        r"\bam i\b",
        r"\bmy psa\b",
        r"\bwhat should i do\b",
        r"\bdo i need\b",
        r"\bmy risk\b",
        r"\bmy result\b",
    ]
    return any(re.search(p, q) for p in patterns)


def confidence_label(confidence: float) -> str:
    if confidence >= config.CONFIDENCE_THRESHOLD + 1.0:
        return "high"
    if confidence >= config.CONFIDENCE_THRESHOLD + 0.3:
        return "medium"
    return "low"


def build_citation_evidence(reranked_chunks, top_n: int = config.TOP_K):
    """Turns reranked (doc, score, method) tuples into citation-ready
    evidence records — every field a reviewer needs to trace a claim back to
    the exact retrieved text."""
    evidence = []
    for doc, score, method in reranked_chunks[:top_n]:
        evidence.append(
            {
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "document_id": doc.metadata.get("document_id", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "pages": doc.metadata.get("pages", "N/A"),
                "text": doc.page_content.strip(),
                "rerank_score": round(float(score), 4),
                "rerank_method": method,
            }
        )
    return evidence


def build_structured_prompt(question: str, evidence: list[dict]) -> str:
    context = "\n\n".join(
        f'chunk_id: {e["chunk_id"]}\nsection: {e["section"]}\npages: {e["pages"]}\ntext: {e["text"]}'
        for e in evidence
    )
    return f"{GROUNDING_SYSTEM_PROMPT}\n\nEvidence:\n{context}\n\nQuestion: {question}\nJSON answer:"


def _extractive_claims(question: str, evidence: list[dict]):
    """Deterministic fallback: one claim per evidence chunk, using the
    sentence with the most token overlap with the question."""
    q_tokens = set(tokenize(question))
    claims = []
    for e in evidence:
        sentences = re.split(r"(?<=[.!?])\s+", e["text"])
        best = max(sentences, key=lambda s: len(q_tokens & set(tokenize(s))), default=e["text"][:200])
        claims.append({"claim_text": best.strip(), "chunk_id": e["chunk_id"]})
    return claims


def _parse_structured_claims(raw_text: str):
    """Parses the model's JSON response into a claims list. Falls back to
    treating the whole response as a single unciteable claim if it isn't
    valid JSON — that becomes an 'unverified citation' downstream rather
    than silently dropping the answer."""
    try:
        cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        claims = parsed.get("claims", [])
        if claims:
            return claims
    except Exception:
        pass
    return [{"claim_text": raw_text.strip(), "chunk_id": None}]


def _first_page(pages_value) -> int:
    try:
        return int(str(pages_value).split(",")[0])
    except (ValueError, IndexError):
        return -1


# ---------------------------------------------------------------------------
# Response builders for the three contract statuses
# ---------------------------------------------------------------------------
def _insufficient_evidence_response(question: str, confidence: float) -> dict:
    return {
        "status": "insufficient_evidence",
        "question": question,
        "answer_summary": "I don't have enough evidence in this guideline to answer that.",
        "claims": [],
        "citation_coverage": 0.0,
        "unverified_citations": [],
        "confidence": round(float(confidence), 3),
        "confidence_label": confidence_label(confidence),
        "backend": get_llm_backend()[0],
    }


def _patient_specific_refusal_response(question: str) -> dict:
    return {
        "status": "patient_specific_refusal",
        "question": question,
        "answer_summary": (
            "I can't give advice about your individual results, risk, or next steps. "
            "This guideline describes population-level clinical recommendations, not a "
            "substitute for a clinician who knows your history. Please discuss this with "
            "your doctor."
        ),
        "claims": [],
        "citation_coverage": 0.0,
        "unverified_citations": [],
        "confidence": None,
        "confidence_label": "n/a",
        "backend": get_llm_backend()[0],
    }


# ---------------------------------------------------------------------------
# THE FROZEN CONTRACT
# ---------------------------------------------------------------------------
def answer_question(question: str, k: int = config.TOP_K) -> dict:
    """Always returns the shape documented at the top of this file, no
    matter what changes underneath in ingestion / indexing / retrieval."""
    if is_patient_specific_query(question):
        return _patient_specific_refusal_response(question)

    _ensure_index_loaded()
    backend_name, llm = get_llm_backend()

    reranked = retrieve(_DB, _BM25, _CHUNKS, question, k=k)
    confidence = retrieval_confidence(reranked)

    if not reranked or confidence < config.CONFIDENCE_THRESHOLD:
        return _insufficient_evidence_response(question, confidence)

    evidence = build_citation_evidence(reranked, top_n=k)
    evidence_lookup = {e["chunk_id"]: e for e in evidence}

    if backend_name == "extractive":
        claims = _extractive_claims(question, evidence)
    else:
        prompt = build_structured_prompt(question, evidence)
        try:
            if backend_name == "groq":
                raw = llm.invoke(prompt, json_mode=True)
                raw_text = raw
            elif backend_name == "openai":
                raw_text = llm.invoke(prompt).content
            else:
                raw_text = llm.invoke(prompt)
            claims = _parse_structured_claims(raw_text)
        except Exception as e:  # pragma: no cover
            claims = _extractive_claims(question, evidence)
            print(f"LLM call failed ({e}); used extractive fallback instead")

    resolved_claims, unverified = [], []
    for c in claims:
        cid = c.get("chunk_id")
        ev = evidence_lookup.get(cid)
        if ev is None:
            if cid:
                unverified.append(cid)
            resolved_claims.append({"claim_text": c["claim_text"], "citation": None})
        else:
            resolved_claims.append(
                {
                    "claim_text": c["claim_text"],
                    "citation": {
                        "document_id": ev["document_id"],
                        "section": ev["section"],
                        "page": _first_page(ev["pages"]),
                        "chunk_id": ev["chunk_id"],
                    },
                }
            )

    coverage = (
        round(sum(1 for c in resolved_claims if c["citation"]) / len(resolved_claims), 2)
        if resolved_claims
        else 0.0
    )

    return {
        "status": "answered",
        "question": question,
        "answer_summary": " ".join(c["claim_text"] for c in resolved_claims),
        "claims": resolved_claims,
        "citation_coverage": coverage,
        "unverified_citations": unverified,
        "confidence": round(float(confidence), 3),
        "confidence_label": confidence_label(confidence),
        "backend": backend_name,
    }
