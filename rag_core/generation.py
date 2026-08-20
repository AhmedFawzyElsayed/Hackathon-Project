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
from .metrics import compute_metrics
from .retrieval import (
    classify_query_type,
    extract_abbreviation_token,
    glossary_hit_for,
    has_content_support,
    retrieve,
    retrieval_confidence,
)

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
    "Answer the user's question directly and concisely. Use the evidence to answer the question, "
    "not to summarize the context.",
    "Ignore retrieved chunks that are unrelated or only loosely connected to the question.",
    "Use ONLY the provided evidence chunks. Do not use outside medical knowledge.",
    "Every claim must be traceable to exactly one evidence chunk_id provided in the context.",
    "If the evidence does not clearly support an answer, return status='insufficient_evidence' "
    "instead of guessing.",
    "Never answer a question that asks for advice about a specific patient's own results, risk, "
    "or next steps -- refuse and direct them to a clinician (handled before generation; see the "
    "patient-specific safety check above).",
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
    """Fixed version (Task 14). Keeps the v1 explicit phrasings, and adds a
    combined signal: first-person pronoun + a clinical-data term + a
    personal-judgment word, so phrasings like 'is that normal for me' are
    caught even without matching a fixed phrase list."""
    q = question.lower()
    explicit_patterns = [
        r"\bshould i\b",
        r"\bam i\b",
        r"\bmy psa\b",
        r"\bwhat should i do\b",
        r"\bdo i need\b",
        r"\bmy risk\b",
        r"\bmy result\b",
    ]
    if any(re.search(p, q) for p in explicit_patterns):
        return True

    has_first_person = bool(re.search(r"\b(i|i'm|im|my|me)\b", q))
    has_clinical_marker = bool(
        re.search(r"\b(psa|psad|dre|mri|mpmri|biopsy|risk|score|level|result|diagnosed|family history)\b", q)
    )
    has_personal_judgment = bool(re.search(r"\b(normal|for me|should|need|worried|ok|okay|concerned)\b", q))
    return has_first_person and has_clinical_marker and has_personal_judgment


def confidence_label(confidence) -> str:
    """FIX: absolute raw-CE bands instead of threshold-relative ones.
    MedCPT-Cross-Encoder logits: clinical answers ~14-16 (high), glossary/
    definitional answers ~5-10 and floored at 5.0 (medium/high),
    sub-threshold/refused answers low."""
    if confidence is None:
        return "n/a"
    if confidence >= 10.0:
        return "high"
    if confidence >= 5.0:
        return "medium"
    return "low"


# ---- FIX (Bug 8): strip PDF headers/footers/page artifacts before any text is used ----
# Every chunk carries the running footer "2026 Guidelines for the Early Detection
# of Prostate Cancer in Australia - Prostate Cancer Foundation of Australia" plus
# a standalone page number; feeding that noise to the LLM polluted the context and
# extractive sentence picking.
_DOC_TITLE_FOOTER = "2026 Guidelines for the Early Detection of Prostate Cancer in Australia"
_FOOTER_PATTERN = re.compile(r"\s*" + re.escape(_DOC_TITLE_FOOTER) + r".*$")


def clean_chunk_text(text: str) -> str:
    """Removes the running doc-title footer line and the standalone page number
    that follows it."""
    lines = text.splitlines()
    out = []
    footer_seen = False
    for ln in lines:
        s = ln.strip()
        if _FOOTER_PATTERN.search(ln):
            footer_seen = True
            continue
        if s.isdigit() and footer_seen:
            footer_seen = False
            continue
        footer_seen = False
        out.append(ln)
    return "\n".join(out)


def _page_list(pages) -> list[int]:
    return [int(p) for p in str(pages).split(",") if p.strip().isdigit()]


def extract_definition_answer(chunk_text: str, abbr_token: str):
    """Pulls the expansion text out of a glossary entry (handles 'ABBR Expansion'
    on one line and 'ABBR\\nExpansion' wrapped entries), returning a single clean
    sentence like 'PSAD stands for Prostate specific antigen density.' Also
    handles table rows written as 'Full Name (ABBR)' (e.g. the p.206 endorsement
    appendix)."""
    lines = [ln.strip() for ln in chunk_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        m = re.match(rf"^({re.escape(abbr_token)})([\s:–\-]+(.*))?$", ln, re.IGNORECASE)
        if not m:
            continue
        definition = (m.group(3) or "").strip().rstrip(".,;")
        if not definition:
            for nxt in lines[i + 1 : i + 3]:
                if re.match(r"^[A-Z0-9\-]{2,}\s+[A-Z]", nxt):
                    break
                definition += " " + nxt.rstrip(".,;")
        else:
            for nxt in lines[i + 1 : i + 3]:
                if re.match(r"^[A-Z0-9\-]{2,}\s+[A-Z]", nxt):
                    break
                definition += " " + nxt.rstrip(".,;")
        definition = re.sub(r"\s*\(Refer[^)]*\)\s*$", "", definition).strip()
        if definition:
            return f"{m.group(1).strip().upper()} stands for {definition}."
    # FIX (relevance): appendix tables write the expansion FIRST, then the token
    # in parentheses, e.g. "Royal Australian and New Zealand College of
    # Radiologists (RANZCR)". The line-start format above never matches those
    # rows, so those answers fell through to the generic sentence picker and
    # returned a heading fragment.
    m = re.search(
        rf"([A-Za-z][A-Za-z'\- ]+?)\s*\(\s*{re.escape(abbr_token)}\s*\)",
        chunk_text,
        re.IGNORECASE,
    )
    if m:
        expansion = re.sub(r"^\s*[•\-–|]+\s*", "", m.group(1)).strip()
        if expansion and not re.search(
            r"^\s*(appendix|table|page|contents|refer|organisation)\b", expansion, re.IGNORECASE
        ):
            return f"{abbr_token.upper()} stands for {expansion.rstrip('.')}."
    return None


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
                # FIX (Bug 8): no header/footer noise.
                "text": clean_chunk_text(doc.page_content).strip(),
                "rerank_score": round(float(score), 4),
                "rerank_method": method,
            }
        )
    return evidence


def build_structured_prompt(
    question: str,
    evidence: list[dict],
    history: str = "",
    long_term_context: str = "",
) -> str:
    sections = [GROUNDING_SYSTEM_PROMPT]
    if history:
        sections.append(f"\n{history}")
    if long_term_context:
        sections.append(
            f"\n{long_term_context}\n"
            "(Use prior findings for context only. Primary evidence must come from the chunks below.)"
        )
    context = "\n\n".join(
        f'chunk_id: {e["chunk_id"]}\nsection: {e["section"]}\npages: {e["pages"]}\ntext: {e["text"]}'
        for e in evidence
    )
    sections.append(f"\nEvidence:\n{context}")
    sections.append(f"\nQuestion: {question}\nJSON answer:")
    return "\n".join(sections)


# FIX (Bug 2): extractive_claims previously emitted one claim PER chunk with no
# relevance filter — five noisy chunks became five claims, half of them
# irrelevant. Now:
#   - definitional queries -> a single clean glossary claim;
#   - otherwise -> only chunks clearing the raw-CE relevance floor
#     (config.EVIDENCE_RELEVANCE_THRESHOLD), capped at top-3, each reduced to
#     its single best sentence on cleaned text.
def extractive_claims(question: str, evidence: list[dict]):
    q_tokens = set(tokenize(question))
    qtype = classify_query_type(question)

    if qtype == "definitional":
        abbr = extract_abbreviation_token(question)
        if abbr:
            for e in evidence:
                definition = extract_definition_answer(clean_chunk_text(e["text"]), abbr)
                if definition:
                    return [{"claim_text": definition, "chunk_id": e["chunk_id"]}]

    ranked = [e for e in evidence if e["rerank_score"] >= config.EVIDENCE_RELEVANCE_THRESHOLD]
    ranked = ranked[:3] if ranked else evidence[:3]

    claims = []
    for e in ranked:
        clean = clean_chunk_text(e["text"])
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if s.strip()]
        best = max(sentences, key=lambda s: len(q_tokens & set(tokenize(s))), default=clean[:200])
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
_NULL_METRICS = {
    "faithfulness": 0.0,
    "answer_relevance": 0.0,
    "hallucination_rate": 0.0,
    "context_utilization": 0.0,
}


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
        "metrics": _NULL_METRICS,
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
        "metrics": _NULL_METRICS,
    }


# ---------------------------------------------------------------------------
# THE FROZEN CONTRACT
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    k: int = config.TOP_K,
    history: str = "",
    long_term_context: str = "",
) -> dict:
    """Always returns the shape documented at the top of this file, no
    matter what changes underneath in ingestion / indexing / retrieval.

    *history* and *long_term_context* are optional pre-formatted strings
    supplied by the backend's memory layer.  They are injected into the LLM
    prompt but do not change the return shape."""
    if is_patient_specific_query(question):
        return _patient_specific_refusal_response(question)

    _ensure_index_loaded()
    backend_name, llm = get_llm_backend()

    reranked = retrieve(_DB, _BM25, _CHUNKS, question, k=k)
    glossary_hit = glossary_hit_for(question, reranked)
    confidence = retrieval_confidence(reranked, glossary_hit)
    content_ok = has_content_support(question, reranked)

    # Glossary-hit answers are exempt from BOTH guards: the found glossary entry
    # is authoritative ground truth, and its raw CE is low only because the
    # abbreviations table scores poorly — a floored confidence of 5.0 must never
    # trigger a refusal.
    if (
        not reranked
        or (not glossary_hit and confidence < config.CONFIDENCE_THRESHOLD)
        or (not glossary_hit and not content_ok)
    ):
        return _insufficient_evidence_response(question, confidence)

    evidence = build_citation_evidence(reranked, top_n=k)
    evidence_lookup = {e["chunk_id"]: e for e in evidence}

    if backend_name == "extractive":
        claims = extractive_claims(question, evidence)
    else:
        prompt = build_structured_prompt(question, evidence, history=history, long_term_context=long_term_context)
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
            claims = extractive_claims(question, evidence)
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

    # FIX (Bug 7): answer_summary used to be the raw concatenation of every
    # claim. For definitional/simple queries that was a wall of text. Pick the
    # single most relevant claim instead.
    if resolved_claims:
        q_tokens = set(tokenize(question))
        best_claim = max(resolved_claims, key=lambda c: len(q_tokens & set(tokenize(c["claim_text"]))))
    else:
        best_claim = None

    metrics = compute_metrics(question, {"claims": resolved_claims, "answer_summary": " ".join(c["claim_text"] for c in resolved_claims)}, evidence)

    return {
        "status": "answered",
        "question": question,
        "answer_summary": best_claim["claim_text"] if best_claim else "",
        "claims": resolved_claims,
        "citation_coverage": coverage,
        "unverified_citations": unverified,
        "confidence": round(float(confidence), 3),
        "confidence_label": confidence_label(confidence),
        "backend": backend_name,
        "metrics": metrics,
    }
