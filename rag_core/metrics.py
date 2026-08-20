"""
rag_core.metrics — deterministic generation-quality and hallucination metrics.

All metrics are token-overlap based (no extra LLM calls) so they add zero
latency and no API-key dependency.  Computed at the end of answer_question()
and returned alongside the standard contract dict.

Metrics
-------
faithfulness        Max Jaccard similarity per claim across evidence chunks,
                    averaged.  Measures claim-to-evidence grounding.
answer_relevance    Overlap coefficient: |question_terms ∩ answer_terms| /
                    |question_terms|.  Measures whether the answer addresses
                    the question.
hallucination_rate  Fraction of claims with citation=None.  0 = no
                    hallucination, 1 = every claim is uncited.
context_utilization Cited evidence chunks / total evidence chunks provided.
                    Measures whether retrieval results were actually used.
"""
import re


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation — matches rag_core.indexing.tokenize."""
    return re.findall(r"[\w.]+", text.lower())


def compute_faithfulness(claims: list[dict], evidence: list[dict]) -> float:
    """For each claim, find the maximum Jaccard similarity between its
    tokens and any evidence chunk's tokens.  Average across claims."""
    if not claims or not evidence:
        return 0.0
    ev_token_sets = [set(_tokenize(e["text"])) for e in evidence]
    scores: list[float] = []
    for claim in claims:
        claim_tokens = set(_tokenize(claim.get("claim_text", "")))
        if not claim_tokens:
            scores.append(0.0)
            continue
        best = 0.0
        for ev_tokens in ev_token_sets:
            if not ev_tokens:
                continue
            inter = len(claim_tokens & ev_tokens)
            union = len(claim_tokens | ev_tokens)
            if union:
                best = max(best, inter / union)
        scores.append(best)
    return round(sum(scores) / len(scores), 4)


def compute_answer_relevance(question: str, answer_summary: str) -> float:
    """Overlap coefficient: what fraction of the question's terms appear
    in the answer."""
    q_tokens = set(_tokenize(question))
    a_tokens = set(_tokenize(answer_summary))
    if not q_tokens:
        return 0.0
    return round(len(q_tokens & a_tokens) / len(q_tokens), 4)


def compute_hallucination_rate(claims: list[dict]) -> float:
    """Fraction of claims without a citation (citation is None)."""
    if not claims:
        return 0.0
    uncited = sum(1 for c in claims if c.get("citation") is None)
    return round(uncited / len(claims), 4)


def compute_context_utilization(claims: list[dict], evidence: list[dict]) -> float:
    """Fraction of evidence chunks that were cited by at least one claim."""
    if not evidence:
        return 0.0
    cited_ids: set[str] = set()
    for c in claims:
        cit = c.get("citation")
        if cit and cit.get("chunk_id"):
            cited_ids.add(cit["chunk_id"])
    ev_ids = {e["chunk_id"] for e in evidence}
    if not ev_ids:
        return 0.0
    return round(len(cited_ids & ev_ids) / len(ev_ids), 4)


def compute_metrics(
    question: str, result: dict, evidence: list[dict]
) -> dict:
    """Compute all generation / hallucination metrics for a single response.
    Called at the end of answer_question(), before returning."""
    claims = result.get("claims", [])
    return {
        "faithfulness": compute_faithfulness(claims, evidence),
        "answer_relevance": compute_answer_relevance(
            question, result.get("answer_summary", "")
        ),
        "hallucination_rate": compute_hallucination_rate(claims),
        "context_utilization": compute_context_utilization(claims, evidence),
    }
