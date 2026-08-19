"""
rag_core.retrieval — retrieve(), lifted from the notebook's Sections 6-8
(clinical abbreviation expansion, hybrid vector+BM25 retrieval fused with
RRF, cross-encoder reranking) plus Section 15 (retrieval confidence).
"""
import re

import numpy as np
from langchain_core.documents import Document

from . import config
from .indexing import tokenize

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
def expand_query(question: str) -> str:
    q_lower = question.lower()
    extra_terms = [
        expansion
        for abbr, expansion in config.ABBREVIATIONS.items()
        if re.search(rf"\b{re.escape(abbr)}\b", q_lower)
    ]
    return question + " " + " ".join(extra_terms) if extra_terms else question


def is_definitional_query(question: str) -> bool:
    """True for 'what does X stand for' style questions, answered by the
    glossary/abbreviations list (Front matter), not the clinical sections."""
    q = question.lower()
    return bool(re.search(r"stand(s)? for|abbreviation|what does .* mean", q))


def section_boost(section: str, definitional: bool = False) -> int:
    if definitional and section == "Front matter":
        return 2
    if section in config.PRIORITY_SECTIONS:
        return 1
    if section in config.DEPRIORITY_SECTIONS:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Hybrid retrieval (dense + BM25, fused with Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
def hybrid_retrieve(
    db,
    bm25,
    chunks: list[Document],
    question: str,
    k: int = 10,
    pool: int = 50,
    rrf_k: int = config.RRF_K,
    boost_weight: float = config.BOOST_WEIGHT,
    bm25_weight: float = config.BM25_WEIGHT,
    dense_weight: float = config.DENSE_WEIGHT,
):
    # bm25_weight > dense_weight leans the fusion toward the stronger single
    # signal on this corpus (clinical guideline text is dense with exact
    # terms/numbers BM25 is naturally good at) without dropping dense to
    # zero — dense still catches paraphrased questions BM25 alone misses.
    expanded = expand_query(question)
    definitional = is_definitional_query(question)

    # Dense retrieval gets the original question, not the expanded one —
    # appending raw keyword expansions distorts the sentence embedding.
    # BM25 is a keyword matcher, so it's the one that benefits from `expanded`.
    semantic_results = db.similarity_search_with_relevance_scores(question, k=pool)

    bm25_scores = bm25.get_scores(tokenize(expanded))
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool]

    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(semantic_results, start=1):
        cid = doc.metadata["chunk_id"]
        doc_lookup[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight / (rrf_k + rank)

    for rank, idx in enumerate(bm25_ranked_idx, start=1):
        doc = chunks[idx]
        cid = doc.metadata["chunk_id"]
        doc_lookup[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_weight / (rrf_k + rank)

    for cid, doc in doc_lookup.items():
        rrf_scores[cid] += boost_weight * section_boost(doc.metadata.get("section", ""), definitional)

    # Every fused candidate is kept (no per-page dedup) so the reranker sees
    # the full pool — a page's second-best chunk shouldn't be deleted before
    # the much more accurate cross-encoder gets a chance to score it.
    ranked_ids = sorted(rrf_scores.keys(), key=rrf_scores.get, reverse=True)[:k]
    return [(doc_lookup[cid], rrf_scores[cid]) for cid in ranked_ids]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
class _MedCPTCrossEncoderAdapter:
    """Wraps ncbi/MedCPT-Cross-Encoder behind the same .predict(pairs) ->
    scores interface sentence-transformers' CrossEncoder exposes."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.to(config.get_device())

    def predict(self, pairs):
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                [[q, d] for q, d in pairs],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(config.get_device())
            logits = self.model(**encoded).logits.squeeze(dim=1)
        return logits.cpu().numpy()


_cross_encoder = None
_cross_encoder_checked = False


def get_cross_encoder():
    global _cross_encoder, _cross_encoder_checked
    if _cross_encoder_checked:
        return _cross_encoder
    _cross_encoder_checked = True
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.CROSS_ENCODER_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(config.CROSS_ENCODER_MODEL_NAME).eval()
        _cross_encoder = _MedCPTCrossEncoderAdapter(tok, model)
        print(f"Reranker: using {config.CROSS_ENCODER_MODEL_NAME} on {config.get_device()}")
    except Exception as e:  # pragma: no cover
        print(f"Reranker: MedCPT-Cross-Encoder unavailable ({e}); trying general cross-encoder")
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder = CrossEncoder(config.FALLBACK_CROSS_ENCODER_MODEL_NAME, device=config.get_device())
            print(f"Reranker: using {config.FALLBACK_CROSS_ENCODER_MODEL_NAME} (fallback) on {config.get_device()}")
        except Exception as e2:  # pragma: no cover
            print(f"Reranker: cross-encoder unavailable ({e2}); using TF-IDF fallback blend")
            _cross_encoder = None
    return _cross_encoder


def _minmax(values):
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / span for v in values]


def rerank(question: str, candidates, top_n: int = 10):
    """candidates: list of (doc, fused_rrf_score). Returns list of (doc, score, method)."""
    if not candidates:
        return []

    ce = get_cross_encoder()
    docs = [doc for doc, _ in candidates]

    if ce is not None:
        # Blend the cross-encoder score with the fused RRF score (still
        # dominant, but not the only vote) so section_boost / definitional
        # boosting a chunk earned in fusion isn't thrown away completely.
        pairs = [(question, d.page_content) for d in docs]
        ce_scores = ce.predict(pairs)
        fused_scores = [s for _, s in candidates]

        ce_norm = _minmax(list(ce_scores))
        fused_norm = _minmax(fused_scores)

        weight = config.RERANK_BLEND_WEIGHT
        blended = [(1 - weight) * c + weight * f for c, f in zip(ce_norm, fused_norm)]
        order = sorted(range(len(docs)), key=lambda i: blended[i], reverse=True)
        return [(docs[i], float(blended[i]), "cross_encoder_blended") for i in order[:top_n]]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [d.page_content for d in docs]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = tfidf.fit_transform([question] + texts)
    keyword_scores = cosine_similarity(X[0:1], X[1:]).ravel()

    fused_scores = [s for _, s in candidates]
    max_fused = max(fused_scores) if max(fused_scores) > 0 else 1.0
    blended = [0.5 * (fs / max_fused) + 0.5 * float(ks) for fs, ks in zip(fused_scores, keyword_scores)]
    order = sorted(range(len(docs)), key=lambda i: blended[i], reverse=True)
    return [(docs[i], float(blended[i]), "tfidf_fallback") for i in order[:top_n]]


def retrieve(db, bm25, chunks: list[Document], question: str, k: int = config.TOP_K, candidate_pool: int = config.CANDIDATE_POOL):
    """Full pipeline: expand -> hybrid fuse -> rerank. Always pulls the full
    candidate_pool out of hybrid_retrieve and lets the reranker (not RRF)
    decide the final top k."""
    candidates = hybrid_retrieve(db, bm25, chunks, question, k=candidate_pool, pool=candidate_pool)
    return rerank(question, candidates, top_n=k)


# ---------------------------------------------------------------------------
# Retrieval confidence (drives the insufficient_evidence refusal)
# ---------------------------------------------------------------------------
def retrieval_confidence(reranked_chunks) -> float:
    """Anchors on the cross-encoder's raw top-1 relevance score (question-to-
    chunk semantic relevance directly, not pool self-similarity) so an
    off-topic top hit scores low even if it beats its neighbors. The
    rank-gap statistic is kept as a secondary signal so a genuinely strong,
    well-separated match isn't penalized."""
    if not reranked_chunks:
        return -999.0

    scores = [s for _, s, _ in reranked_chunks]
    top_score = scores[0]
    top_method = reranked_chunks[0][2]

    if len(scores) < 2:
        gap_z = 1.0
    else:
        rest = scores[1:]
        gap_z = (top_score - np.mean(rest)) / (np.std(rest) + 1e-9)

    if top_method in ("cross_encoder", "cross_encoder_blended"):
        return min(top_score, gap_z)
    return gap_z
