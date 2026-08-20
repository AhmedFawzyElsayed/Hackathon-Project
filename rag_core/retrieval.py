"""
rag_core.retrieval — retrieve(), lifted from the notebook's Sections 6-8
(clinical abbreviation expansion, hybrid vector+BM25 retrieval fused with
RRF, cross-encoder reranking) plus Section 15 (retrieval confidence).

Section 7 migration (Bugs 1/4/6): query-type-aware retrieval.
Section 8 migration: raw ce_score reporting + glossary injection/promotion.
Section 15 migration (Bug 3): raw top-1 CE confidence + glossary floor.
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


# ---------------------------------------------------------------------------
# Query-type classification (notebook Section 7 — Bugs 4 & 6)
# ---------------------------------------------------------------------------
def is_definitional_query(question: str) -> bool:
    """True for 'what does X stand for' / 'abbreviation for' / 'X short for' /
    'X abbreviated as' style questions, answered by the glossary/abbreviations
    list (Front matter, p.11), not by the clinical recommendation sections.
    Bug 6 fix: broadened to catch 'short for', 'the abbreviation for', and
    'abbreviated as' phrasings the old regex missed."""
    q = question.lower()
    return bool(
        re.search(
            r"stand(s)?\s+for|abbreviation|abbreviated|what does .* mean|"
            r"what is .+ short for|what is the abbreviation for|"
            r"refer\s+to|referred\s+to\s+as|refers?\s+to",
            q,
        )
    )


def extract_abbreviation_token(question: str):
    """Pulls the abbreviation token out of a definitional question. The
    character class keeps hyphens/slashes so PI-RADS / QLQ-C30-style tokens
    survive as a single token."""
    q_clean = re.sub(r"[?.:]", "", question.strip())
    for pat in [
        r"abbreviated\s+as\s+([A-Za-z0-9][A-Za-z0-9\-/]*)\b",
        r"abbreviation\s+for\s+([A-Za-z0-9][A-Za-z0-9\-/]*)\b",
        r"what does\s+([A-Za-z0-9][A-Za-z0-9\-/]*)\b",
        r"what is\s+([A-Za-z0-9][A-Za-z0-9\-/]*)\b",
    ]:
        m = re.search(pat, q_clean, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None


def is_threshold_query(question: str) -> bool:
    """True for questions asking for a specific number / cut-off / age /
    interval from the guideline."""
    q = question.lower()
    if re.search(
        r"\d+(?:\.\d+)?\s*(?:µg|ug|ng|ng/ml|µg/l|ug/l|ml|years?|months?|weeks?|%|per|micrograms per litre)",
        q,
    ):
        return True
    return bool(
        re.search(
            r"\b(threshold|cutoff|cut-off|interval|how many|how soon|how frequently|how often|"
            r"at what age|what age|what psa value|what psa level|what psa density|what percentage|"
            r"how much|how many times|what proportion|what is the maximum|what maximum|what clinical stages|"
            r"how many cores|how many samples)\b",
            q,
        )
    )


def classify_query_type(question: str) -> str:
    """Architecture item: query-type-aware retrieval and answer generation."""
    if is_definitional_query(question):
        return "definitional"
    if is_threshold_query(question):
        return "threshold"
    return "clinical"


def section_boost(section: str, definitional: bool = False) -> int:
    if definitional and section == "Front matter":
        # Kept as a small additive signal; the 6x multiplier below is the real
        # definitional boost.
        return 2
    if section in config.PRIORITY_SECTIONS:
        return 1
    if section in config.DEPRIORITY_SECTIONS:
        return -1
    return 0


def _glossary_entry_match(text: str, abbr: str) -> bool:
    """True if `abbr` appears as a glossary-style entry: abbreviation at line
    start followed by a capitalised definition on the same line ('PSA Prostate
    specific antigen'), or with the definition starting on the next line
    ('MCID\\nMinimal clinically important difference'). Matching the entry
    shape — not mere token presence — stops promotion from firing on front-
    matter prose that merely happens to contain the token (e.g. title-page
    'PSA testing')."""
    pat = re.compile(rf"(?m)^\s*{re.escape(abbr)}\b", re.IGNORECASE)
    for m in pat.finditer(text):
        after = text[m.end() :].lstrip()
        if after and after[0].isupper():
            return True
    return False


def find_glossary_chunks(abbr_token: str, all_chunks: list[Document]):
    """Front-matter chunks that contain a glossary entry for `abbr_token` (the
    authoritative p.11 abbreviations list — the answer source for definitional
    queries)."""
    if not abbr_token:
        return []
    return [
        c
        for c in all_chunks
        if c.metadata.get("section") == "Front matter"
        and _glossary_entry_match(c.page_content, abbr_token)
    ]


# --- content-support guard (used by the confidence / abstention logic) ---
_CONTENT_STOPWORDS = set(
    """
what does do did is are was were the a an of in to for and or with on at be by from this that it as
about how when where which who whom whose whats according within should would could can may might will
not no nor but if then than so such only just very too also i me my im ive we our you your there their
its them they recommended recommend recommendation recommendations treatment treatments treated manage
managing managed management test tests testing tested use using used screening screened
""".split()
)


def _content_tokens(question: str):
    """Distinctive (non-scaffolding) tokens of a question. Hyphens are kept so
    'PI-RADS' stays one token and can match the chunk text directly."""
    toks = re.findall(r"[a-z0-9\-/]+", question.lower())
    return [t for t in toks if t not in _CONTENT_STOPWORDS and len(t) >= 3]


def has_content_support(question: str, reranked_chunks) -> bool:
    """The raw cross-encoder score can be high for *topic-adjacent* out-of-
    scope questions. If the retrieved top-3 chunks share NO distinctive content
    token with the question, the top hit is adjacent-but-unanswerable, so we
    refuse. Skipped for glossary-hit definitional questions, which are
    validated by the authoritative-glossary confidence floor instead."""
    tokens = _content_tokens(question)
    if not tokens:
        return True
    ev_tokens = set()
    for doc, _, _ in reranked_chunks[:3]:
        ev_tokens |= set(re.findall(r"[a-z0-9\-/]+", doc.page_content.lower()))
    return any(t in ev_tokens for t in tokens)


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
    definitional = is_definitional_query(question)

    # FIX (Bug 4): definitional queries skip BM25 expansion. Appending
    # "prostate specific antigen density psa density" etc. made keyword
    # retrieval prefer p.103/p.109 (biopsy triage / PICO) over the p.11
    # glossary — the opposite of what the question needs.
    expanded = question if definitional else expand_query(question)

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

    abbr_token = extract_abbreviation_token(question) if definitional else None
    glossary_cids = (
        {c.metadata["chunk_id"] for c in find_glossary_chunks(abbr_token, chunks)} if abbr_token else set()
    )

    for cid, doc in doc_lookup.items():
        if cid in glossary_cids:
            # Bug 1 fix: 6x multiplier — a REAL boost for the authoritative
            # glossary chunks (the old +0.02 additive term could never outrank
            # the body pages mentioning the abbreviation).
            rrf_scores[cid] *= config.DEFINITIONAL_FRONT_MATTER_BOOST
        else:
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
        # Regression fix (Section 8): min-max normalization always maps whoever
        # scored highest to exactly 1.0 — on-topic or not. retrieval_confidence
        # needs the RAW cross-encoder score near/below 0 for irrelevant top
        # hits to trigger abstention. Use `blended` only for ordering; report
        # the untouched raw ce_score as the tuple's score.
        order = sorted(range(len(docs)), key=lambda i: blended[i], reverse=True)
        return [(docs[i], float(ce_scores[i]), "cross_encoder_blended") for i in order[:top_n]]

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


def retrieve_and_rerank(db, bm25, chunks: list[Document], question: str, k: int = config.TOP_K, candidate_pool: int = config.CANDIDATE_POOL):
    """Full pipeline: expand -> hybrid fuse -> rerank, with the Bug 1 glossary
    injection/promotion for definitional queries. Always pulls the full
    candidate_pool out of hybrid_retrieve and lets the reranker (not RRF)
    decide the final top k."""
    candidates = hybrid_retrieve(db, bm25, chunks, question, k=candidate_pool, pool=candidate_pool)

    # Bug 1 fix (hard reorder): for definitional queries the authoritative
    # glossary chunk on p.11 is often OUTSIDE the fused top-50 pool. Inject
    # every glossary chunk containing the abbreviation into the rerank pool,
    # rerank the enlarged pool, then PROMOTE the injected glossary chunks to
    # the top of the final ranking so the answer always comes from the
    # authoritative list on p.11 (not the p.103 biopsy-triage or p.109 PICO
    # tables).
    definitional = is_definitional_query(question)
    abbr_token = extract_abbreviation_token(question) if definitional else None
    injected_ids: list[str] = []
    if abbr_token:
        pool_ids = {d.metadata["chunk_id"] for d, _ in candidates}
        for c in find_glossary_chunks(abbr_token, chunks):
            if c.metadata["chunk_id"] not in pool_ids:
                candidates.append((c, 0.0))
                pool_ids.add(c.metadata["chunk_id"])
            injected_ids.append(c.metadata["chunk_id"])
    # When glossary chunks were injected the candidate list is bigger than the
    # bare pool, so top_n must cover the whole enlarged list or an injected
    # glossary chunk could be truncated before the hard promotion below gets a
    # chance to surface it.
    rr = rerank(question, candidates, top_n=len(candidates) if injected_ids else max(k, candidate_pool))
    if injected_ids:
        promoted = [r for r in rr if r[0].metadata["chunk_id"] in injected_ids]
        rest = [r for r in rr if r[0].metadata["chunk_id"] not in injected_ids]
        rr = (promoted + rest)[:k]
    else:
        rr = rr[:k]
    return rr


# `retrieve` is the name rag_core.generation and the README import; it now
# carries the notebook's full retrieve_and_rerank behaviour.
retrieve = retrieve_and_rerank


# ---------------------------------------------------------------------------
# Retrieval confidence (drives the insufficient_evidence refusal)
# ---------------------------------------------------------------------------
def retrieval_confidence(reranked_chunks, glossary_hit: bool = False) -> float:
    """FIX (Bug 3): the old version returned min(top_score, gap_z). Because the
    pool's min-max normalization always maps *some* candidate to the top, gap_z
    (a relative rank-gap statistic) was high even for clearly off-topic
    questions, and min() could clamp a genuinely strong raw score down to a
    weak gap-based value. Abstention never triggered because
    CONFIDENCE_THRESHOLD was 0.0.

    New version anchors on the cross-encoder's RAW top-1 relevance logit
    (rerank() reports the untouched ce_score as its tuple score): in-scope
    answers score ~7-16, adjacent-but-out-of-scope topics ~1.2-9.2, and
    unrelated out-of-scope topics go negative. Calibrated
    CONFIDENCE_THRESHOLD = 6.5 on the chosen custom_900_175 chunk config.

    glossary_hit=True: the top chunk IS the authoritative abbreviations-list
    entry (a table the cross-encoder scores poorly — 'CI' scores -4.4 despite
    being correct). A found glossary entry is ground truth, so the confidence
    gets a floor.
    """
    if not reranked_chunks:
        return -999.0

    scores = [s for _, s, _ in reranked_chunks]
    top_score = scores[0]
    top_method = reranked_chunks[0][2]

    if glossary_hit:
        return max(float(top_score), config.GLOSSARY_HIT_CONFIDENCE_FLOOR)

    if len(scores) < 2:
        gap_z = 1.0
    else:
        rest = scores[1:]
        gap_z = (top_score - np.mean(rest)) / (np.std(rest) + 1e-9)

    if top_method in ("cross_encoder", "cross_encoder_blended"):
        return float(top_score)
    return float(gap_z)


def glossary_hit_for(question: str, reranked_chunks) -> bool:
    """True when the top reranked chunk is the authoritative glossary entry for
    the question's abbreviation (i.e. the definitional promotion in
    retrieve_and_rerank fired)."""
    if not is_definitional_query(question) or not reranked_chunks:
        return False
    abbr = extract_abbreviation_token(question)
    if not abbr:
        return False
    return _glossary_entry_match(reranked_chunks[0][0].page_content, abbr)