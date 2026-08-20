# ByteCode — Clinical RAG Assistant for the 2026 Prostate Cancer Guidelines

A grounded, cited question-answering assistant over the *2026 Guidelines for
the Early Detection of Prostate Cancer in Australia* (PCFA). It retrieves
evidence with hybrid search (dense + BM25, RRF-fused), reranks it with a
biomedical cross-encoder, and generates answers that are traceable to exact
pages of the guideline — while refusing to guess when the evidence is weak,
and refusing to give patient-specific medical advice at all.

> ⚠️ **ByteCode is a research/demo system, not medical software.** It answers
> from one population-level clinical guideline. Always verify information with
> a clinician.

---

## Problem

General-purpose LLM chatbots hallucinate. Asked a medical question they
produce fluent-sounding but unverifiable answers, and they will happily
"interpret" a personal PSA result or biopsy risk — advice they have no
business giving. For a clinical-guideline Q&A tool, two things matter more
than raw fluency:

1. **Every claim must be traceable** to the exact guideline page it came from.
2. **The system must refuse** — when the evidence doesn't support an answer,
   and when the question is about a specific patient's own results or next steps.

This project builds both guarantees into a RAG pipeline, then measures them.

---

## Solution

The assistant is a retrieval-augmented generation (RAG) pipeline over the
236-page guideline:

```
question
   │
   ▼
is_patient_specific_query()? ──yes──► status = patient_specific_refusal
   │                                     (no retrieval, no generation)
   ▼
hybrid retrieve   (BGE-M3 dense + BM25, reciprocal-rank fusion, 50-candidate pool)
   │
   ▼
cross-encoder rerank (ncbi/MedCPT-Cross-Encoder), blended with the fused score
   │
   ▼
confidence gate  (raw CE logit ≥ 6.5 AND topical content support,
   │              glossary-hit answers exempt with a confidence floor)
   │
   ▼
grounded generation (Groq / OpenAI / Ollama, or a deterministic extractive fallback)
   │
   ▼
status = answered, with claim → chunk_id → page citations + quality metrics
```

Every response carries one of three statuses:

| Status | Meaning |
|--------|---------|
| `answered` | Evidence supported an answer; claims cite exact pages (`chunk_id`, section, page). |
| `insufficient_evidence` | Retrieval or confidence gating decided the guideline doesn't support an answer — the system declines to guess. |
| `patient_specific_refusal` | The question asks for advice on a specific patient's results/risk/next steps — refused before any retrieval runs. |

---

## Key features

- **Hybrid retrieval** — dense (BGE-M3) + BM25 keyword search fused with
  reciprocal-rank fusion, so semantically- and lexically-relevant chunks both
  surface.
- **Biomedical reranking** — `ncbi/MedCPT-Cross-Encoder` re-orders the fusion
  pool; the final order blends cross-encoder and fusion scores
  (`RERANK_BLEND_WEIGHT`).
- **Confidence gating** — a calibrated raw-logit threshold (6.5 on the MedCPT
  scale) plus a topical content-support check prevent weak answers from being
  generated. Definitional/glossary answers get a confidence floor because the
  abbreviations table legitimately scores low.
- **Patient-safety gate** — first-person + clinical-marker + personal-judgment
  pattern matching (plus explicit phrasings like "should I") catches
  patient-specific questions before retrieval.
- **Cited, structured answers** — every claim resolves to one `chunk_id`
  → section → page; unverifiable citations are surfaced explicitly.
- **Deterministic quality metrics** — faithfulness, answer relevance,
  hallucination rate, and context utilization are computed per answer with
  token overlap (no extra LLM calls, no added latency).
- **Degraded-mode visibility** — if the GPU models can't load, the pipeline
  falls back (MiniLM reranker / TF-IDF embeddings) and reports
  `reranker_degraded` / `embedding_degraded` in every response rather than
  failing silently.
- **Conversation memory** — backend memory layer threads history and
  long-term context into subsequent turns.
- **Evaluation harness** — a 200-question eval set (Direct, Paraphrased,
  Abbreviation, Threshold, Out-of-scope) plus Precision/Recall/MRR/nDCG
  metrics, chunk-config comparison, and stage-by-stage retrieval diagnostics
  live in the companion notebook.

---

## Architecture

```
notebook (keeps changing)  →  rag_core/ (stable interface)  →  backend  →  frontend
```

`rag_core/` is the seam: the notebook is free to keep changing chunking,
retrieval, and reranking, but `rag_core.answer_question(question)` always
returns the same shape, so the backend and frontend never change when the
pipeline underneath does.

```
rag_core/            stable interface — ingestion, indexing, retrieval, generation
├── config.py          paths, model names, thresholds (one place to tune)
├── ingestion.py       load_and_chunk_pdf()
├── indexing.py        build_or_load_index() / load_index()  (Chroma + BM25)
├── retrieval.py       retrieve()  (expansion, hybrid fusion, rerank, confidence)
├── generation.py      answer_question()  ← the frozen contract
├── metrics.py         deterministic faithfulness / relevance / hallucination metrics
└── build_index.py     CLI: (re)build persisted index artifacts

backend/              FastAPI — POST /ask, GET /health
├── app/api/routes/chat.py
├── app/services/     rag_service.py, memory.py
└── tests/            pytest suite (mocked, offline, fast)

frontend/             React + TypeScript + Vite — chat UI with cited claims
└── src/components/   AnswerCard, RefusalCard, MetricsBar, ChatInput
```

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Corpus | 2026 PCFA Early Detection of Prostate Cancer guidelines (236 pp.) |
| Embeddings | BAAI/bge-m3 (dense, 1024-d) |
| Sparse index | BM25Okapi (`rank_bm25`) |
| Vector store | ChromaDB (via langchain-chroma) |
| Reranker | ncbi/MedCPT-Cross-Encoder (fallbacks: MiniLM, TF-IDF) |
| Generation | Groq (`openai/gpt-oss-120b`) / OpenAI (`gpt-4o-mini`) / Ollama / extractive |
| Backend | Python, FastAPI, Pydantic |
| Frontend | React 18, TypeScript, Vite |
| GPU | CUDA (RTX-class) for embeddings + reranking |

---

## Evaluation (controlled experiments)

Retrieval and safety were validated on the notebook's 200-question eval set
(180 in-scope questions) with a two-stage controlled experiment:

- **Stage 1 — rerank blend weight.** Swept `RERANK_BLEND_WEIGHT` on the
  canonical chunk config. Best = **0.325** (MRR 0.533, nDCG@5 0.479,
  Hit@10 151/180), recovering 4 questions the reranker had mis-ranked with
  zero regressions.
- **Section 15 — trust test.** At the winning weight, out-of-scope questions
  abstained 20/20 and in-scope questions had **0 false refusals**.
- **Stage 2 — chunking.** No alternative chunk size robustly beat the
  canonical **850/150** config without breaking confidence-gate behaviour, so
  the canonical chunking was retained.

Final configuration: `RERANK_BLEND_WEIGHT = 0.325`, chunk size **850**,
overlap **150**. The same experiment also confirmed the confidence threshold
(6.5) is only valid on the MedCPT raw-logit scale, which is why the fallback
rerankers are gated on their own scales.

---

## Getting started

Requires Python 3.11+, Node 18+, a GPU (optional — CPU works with fallbacks),
and the source PDF at the project root (or `GUIDELINE_PDF_PATH` in `.env`).

```bash
# 1. Python environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend/requirements.txt

# 2. Environment variables
cp .env.example .env             # fill in GROQ_API_KEY (or OPENAI_API_KEY)
cp backend/.env.example backend/.env

# 3. Build the index (once, and again after any chunking change)
python -m rag_core.build_index

# 4. Backend
cd backend
uvicorn app.main:app --reload    # http://localhost:8000/docs

# 5. Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env             # VITE_API_BASE_URL=http://localhost:8000
npm run dev                      # http://localhost:5173
```

### API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness check. |
| `POST /ask`  | `{"question": "...", "conversation_id": "..."}` → answer with claims, citations, confidence label, and metrics. |

### Tests

```bash
cd backend
pytest
```

The tests mock `rag_core.answer_question()` so they run fast and offline —
one supported-question case, one patient-specific refusal case, and input
validation.

---

## Handling "the notebook keeps changing"

- The notebook stays your scratch space for trying new chunking, retrieval,
  or generation ideas.
- When something works, port that logic into the matching `rag_core/` file
  and confirm `answer_question()` still returns the same shape. Add a dated
  line to `rag_core/CHANGELOG.md`.
- If a change *needs* to alter the shape (new field), update
  `backend/app/schemas/chat.py` and `frontend/src/types/chat.ts` in the same
  commit.
- If chunking or ingestion changes, rerun `python -m rag_core.build_index`
  before restarting the backend.

---

## Security note

The original notebook had a live Groq API key hardcoded in plain text.
**Nothing in this repo hardcodes a key** — everything reads
`GROQ_API_KEY` from the environment (`.env`, not committed to git).
If you ever suspect a key was exposed, revoke it in your Groq console and
generate a new one. Never commit a real `.env`.

---

## Project team

Built by a five-person team, each owning a layer of the stack:

1. **RAG & Retrieval Engineer** — chunking, dense/BM25 indexes, RRF fusion, hybrid retrieval (`rag_core/ingestion.py`, `indexing.py`, `retrieval.py`).
2. **Reranking & Model Engineer** — MedCPT cross-encoder integration, blend-weight tuning, confidence gating (`rag_core/generation.py`, `config.py`).
3. **Evaluation / ML Engineer** — eval question set, metrics, controlled experiments, Section-15 trust test (notebook, `rag_core/metrics.py`, `run_chunking_comparison.py`).
4. **Backend / Application Engineer** — FastAPI endpoints, memory layer, orchestration (`backend/`).
5. **MLOps / QA & Compliance** — environment, reproducibility, tests, deployment, trust-test governance.