# Clinical RAG Assistant

Grounded, cited Q&A over the *2026 Guidelines for the Early Detection of
Prostate Cancer in Australia* — hybrid retrieval (dense + BM25, RRF-fused),
cross-encoder reranking, and structured generation that refuses to guess
when the evidence doesn't support an answer, and refuses to give
patient-specific medical advice at all.

```
notebook (keeps changing)  →  rag_core/ (stable interface)  →  backend  →  frontend
```

`rag_core/` is the seam: the notebook is free to keep changing chunking,
retrieval, and reranking, but `rag_core.answer_question(question)` always
returns the same shape (see `rag_core/generation.py`), so the backend and
frontend never need to change when the pipeline underneath does.

## Architecture

```
rag_core/            stable interface — ingestion, indexing, retrieval, generation
├── config.py          paths, model names, thresholds
├── ingestion.py        load_and_chunk_pdf()
├── indexing.py           build_or_load_index() / load_index()  (Chroma + BM25)
├── retrieval.py            retrieve()  (expansion, hybrid fusion, rerank, confidence)
├── generation.py             answer_question()  ← the frozen contract
└── build_index.py              CLI: (re)build persisted index artifacts

backend/              FastAPI — POST /ask, GET /health
frontend/             React + TypeScript + Vite — chat UI with cited claims
```

## ⚠️ Before you do anything else

The original notebook had a live Groq API key hardcoded in plain text in one
of its cells. **Revoke that key in your Groq console and generate a new
one.** Nothing in this repo hardcodes a key — everything reads
`GROQ_API_KEY` from the environment (`.env`, not committed to git).

## Setup

Requires Python 3.11+, Node 18+, and the source PDF
(`2026-guidelines-for-the-early-detection-of-prostate-cancer.pdf`) placed at
the project root (or pointed to via `GUIDELINE_PDF_PATH` in `.env`).

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

## Tests

```bash
cd backend
pytest
```

The included tests mock `rag_core.answer_question()` so they run fast and
offline — one supported-question case, one patient-specific refusal case
(the guideline's strongest safety differentiator), and input validation.
For a real end-to-end check, build the index and hit `/ask` manually.

## Demo prep

See [`DEMO.md`](./DEMO.md) — three scripted questions pulled from the
notebook's own eval set (one clean citation trail, one out-of-scope
refusal, one patient-specific safety refusal), a pre-flight checklist, and
an honest "what would you do with more time" answer sourced from the
notebook's own end-of-day review.

## What's deliberately out of scope here

The evaluation harness in the notebook (Sections 9–16: eval question sets,
manual relevance review, Precision/MRR/nDCG metrics, chunk-config
comparison, stage-by-stage retrieval diagnostics) stays in the notebook —
it's how you *choose* `rag_core/config.py`'s `CHUNK_SIZE` / `CHUNK_OVERLAP`
and validate retrieval quality, not something the running app needs at
request time.
