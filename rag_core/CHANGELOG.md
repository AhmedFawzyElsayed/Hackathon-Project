# rag_core CHANGELOG

One line per change, dated, whenever a notebook improvement gets copied into
`rag_core/`. Keeps a record of what changed without digging through notebook
diffs.

- 2026-08-19: Initial extraction from `Clinical_RAG_Day3_FIXED.ipynb` into
  `rag_core/`. Chunking config defaulted to `900/175` (chars/overlap) per the
  notebook's Section 14 comparison — rerun that comparison and update
  `config.CHUNK_SIZE` / `config.CHUNK_OVERLAP` if you change the source PDF.
  `generate_structured_answer()`'s `refused_patient_specific` status was
  renamed to `patient_specific_refusal` to match the frozen `answer_question()`
  contract; `pages` (comma-joined string) is exposed to callers as a single
  `page` int (first page) in each claim's citation.
