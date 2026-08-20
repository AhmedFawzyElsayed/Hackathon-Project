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
- 2026-08-20: Ported `Clinical_RAG_Prostate_Cancer_Day(1).ipynb` (V2).
  `config.py`: `CONFIDENCE_THRESHOLD` 0.0→6.5 (raw-CE calibrated on
  custom_900_175); added `GLOSSARY_HIT_CONFIDENCE_FLOOR` (5.0),
  `DEFINITIONAL_FRONT_MATTER_BOOST` (6.0), `EVIDENCE_RELEVANCE_THRESHOLD`
  (0.5). `retrieval.py`: query-type-aware retrieval (Bugs 1/4/6) — broadened
  `is_definitional_query`, added `extract_abbreviation_token`,
  `is_threshold_query`, `classify_query_type`, `_glossary_entry_match`,
  `find_glossary_chunks`, `has_content_support`; definitional queries skip
  BM25 expansion; glossary chunks get 6x RRF multiplier; `rerank()` reports the
  raw ce_score (blend used only for ordering); new `retrieve_and_rerank` with
  glossary injection + hard promotion (`retrieve` kept as the module name).
  `retrieval_confidence()` anchors on raw top-1 CE with a glossary_hit floor.
  `generation.py`: 8 grounding rules; Task-14 `is_patient_specific_query`
  (combined first-person + clinical + judgment signal); absolute confidence
  bands (≥10 high, ≥5 medium); Bug 8 `clean_chunk_text` used by citation
  evidence and extractive claims; Bug 2 `extractive_claims` (relevance floor,
  top-3 cap, definitional single claim); Bug 7 `answer_summary` = single best
  claim. Frozen contract (statuses, page int, citation_coverage float,
  confidence_label str, metrics) unchanged.
