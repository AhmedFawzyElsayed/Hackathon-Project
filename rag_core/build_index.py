"""
CLI: (re)build the persisted index artifacts (Chroma vector store + BM25 +
chunk metadata) that the backend loads at startup.

    python -m rag_core.build_index

Run this once before starting the backend, and again any time you change
chunking or ingestion. Retrieval tuning, reranking, and generation prompt
changes do NOT require rebuilding the index.
"""
from rag_core import config
from rag_core.indexing import build_or_load_index
from rag_core.ingestion import load_and_chunk_pdf


def main():
    print(f"Loading + chunking {config.PDF_PATH} ...")
    chunks = load_and_chunk_pdf()
    print(f"{len(chunks)} chunks produced (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    print(f"Building index at {config.INDEX_STORE_DIR} ...")
    build_or_load_index(chunks, force_rebuild=True)

    print("Done. Index persisted to", config.INDEX_STORE_DIR)
    print("Start the backend now, or run a smoke test with:")
    print('  python -c "from rag_core import answer_question, load_index; load_index(); '
          "import json; print(json.dumps(answer_question('What does PSAD stand for?'), indent=2))\"")


if __name__ == "__main__":
    main()
