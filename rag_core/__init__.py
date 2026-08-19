"""
rag_core — the stable interface between the (constantly-changing) research
notebook and everything downstream (backend, frontend).

Only the functions re-exported here are meant to be imported by outside code:

    from rag_core import answer_question, load_index

`answer_question(question)` always returns the frozen dict shape documented
in `generation.py` and in the project guide, no matter what changes inside
ingestion / indexing / retrieval / generation. Rewrite the retrieval strategy
as many times as you want — this contract is the only thing the backend
depends on.
"""

from .generation import answer_question, load_index, is_index_loaded

__all__ = ["answer_question", "load_index", "is_index_loaded"]
