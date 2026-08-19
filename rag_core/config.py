"""
rag_core.config — one place to tune paths, model names, and thresholds.

Nothing in here should require touching any other rag_core module. If you
find yourself hardcoding a number in ingestion.py / indexing.py /
retrieval.py / generation.py, it probably belongs here instead.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The source guideline PDF. Not committed to git (see .gitignore) — set
# GUIDELINE_PDF_PATH in your .env if it doesn't live at the project root.
PDF_PATH = Path(
    os.environ.get(
        "GUIDELINE_PDF_PATH",
        PROJECT_ROOT / "2026-guidelines-for-the-early-detection-of-prostate-cancer.pdf",
    )
)

# Persisted index artifacts (Chroma + BM25 + chunk metadata). Rebuildable —
# not committed to git. Run `python -m rag_core.build_index` to (re)create.
INDEX_STORE_DIR = Path(os.environ.get("INDEX_STORE_DIR", PROJECT_ROOT / "index_store"))
CHROMA_PERSIST_DIR = INDEX_STORE_DIR / "chroma"
BM25_PATH = INDEX_STORE_DIR / "bm25.pkl"
CHUNKS_PATH = INDEX_STORE_DIR / "chunks.pkl"
CHROMA_COLLECTION_NAME = "pcfa_guideline"

# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------
DOC_ID = "PCFA-EDPC-2026-001"
DOC_TITLE = "2026 Guidelines for the Early Detection of Prostate Cancer in Australia"
DOC_VERSION = "2026"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Defaults come from the notebook's chunk-config comparison (Section 14 —
# "custom_900_175" scored best there). If you change the source PDF or rerun
# that comparison and get a different winner, update these two numbers —
# everything downstream (indexing, retrieval) just reads from here.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 175))

# Section map used for cross-page, section-aware chunking + priority boosting.
# Keep in sync with the notebook if the guideline's table of contents changes.
SECTION_STARTS = [
    (1, "Front matter"),
    (14, "Introduction"),
    (21, "Executive Summary"),
    (25, "Clinical Practice Recommendations"),
    (49, "Section A: Risk assessment"),
    (72, "Section B: Decision support"),
    (74, "Section C: Priority populations"),
    (78, "Section D: Early detection"),
    (141, "Section E: Management"),
    (176, "Section F: Guideline implementation and monitoring"),
    (177, "APPENDIX 1: Governance structure and group membership"),
    (185, "APPENDIX 2: Clinical questions and PICO/PECO"),
    (189, "APPENDIX 3: Literature reviews"),
    (201, "APPENDIX 4: Comparison of selected international guidelines"),
    (206, "APPENDIX 5: Organisations approached for endorsement"),
    (209, "APPENDIX 6: Glossary of terms"),
    (215, "Resources and useful links"),
    (217, "References"),
]

PRIORITY_SECTIONS = {
    "Clinical Practice Recommendations",
    "Section A: Risk assessment",
    "Section D: Early detection",
    "Section E: Management",
}

# PICO/PECO, References, and Resources links are citation/admin-only
# sections with no eval questions expecting answers there — see the
# notebook's Section 3 for why Appendices 3 & 4 were removed from this set.
DEPRIORITY_SECTIONS = {
    "APPENDIX 2: Clinical questions and PICO/PECO",
    "References",
    "Resources and useful links",
}

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "psa": "prostate specific antigen",
    "psad": "prostate specific antigen density psa density",
    "mpmri": "multiparametric magnetic resonance imaging mri",
    "mri": "magnetic resonance imaging",
    "pi-rads": "prostate imaging reporting and data system pirads",
    "pirads": "prostate imaging reporting and data system pi-rads",
    "dre": "digital rectal examination",
}

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
CANDIDATE_POOL = 50   # candidates pulled from fusion before reranking
TOP_K = 5             # final chunks handed to generation
RRF_K = 60
BOOST_WEIGHT = 0.01
BM25_WEIGHT = 1.2
DENSE_WEIGHT = 1.0
RERANK_BLEND_WEIGHT = 0.15  # weight given to the fused/boosted score; cross-encoder gets the rest

# ---------------------------------------------------------------------------
# Confidence / grounding
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_CONFIDENCE_THRESHOLD", 0.0))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
FALLBACK_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"
OPENAI_MODEL = "gpt-4o-mini"


def get_device() -> str:
    """cuda if available, else cpu. Shared by embeddings and the reranker so
    they don't disagree about where the GPU is."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
