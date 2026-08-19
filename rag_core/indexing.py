"""
rag_core.indexing — build_or_load_index(), lifted from the notebook's
Sections 4-5 (embeddings + Chroma, BM25 keyword index).

Two entry points:
  * build_or_load_index(chunks, ...)  — (re)build from a fresh chunk list
    and persist to disk. Called by build_index.py.
  * load_index(...)                   — load already-persisted artifacts.
    Called once at backend startup — never per-request.
"""
import pickle
import re
import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import config

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class OfflineLSAEmbeddings(Embeddings):
    """Local TF-IDF + SVD fallback, used only if BGE-M3 can't be reached/loaded."""

    def __init__(self, texts, dim=256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        self.svd = TruncatedSVD(n_components=min(dim, max(2, len(texts) - 1)), random_state=42)
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)

    def _embed(self, texts):
        from sklearn.preprocessing import normalize

        v = self.svd.transform(self.vectorizer.transform(texts))
        return normalize(v).tolist()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


_BGE_M3_MODEL = None
_BGE_M3_CHECKED = False


def _load_bge_m3():
    global _BGE_M3_MODEL, _BGE_M3_CHECKED
    if _BGE_M3_CHECKED:
        return _BGE_M3_MODEL
    _BGE_M3_CHECKED = True
    try:
        import os

        from sentence_transformers import SentenceTransformer

        load_kwargs = {"device": config.get_device()}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            load_kwargs["token"] = hf_token
        _BGE_M3_MODEL = SentenceTransformer(config.EMBEDDING_MODEL_NAME, **load_kwargs)
        print(f"Loaded {config.EMBEDDING_MODEL_NAME} on {config.get_device()}")
    except Exception as e:  # pragma: no cover - depends on network/model availability
        print(f"BGE-M3 unavailable ({e})")
        _BGE_M3_MODEL = None
    return _BGE_M3_MODEL


class BGEM3Embeddings(Embeddings):
    """Dense embeddings from BAAI/bge-m3. Same encoder for docs and queries,
    cosine similarity via L2-normalized vectors (what Chroma's default
    similarity search expects)."""

    def __init__(self, batch_size=32):
        self.model = _load_bge_m3()
        self.batch_size = batch_size

    def embed_documents(self, texts):
        embeds = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return embeds.tolist()

    def embed_query(self, text):
        embed = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return embed.tolist()


def build_embedding_model(chunks_for_this_corpus: list[Document]) -> Embeddings:
    model = _load_bge_m3()
    if model is not None:
        print(f"Using {config.EMBEDDING_MODEL_NAME} (device={config.get_device()})")
        return BGEM3Embeddings(batch_size=32)
    print("BGE-M3 unavailable; building a fresh TF-IDF+SVD fallback for this corpus")
    return OfflineLSAEmbeddings([c.page_content for c in chunks_for_this_corpus])


# ---------------------------------------------------------------------------
# BM25 keyword index
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    # \w with re.UNICODE (Python 3 default) matches Unicode letters/digits,
    # so symbols like "μ" and "≥" survive tokenization. "." is kept so
    # decimals like "3.0" tokenize as one piece.
    return re.findall(r"[\w.]+", text.lower())


def build_bm25(chunks: list[Document]):
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize(c.page_content) for c in chunks]
    return BM25Okapi(corpus_tokens)


# ---------------------------------------------------------------------------
# Build / persist / load
# ---------------------------------------------------------------------------
def build_or_load_index(
    chunks: list[Document],
    persist_dir: Path | None = None,
    bm25_path: Path | None = None,
    chunks_path: Path | None = None,
    force_rebuild: bool = True,
):
    """Build the Chroma vector store + BM25 index from `chunks` and persist
    both (plus the chunk list itself, since BM25 needs it to map scores back
    to documents) to disk. Returns (db, bm25, chunks).
    """
    from langchain_chroma import Chroma

    persist_dir = Path(persist_dir or config.CHROMA_PERSIST_DIR)
    bm25_path = Path(bm25_path or config.BM25_PATH)
    chunks_path = Path(chunks_path or config.CHUNKS_PATH)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embedding_model = build_embedding_model(chunks)

    db = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )

    bm25 = build_bm25(chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return db, bm25, chunks


def load_index(
    persist_dir: Path | None = None,
    bm25_path: Path | None = None,
    chunks_path: Path | None = None,
):
    """Load already-persisted index artifacts. Raises FileNotFoundError if
    `build_index.py` hasn't been run yet. Call this once at backend startup."""
    from langchain_chroma import Chroma

    persist_dir = Path(persist_dir or config.CHROMA_PERSIST_DIR)
    bm25_path = Path(bm25_path or config.BM25_PATH)
    chunks_path = Path(chunks_path or config.CHUNKS_PATH)

    if not persist_dir.exists() or not bm25_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Index artifacts not found under "
            f"{persist_dir.parent}. Run `python -m rag_core.build_index` first."
        )

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    embedding_model = build_embedding_model(chunks)
    db = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )

    return db, bm25, chunks
