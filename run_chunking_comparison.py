"""Step 1: Chunking configuration comparison with all 6 configs."""
import sys, os, re, time, json, warnings
sys.path.insert(0, r"C:\Users\af109\Desktop\Final project\clinical-rag-app")
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PDF_NAME = "C:/Users/af109/Desktop/project/2026-guidelines-for-the-early-detection-of-prostate-cancer.pdf"
DOC_ID = "PCFA-EDPC-2026-001"

print("=" * 60)
print("STEP 1: CHUNKING CONFIGURATION COMPARISON (6 configs)")
print("=" * 60)

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = Path(PDF_NAME)
if not PDF_PATH.exists():
    for p in Path("C:/Users/af109/Desktop").rglob("*prostate*cancer*.pdf"):
        PDF_PATH = p
        break

print(f"PDF: {PDF_PATH}")
pages = PyPDFLoader(str(PDF_PATH)).load()
for p in pages:
    p.metadata["document_id"] = DOC_ID
    p.metadata["page_number"] = p.metadata.get("page", 0) + 1
print(f"Pages: {len(pages)}")

SECTION_STARTS = [
    (1, "Front matter"), (14, "Introduction"), (21, "Executive Summary"),
    (25, "Clinical Practice Recommendations"), (49, "Section A: Risk assessment"),
    (72, "Section B: Decision support"), (74, "Section C: Priority populations"),
    (78, "Section D: Early detection"), (141, "Section E: Management"),
    (176, "Section F: Guideline implementation and monitoring"),
    (177, "APPENDIX 1"), (185, "APPENDIX 2"),
    (189, "APPENDIX 3"), (201, "APPENDIX 4"),
    (206, "APPENDIX 5"), (209, "APPENDIX 6"),
    (215, "Resources and useful links"), (217, "References"),
]

PRIORITY_SECTIONS = {"Clinical Practice Recommendations", "Section A: Risk assessment",
                     "Section D: Early detection", "Section E: Management"}
DEPRIORITY_SECTIONS = {"APPENDIX 2", "References", "Resources and useful links"}

def section_for_page(pn):
    s = SECTION_STARTS[0][1]
    for sp, name in SECTION_STARTS:
        if pn >= sp: s = name
        else: break
    return s

def _pages_by_section():
    grouped = {}
    for page in pages:
        pn = int(page.metadata["page_number"])
        grouped.setdefault(section_for_page(pn), []).append((pn, page.page_content))
    for s in grouped: grouped[s].sort(key=lambda t: t[0])
    return grouped

def make_chunks(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "], add_start_index=True)
    all_chunks = []
    for sec, plist in _pages_by_section().items():
        combined, offsets = "", []
        for pn, text in plist:
            start = len(combined)
            combined += text + "\n"
            offsets.append((start, len(combined), pn))
        sec_doc = Document(page_content=combined, metadata={"section": sec, "document_id": DOC_ID})
        for piece in splitter.split_documents([sec_doc]):
            si = piece.metadata.get("start_index", 0)
            ei = si + len(piece.page_content)
            sp = sorted({pn for (s, e, pn) in offsets if s < ei and e > si})
            if not sp: sp = [min(offsets, key=lambda t: abs(t[0] - si))[2]]
            piece.metadata.update({"section": sec, "document_id": DOC_ID,
                "page_start": sp[0], "page_end": sp[-1], "page_number": sp[0],
                "pages": ",".join(str(p) for p in sp)})
            piece.metadata.pop("start_index", None)
            all_chunks.append(piece)
    for i, c in enumerate(all_chunks, 1):
        c.metadata["chunk_id"] = f"{DOC_ID}-CH-{i:04d}"
    return all_chunks

CHUNK_CONFIGS = {
    "baseline_850_150": (850, 150),
    "custom_900_175": (900, 175),
    "tight_500_120": (500, 120),
    "wide_1100_200": (1100, 200),
    "overlap30_850_255": (850, 255),
    "overlap30_900_270": (900, 270),
}

t0 = time.time()
chunk_sets = {n: make_chunks(s, o) for n, (s, o) in CHUNK_CONFIGS.items()}
for n, cs in chunk_sets.items():
    s, o = CHUNK_CONFIGS[n]
    cross = sum(1 for c in cs if c.metadata["page_start"] != c.metadata["page_end"])
    print(f"  {n:25s} size={s:<5d} overlap={o:<4d} chunks={len(cs):<5d} cross-page={cross}")
print(f"Chunking: {time.time()-t0:.1f}s\n")

# Embeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

class LSABackend(Embeddings):
    def __init__(self, texts, dim=256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.v = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))
        self.s = TruncatedSVD(n_components=min(dim, max(2, len(texts) - 1)), random_state=42)
        self.s.fit(self.v.fit_transform(texts))
    def _e(self, ts):
        from sklearn.preprocessing import normalize
        return normalize(self.s.transform(self.v.transform(ts))).tolist()
    def embed_documents(self, ts): return self._e(ts)
    def embed_query(self, t): return self._e([t])[0]

t0 = time.time()
print("Building embeddings + Chroma indexes...")
emb_models = {n: LSABackend([c.page_content for c in cs]) for n, cs in chunk_sets.items()}
dbs = {n: Chroma.from_documents(cs, emb_models[n], collection_name=f"pcfa_{n}") for n, cs in chunk_sets.items()}
print(f"Embeddings+Chroma: {time.time()-t0:.1f}s\n")

# BM25
from rank_bm25 import BM25Okapi
def tokenize(text): return re.findall(r"[\w.]+", text.lower())
bm25_ix = {n: BM25Okapi([tokenize(c.page_content) for c in cs]) for n, cs in chunk_sets.items()}
print("BM25 indexes ready.\n")

# Query expansion + retrieval
ABBREVIATIONS = {
    "psa": "prostate specific antigen",
    "psad": "prostate specific antigen density psa density",
    "mpmri": "multiparametric magnetic resonance imaging mri",
    "mri": "magnetic resonance imaging",
    "pi-rads": "prostate imaging reporting and data system pirads",
    "pirads": "prostate imaging reporting and data system pi-rads",
    "dre": "digital rectal examination",
}

def expand_query(q):
    ql = q.lower()
    ex = [e for a, e in ABBREVIATIONS.items() if re.search(rf"\b{re.escape(a)}\b", ql)]
    return q + " " + " ".join(ex) if ex else q

def is_definitional(q):
    return bool(re.search(r"stand(s)? for|abbreviation|what does .* mean|what does .* refer to|what does .* mean by", q.lower()))

def section_boost(sec, definitional=False):
    if definitional and sec == "Front matter": return 2
    if sec in PRIORITY_SECTIONS: return 1
    if sec in DEPRIORITY_SECTIONS: return -1
    return 0

def hybrid_retrieve(db, bm25, chunks, question, k=10, pool=50, rrf_k=60, boost_weight=0.01, bm25_weight=1.2, dense_weight=1.0):
    expanded = expand_query(question)
    definitional = is_definitional(question)
    sr = db.similarity_search_with_relevance_scores(question, k=pool)
    bs = bm25.get_scores(tokenize(expanded))
    bri = sorted(range(len(bs)), key=lambda i: bs[i], reverse=True)[:pool]
    rrf, dl = {}, {}
    for rank, (doc, _) in enumerate(sr, 1):
        cid = doc.metadata["chunk_id"]
        dl[cid] = doc
        rrf[cid] = rrf.get(cid, 0.0) + dense_weight / (rrf_k + rank)
    for rank, idx in enumerate(bri, 1):
        doc = chunks[idx]
        cid = doc.metadata["chunk_id"]
        dl[cid] = doc
        rrf[cid] = rrf.get(cid, 0.0) + bm25_weight / (rrf_k + rank)
    for cid, doc in dl.items():
        rrf[cid] += boost_weight * section_boost(doc.metadata.get("section", ""), definitional)
    top = sorted(rrf.keys(), key=rrf.get, reverse=True)[:k]
    return [(dl[cid], rrf[cid]) for cid in top]

def rerank_fallback(question, candidates, top_n=10):
    if not candidates: return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    docs = [d for d, _ in candidates]
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([question] + [d.page_content for d in docs])
    ks = cosine_similarity(X[0:1], X[1:]).ravel()
    fs = [s for _, s in candidates]
    mx = max(fs) if max(fs) > 0 else 1.0
    b = [0.85 * (f / mx) + 0.15 * float(k) for f, k in zip(fs, ks)]
    order = sorted(range(len(docs)), key=lambda i: b[i], reverse=True)
    return [(docs[i], float(b[i]), "tfidf_fallback") for i in order[:top_n]]

def retrieve_and_rerank(cn, q, k=10, pool=50):
    cands = hybrid_retrieve(dbs[cn], bm25_ix[cn], chunk_sets[cn], q, k=pool, pool=pool)
    return rerank_fallback(q, cands, top_n=k)

# Evaluation set - load from the eval_questions.py file or define inline
print("Loading evaluation questions...")
EVAL_QUESTIONS = json.loads(open(r"C:\Users\af109\Desktop\Final project\clinical-rag-app\eval_questions.json", "r").read()) if os.path.exists(r"C:\Users\af109\Desktop\Final project\clinical-rag-app\eval_questions.json") else None

if EVAL_QUESTIONS is None:
    # Try the project directory
    for p in [r"C:\Users\af109\Desktop\eval_questions.json", r"C:\Users\af109\Desktop\project\eval_questions.json"]:
        if os.path.exists(p):
            EVAL_QUESTIONS = json.loads(open(p).read())
            break

if EVAL_QUESTIONS is None:
    print("ERROR: Cannot find eval_questions.json. Building minimal eval set from notebook data.")
    sys.exit(1)

print(f"Loaded {len(EVAL_QUESTIONS)} questions")

# Metrics
def precision_at_k(labels, k): return sum(labels[:k]) / k if k else 0.0
def recall_at_k(labels, k, total):
    if total == 0: return None
    return sum(labels[:k]) / total
def mrr(labels):
    for i, l in enumerate(labels, 1):
        if l: return 1 / i
    return 0.0
def dcg(labels, k): return sum(l / np.log2(i + 2) for i, l in enumerate(labels[:k]))
def ndcg(labels, k):
    ideal = dcg(sorted(labels, reverse=True), k)
    return dcg(labels, k) / ideal if ideal > 0 else 0.0

IN_SCOPE = {q["id"] for q in EVAL_QUESTIONS if q["type"] != "Out-of-scope" and not q.get("duplicate_of")}

def heuristic_label(qid, pages_field):
    item = next(x for x in EVAL_QUESTIONS if x["id"] == qid)
    if item["type"] == "Out-of-scope": return 0
    ep = set(item["expected_pages"])
    cp = {int(p) for p in str(pages_field).split(",") if str(p).strip().isdigit()}
    return int(bool(ep & cp))

def compute_metrics(df, qids=None):
    qids = qids or sorted(df["question_id"].unique())
    rows = []
    for qid in qids:
        g = df[df["question_id"] == qid].sort_values("rank")
        labels = list(g["final_label"])
        total_rel = sum(labels)
        rows.append({
            "question_id": qid,
            "Precision@3": round(precision_at_k(labels, 3), 3),
            "Precision@5": round(precision_at_k(labels, 5), 3),
            "Recall@3": recall_at_k(labels, 3, total_rel),
            "Recall@5": recall_at_k(labels, 5, total_rel),
            "MRR": round(mrr(labels), 3),
            "nDCG@5": round(ndcg(labels, 5), 3),
        })
    t = pd.DataFrame(rows)
    avg = t.drop(columns="question_id").mean(numeric_only=True).round(3)
    return t, avg

# Run all configs
print(f"\nRunning retrieval for {len(IN_SCOPE)} in-scope questions across {len(CHUNK_CONFIGS)} configs...")
all_results = {}
for cname in CHUNK_CONFIGS:
    t0 = time.time()
    rows = []
    for item in EVAL_QUESTIONS:
        rr = retrieve_and_rerank(cname, item["question"], k=10)
        for rank, (doc, score, method) in enumerate(rr, 1):
            rows.append({
                "question_id": item["id"], "question_type": item["type"],
                "question": item["question"], "config": cname, "rank": rank,
                "score": round(float(score), 5), "rerank_method": method,
                "page": int(doc.metadata.get("page_number", 0)),
                "pages": doc.metadata.get("pages", ""),
                "section": doc.metadata.get("section", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "chunk_text": doc.page_content.strip()[:200],
            })
    df = pd.DataFrame(rows)
    df["final_label"] = df.apply(lambda r: heuristic_label(r["question_id"], r["pages"]), axis=1)
    all_results[cname] = df
    print(f"  {cname}: {len(df)} rows in {time.time()-t0:.1f}s")

# Compare
print("\n" + "=" * 80)
print("CHUNKING CONFIGURATION COMPARISON RESULTS")
print("=" * 80)

config_summaries = []
for name in CHUNK_CONFIGS:
    _, avg = compute_metrics(all_results[name], IN_SCOPE)
    s, o = CHUNK_CONFIGS[name]
    combined = round(float(np.mean([avg["Precision@5"], avg["MRR"], avg["nDCG@5"]])), 3)
    config_summaries.append({
        "configuration": name, "chunk_size": s, "overlap": o,
        "Precision@3": avg["Precision@3"], "Precision@5": avg["Precision@5"],
        "Recall@3": avg["Recall@3"], "Recall@5": avg["Recall@5"],
        "MRR": avg["MRR"], "nDCG@5": avg["nDCG@5"], "combined_score": combined,
    })

chunk_experiment = pd.DataFrame(config_summaries).sort_values("combined_score", ascending=False)
print(chunk_experiment.to_string(index=False))

best = chunk_experiment.iloc[0]
print(f"\nBEST CONFIG: {best['configuration']} (combined={best['combined_score']})")
print(f"\nBASELINE (custom_900_175):")
base = chunk_experiment[chunk_experiment["configuration"] == "custom_900_175"].iloc[0]
print(f"  Precision@3: 0.322 -> {best['Precision@3']}")
print(f"  Precision@5: 0.263 -> {best['Precision@5']}")
print(f"  Recall@3:    0.563 -> {best['Recall@3']}")
print(f"  Recall@5:    0.756 -> {best['Recall@5']}")
print(f"  MRR:         0.578 -> {best['MRR']}")
print(f"  nDCG@5:      0.541 -> {best['nDCG@5']}")

chunk_experiment.to_csv(r"C:\Users\af109\Desktop\Final project\clinical-rag-app\chunking_experiment_results.csv", index=False)
print("\nSaved: chunking_experiment_results.csv")
