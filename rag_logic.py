# rag_logic.py — Logical RAG with optional FAISS; NumPy fallback if FAISS missing
from __future__ import annotations
import os, pickle, re
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import trafilatura

INDEX_DIR = os.getenv("RAG_LOGIC_DIR", "rag_logic")
EMB_NAME = os.getenv("RAG_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
os.makedirs(INDEX_DIR, exist_ok=True)

try:
    import faiss  # optional
except Exception:
    faiss = None

LOGICAL_PATTERNS = {
    "deal_terms": r"\b(merger|acquisition|LOI|definitive agreement|all[- ]cash|stock[- ]for[- ]stock|tender offer|SPAC)\b",
    "litigation": r"\b(class action|lawsuit|litigation|subpoena|SEC investigation|CFTC|DoJ|settlement)\b",
    "financing": r"\b(private placement|convertible notes|rights offering|PIPE|credit facility|term loan|bond issue)\b",
    "regulatory": r"\b(antitrust|CMA|FTC|EC|SEC filing|Form 10[- ]?K|10[- ]?Q|8[- ]?K|S[- ]?1|proxy)\b",
}

def _embedder():
    global _EMB
    try:
        _EMB
    except NameError:
        _EMB = SentenceTransformer(EMB_NAME)
    return _EMB

def _paths():
    return (os.path.join(INDEX_DIR, "faiss.index"),
            os.path.join(INDEX_DIR, "meta.pkl"))

def _chunk(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = (text or "").strip()
    out=[]; i=0
    while i < len(text):
        out.append(text[i:i+size]); i += max(1, size - overlap)
    return out

def _load_index():
    idx_path, meta_path = _paths()
    meta = {"chunks": [], "vectors": 0}
    index = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f: meta = pickle.load(f)
    if faiss and os.path.exists(idx_path):
        try:
            index = faiss.read_index(idx_path)
        except Exception:
            index = None
    return index, meta

def _save_index(index, meta):
    idx_path, meta_path = _paths()
    with open(meta_path, "wb") as f: pickle.dump(meta, f)
    if faiss and index is not None:
        faiss.write_index(index, idx_path)

def fetch_and_clean(url: str):
    downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=10)
    if not downloaded: return "", ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted: return "", ""
    md = trafilatura.extract_metadata(downloaded)
    title = md.title if md else ""
    return extracted.strip(), title or url

def add_from_urls_logic(urls: List[str]) -> int:
    if not urls: return 0
    index, meta = _load_index(); emb = _embedder(); added=0
    if faiss and index is None:
        d = emb.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(d)
    for url in urls:
        try:
            text, title = fetch_and_clean(url)
            if not text or len(text) < 200: continue
            chunks = _chunk(text)
            vecs = emb.encode(chunks, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            if faiss and index is not None:
                index.add(vecs)
            else:
                cur = np.array(meta.get("vecs", []), dtype="float32")
                meta["vecs"] = np.vstack([cur, vecs]) if cur.size else vecs
            for ch in chunks:
                meta["chunks"].append({"url": url, "title": title, "text": ch})
            meta["vectors"] += len(chunks); added += 1
        except Exception:
            continue
    _save_index(index, meta)
    return added

def retrieve_logic(query: str, k: int = 5) -> List[Dict[str, Any]]:
    index, meta = _load_index()
    if meta["vectors"] == 0: return []
    emb = _embedder()
    q = emb.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    res=[]
    if faiss and index is not None:
        D, I = index.search(q, min(k, meta["vectors"]))
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(meta["chunks"]):
                ch = meta["chunks"][idx]
                res.append({"url": ch["url"], "title": ch["title"], "snippet": ch["text"][:500], "score": float(score)})
    else:
        vecs = np.array(meta.get("vecs", []), dtype="float32")
        if vecs.size == 0: return []
        sims = (q @ vecs.T)[0]
        top = np.argsort(sims)[::-1][:min(k, len(meta["chunks"]))]  # best k
        for idx in top:
            ch = meta["chunks"][idx]
            res.append({"url": ch["url"], "title": ch["title"], "snippet": ch["text"][:500], "score": float(sims[idx])})
    return res

def extract_logical(snippets: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    text = " ".join(snippets or [])
    if not text: return out
    for k, pat in LOGICAL_PATTERNS.items():
        m = re.findall(pat, text, flags=re.I)
        if m:
            out[k] = sorted(set(s.lower() for s in m))
    return out
