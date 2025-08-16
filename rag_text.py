# rag_text.py — Qualitative RAG with optional FAISS; NumPy fallback if FAISS missing
from __future__ import annotations
import os, pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import trafilatura

INDEX_DIR = os.getenv("RAG_TEXT_DIR", "rag_text")
EMB_NAME = os.getenv("RAG_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MAX_DOCS = int(os.getenv("RAG_MAX_DOCS", "2000"))
os.makedirs(INDEX_DIR, exist_ok=True)

try:
    import faiss  # optional
except Exception:
    faiss = None

def _chunk(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = (text or "").strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size]); i += max(1, size - overlap)
    return out

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

def _load_index():
    idx_path, meta_path = _paths()
    meta = {"chunks": [], "vectors": 0}
    index = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    if faiss and os.path.exists(idx_path):
        try:
            index = faiss.read_index(idx_path)
        except Exception:
            index = None
    return index, meta

def _save_index(index, meta):
    idx_path, meta_path = _paths()
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    if faiss and index is not None:
        faiss.write_index(index, idx_path)

def fetch_and_clean(url: str) -> Tuple[str, str]:
    downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=10)
    if not downloaded: return "", ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted: return "", ""
    md = trafilatura.extract_metadata(downloaded)
    title = md.title if md else ""
    return extracted.strip(), title or url

def add_from_urls_text(urls: List[str]) -> int:
    if not urls: return 0
    index, meta = _load_index()
    emb = _embedder(); added=0
    # Ensure FAISS index exists if available
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
            if len(meta["chunks"]) > MAX_DOCS: break
        except Exception:
            continue
    _save_index(index, meta)
    return added

def retrieve_text(query: str, k: int = 5) -> List[Dict[str, Any]]:
    index, meta = _load_index()
    if meta["vectors"] == 0: return []
    emb = _embedder()
    q = emb.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    results = []
    if faiss and index is not None:
        D, I = index.search(q, min(k, meta["vectors"]))
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(meta["chunks"]):
                ch = meta["chunks"][idx]
                results.append({"url": ch["url"], "title": ch["title"], "snippet": ch["text"][:400], "score": float(score)})
    else:
        vecs = np.array(meta.get("vecs", []), dtype="float32")
        if vecs.size == 0: return []
        sims = (q @ vecs.T)[0]  # cosine on normalized vecs
        top = np.argsort(sims)[::-1][:min(k, len(meta["chunks"]))]
        for idx in top:
            ch = meta["chunks"][idx]
            results.append({"url": ch["url"], "title": ch["title"], "snippet": ch["text"][:400], "score": float(sims[idx])})
    return results
