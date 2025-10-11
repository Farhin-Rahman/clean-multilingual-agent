# services/attachments.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any

# Optional deps (we degrade gracefully if not installed)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


class AttachmentIndex:
    """
    Tiny in-memory index for attachments. Good enough for demo:
    - add(text, meta)
    - build()
    - query(q, k) -> list[(text, score)]
    """
    def __init__(self):
        self._docs: List[Tuple[str, Dict[str, Any]]] = []
        self.is_built: bool = False

    def add(self, text: str, meta: Dict[str, Any] | None = None) -> None:
        if text:
            self._docs.append((text, meta or {}))

    def build(self) -> None:
        self.is_built = len(self._docs) > 0

    def query(self, q: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.is_built or not q:
            return []
        q_terms = set(q.lower().split())
        scored: List[Tuple[str, float]] = []
        for text, _meta in self._docs:
            t = text.lower()
            score = sum(1.0 for w in q_terms if w and w in t)
            scored.append((text, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def _extract_text_from_pdf(path: str) -> str:
    if not fitz:
        return ""
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        return ""


def _extract_text_from_image(path: str) -> str:
    if not (Image and pytesseract):
        return ""
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def process_attachment_file(path: str, mime: str | None = None) -> Tuple[str, Dict[str, Any]]:
    m = (mime or "").lower()
    text = ""
    if path.lower().endswith(".pdf") or "pdf" in m:
        text = _extract_text_from_pdf(path)
    elif any(path.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")) or "image" in m:
        text = _extract_text_from_image(path)
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""

    meta = {"path": path, "mime": m}
    return text, meta


def process_attachment_files(files: List[Tuple[str, str | None]]) -> AttachmentIndex:
    """
    files: list of (path, mime)
    """
    idx = AttachmentIndex()
    for path, mime in files:
        text, meta = process_attachment_file(path, mime)
        if text:
            idx.add(text, meta)
    idx.build()
    return idx
