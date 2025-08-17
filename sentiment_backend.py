# sentiment_backend.py — unified sentiment with FinBERT primary, VADER fallback
from __future__ import annotations
import os
from typing import List, Optional

_SENT_PIPE = None
_SIA = None
_ENGINE = (os.getenv("SENTIMENT_ENGINE") or "auto").lower()  # auto | finbert | vader

def _init_finbert():
    global _SENT_PIPE
    if _SENT_PIPE is not None:
        return _SENT_PIPE
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _SENT_PIPE = pipeline("sentiment-analysis", model=mdl, tokenizer=tok)
        return _SENT_PIPE
    except Exception:
        _SENT_PIPE = None
        return None

def _init_vader():
    global _SIA
    if _SIA is not None:
        return _SIA
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _SIA = SentimentIntensityAnalyzer()
        return _SIA
    except Exception:
        _SIA = None
        return None

def _score_finbert(texts: List[str]) -> Optional[float]:
    pipe = _init_finbert()
    if not pipe:
        return None
    label_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    vals = []
    for t in texts[:10]:
        if not t:
            continue
        try:
            r = pipe(t)[0]
            vals.append(label_map.get(r["label"].lower(), 0.0) * float(r.get("score", 0.0)))
        except Exception:
            continue
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))

def _score_vader(texts: List[str]) -> Optional[float]:
    sia = _init_vader()
    if not sia:
        return None
    s, n = 0.0, 0
    for t in texts[:10]:
        try:
            s += float(sia.polarity_scores(t or "").get("compound", 0.0)); n += 1
        except Exception:
            continue
    return float(s / max(1, n))

def sent_score(texts: List[str]) -> float:
    """
    Returns sentiment in [-1, 1].
    auto: FinBERT → VADER → neutral
    finbert: FinBERT only (else 0.0)
    vader: VADER only (else 0.0)
    """
    texts = texts or []
    if _ENGINE in ("auto", "finbert"):
        s = _score_finbert(texts)
        if s is not None:
            return s
        if _ENGINE == "finbert":
            return 0.0
    if _ENGINE in ("auto", "vader"):
        s = _score_vader(texts)
        if s is not None:
            return s
    return 0.0
