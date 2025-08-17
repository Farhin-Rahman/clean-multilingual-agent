# rag_numeric.py — Quantitative RAG: normalize yfinance statements into SQLite and retrieve
from __future__ import annotations
import sqlite3, time
from typing import List, Dict, Any, Tuple
import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
import numpy as np

DB = "secure_portfolios.db"
EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"

METRIC_ALIASES = {
    "revenue": ["revenue", "total revenue", "sales"],
    "net_income": ["net income", "profit", "earnings"],
    "eps": ["eps", "earnings per share"],
    "operating_income": ["operating income", "operating profit", "ebit"],
    "ebitda": ["ebitda"],
    "operating_cash_flow": ["operating cash flow", "ocf"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "total_debt": ["total debt", "long-term debt", "lt debt"],
    "cash": ["cash", "cash and cash equivalents"],
}

def _emb():
    global _E
    try:
        _E
    except NameError:
        _E = SentenceTransformer(EMB_NAME)
    return _E

def _conn():
    return sqlite3.connect(DB)

def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS quant_metrics (
                symbol TEXT NOT NULL,
                metric TEXT NOT NULL,
                period TEXT NOT NULL,
                value REAL NOT NULL,
                source TEXT,
                fetched_at INTEGER,
                PRIMARY KEY(symbol, metric, period)
            )
        """)
        c.commit()

def df_to_records(sym: str, df: pd.DataFrame, source: str) -> List[Tuple[str,str,str,float,str,int]]:
    if df is None or df.empty:
        return []
    out = []
    for row_name, row in df.iterrows():
        metric = str(row_name).strip().lower().replace(" ", "")
        for period, val in row.items():
            try:
                if pd.isna(val): continue
                period_str = str(period)[:10]
                out.append((sym, metric, period_str, float(val), source, int(time.time())))
            except Exception:
                continue
    return out

def ingest_numeric_for_symbols(symbols: List[str]) -> int:
    init_db()
    added = 0
    with _conn() as c:
        for sym in symbols:
            try:
                t = yf.Ticker(sym)
                fins = getattr(t, "financials", pd.DataFrame())
                bs = getattr(t, "balance_sheet", pd.DataFrame())
                cf = getattr(t, "cashflow", pd.DataFrame())
                recs = []
                recs += df_to_records(sym, fins, "yfinance:financials")
                recs += df_to_records(sym, bs, "yfinance:balance_sheet")
                recs += df_to_records(sym, cf, "yfinance:cashflow")
                if recs:
                    c.executemany("""
                        INSERT OR REPLACE INTO quant_metrics(symbol, metric, period, value, source, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, recs)
                    added += len(recs)
                info = t.info or {}
                for k in ["trailingEps","ebitda","totalDebt","operatingCashflow","freeCashflow"]:
                    if k in info and info[k]:
                        c.execute("""
                            INSERT OR REPLACE INTO quant_metrics(symbol, metric, period, value, source, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (sym, k.lower(), "latest", float(info[k]), "yfinance:info", int(time.time())))
                c.commit()
            except Exception:
                continue
    return added

def _metric_candidates_for(symbols: List[str]) -> List[Dict[str, Any]]:
    init_db()
    with _conn() as c:
        qs = ",".join("?"*len(symbols))
        rows = c.execute(f"""
            SELECT symbol, metric, period, value, source FROM quant_metrics
            WHERE symbol IN ({qs})
        """, symbols).fetchall()
        return [{"symbol":r[0], "metric":r[1], "period":r[2], "value":r[3], "source":r[4]} for r in rows]

def _alias_flat():
    items = []
    for canon, alts in METRIC_ALIASES.items():
        for a in [canon] + alts:
            items.append((canon, a))
    return items

def numeric_retrieve(query: str, symbols: List[str], topk: int = 6) -> List[Dict[str, Any]]:
    alias_pairs = _alias_flat()
    texts = [a for _, a in alias_pairs]
    emb = _emb()
    Q = emb.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    A = emb.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    sims = (Q @ A.T)[0]
    top_alias_idx = np.argsort(sims)[::-1][:5]
    selected_canons = {alias_pairs[i][0] for i in top_alias_idx}

    rows = _metric_candidates_for(symbols)
    scored = []
    for r in rows:
        canon = None
        for k in selected_canons:
            if k in r["metric"]:
                canon = k; break
        if not canon:
            continue
        recency = 1.0 if r["period"] in ("latest","Most Recent Quarter") else 0.8
        scored.append((canon, recency, r))
    scored.sort(key=lambda x: (x[0], x[1], x[2]["value"]), reverse=True)

    out = []
    seen = set()
    for canon, _, r in scored:
        key = (r["symbol"], canon)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "symbol": r["symbol"],
            "metric": canon,
            "period": r["period"],
            "value": r["value"],
            "source": r["source"],
            "url": "",
            "title": f"{canon} ({r['period']})"
        })
        if len(out) >= topk:
            break
    return out
