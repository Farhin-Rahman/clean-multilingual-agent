# support_agent.py
from __future__ import annotations
from typing import TypedDict, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv

import fitz  # PyMuPDF

import os, re, time, json, math, logging, hashlib, sqlite3
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential

# Optional: language detection/translation
try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None  # translation optional

# Search
from duckduckgo_search import DDGS

# Validation & parsing
from pydantic import BaseModel, Field, ValidationError
import sqlglot
from sqlglot.expressions import Select

# RAG modules (project-local)
from rag_text import add_from_urls_text, retrieve_text
from rag_logic import add_from_urls_logic, retrieve_logic, extract_logical as extract_logical_from_hits
from rag_numeric import ingest_numeric_for_symbols, numeric_retrieve

# Sentiment (FinBERT primary, VADER fallback handled in sentiment backend)
from sentiment_backend import sent_score as _sent_score

load_dotenv()

# --------------------------- Logging & knobs ------------------------------
LOGLEVEL = (os.getenv("LOGLEVEL") or "INFO").upper()
logging.basicConfig(level=getattr(logging, LOGLEVEL, logging.INFO))
logger = logging.getLogger("support_agent")

# Optional HTTP cache (global install)
try:
    import requests_cache  # type: ignore
    requests_cache.install_cache("web_cache", backend="sqlite", expire_after=3600)
    logger.info("requests-cache enabled.")
except Exception:
    logger.warning("requests-cache unavailable; continuing without HTTP cache.")
    requests_cache = None  # type: ignore

# Demo / hardening env knobs
AGENT_DISABLE_LLM = (os.getenv("AGENT_DISABLE_LLM", "0") == "1")
YF_MAX_SYMBOLS = int(os.getenv("YF_MAX_SYMBOLS", "8"))  # keep small to avoid throttling
SP500_CACHE_FILE = "sp500_cache.csv"
SP500_CACHE_TTL_HOURS = 24
SP500_FULL_CACHE_FILE = "sp500_full_cache.csv"

# ---- Performance budget knob (added) ----
AGENT_BUDGET_MS = int(os.getenv("AGENT_BUDGET_MS", "1200"))

def _over_budget(state) -> bool:
    """Return True if we've exceeded the overall agent time budget."""
    t0 = state.get("budget_t0")
    return bool(t0) and ((time.time() - t0) * 1000.0 > AGENT_BUDGET_MS)

# ------------------------------ Security ---------------------------------
class SecurityManager:
    @staticmethod
    def allowlist_sql(query: str) -> str:
        q = (query or "").strip()
        if re.search(r"(--|/\*|\*/)", q):
            raise ValueError("SQL comments are not allowed.")
        q = q.replace(";", "")
        if not re.match(r"^\s*select\b", q, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        return q

    @staticmethod
    def validate_user_id(user_id: str) -> str:
        if not user_id or len(user_id) < 1:
            return "anonymous_user"
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", user_id)[:50]
        return sanitized or "anonymous_user"

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()[:16]

APPROVED_TABLES = {"companies", "portfolios"}

def validate_sql_ast(query: str) -> str:
    safe_q = SecurityManager.allowlist_sql(query)
    try:
        ast = sqlglot.parse_one(safe_q, read="sqlite")
        if not isinstance(ast, Select):
            raise ValueError("Only SELECT statements are allowed.")
        tables = {t.name.lower() for t in ast.find_all(sqlglot.exp.Table)}
        if not tables.issubset(APPROVED_TABLES):
            raise ValueError(f"Query references non-approved tables: {tables - APPROVED_TABLES}")
        if any(isinstance(node, sqlglot.exp.Union) for node in ast.find_all(sqlglot.exp.Union)):
            raise ValueError("UNION is not allowed.")
        return safe_q
    except Exception as e:
        raise ValueError(f"SQL validation failed: {e}")

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.limits = {"free": {"requests": 100, "window": 3600}}

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        now = time.time()
        user_key = f"{user_id}_free"
        if user_key not in self.requests:
            self.requests[user_key] = []
        window = self.limits["free"]["window"]
        self.requests[user_key] = [t for t in self.requests[user_key] if now - t < window]
        current = len(self.requests[user_key])
        limit = self.limits["free"]["requests"]
        if current < limit:
            self.requests[user_key].append(now)
            return True, limit - current - 1
        return False, 0

class ConversationMemory:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

    def add_exchange(self, user_id: str, user_msg: str, agent_msg: str, context: dict | None = None):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        ex = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "agent": agent_msg,
            "context": context or {},
            "tokens": len(user_msg.split()) + len(agent_msg.split()),
        }
        self.conversations[user_id].append(ex)
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]

    def get_context_summary(self, user_id: str, max_tokens: int = 1000) -> str:
        if user_id not in self.conversations:
            return ""
        recent = self.conversations[user_id][-5:]
        parts, total = [], 0
        for ex in reversed(recent):
            if total + ex["tokens"] > max_tokens:
                break
            parts.append(f"User: {ex['user']}\nAssistant: {ex['agent'][:200]}...")
            total += ex["tokens"]
        parts.reverse()
        fin = self.extract_financial_preferences(user_id)
        if fin:
            parts.append(f"\nUser Financial Profile: {fin}")
        return "\n---\n".join(parts)

    def extract_financial_preferences(self, user_id: str) -> str:
        if user_id not in self.conversations:
            return ""
        prefs = {"risk_tolerance": None, "sectors": []}
        all_text = " ".join([ex["user"] + " " + ex["agent"] for ex in self.conversations[user_id]]).lower()
        if any(w in all_text for w in ["conservative", "safe", "low risk"]):
            prefs["risk_tolerance"] = "Conservative"
        elif any(w in all_text for w in ["aggressive", "high risk", "growth"]):
            prefs["risk_tolerance"] = "Aggressive"
        sectors = ["technology", "healthcare", "finance", "energy", "consumer"]
        for s in sectors:
            if s in all_text:
                prefs["sectors"].append(s)
        out = []
        if prefs["risk_tolerance"]:
            out.append(f"Risk: {prefs['risk_tolerance']}")
        if prefs["sectors"]:
            out.append(f"Interested sectors: {', '.join(prefs['sectors'])}")
        return "; ".join(out)

# Globals
rate_limiter: Optional[RateLimiter] = None
conversation_memory: Optional[ConversationMemory] = None

@dataclass
class _LLMResponse:
    content: str

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

MODEL_CATALOG = {
    "sql": ["llama3.1:8b-instruct-q4_0", "mistral:7b-instruct-q4_0"],
    "analysis": ["mistral:7b-instruct-q4_0", "llama3.1:8b-instruct-q4_0"],
    "default": ["llama3.1:8b-instruct-q4_0", "mistral:7b-instruct-q4_0"],
}

def _ollama_invoke(messages: List[Dict[str, str]], model: str, temperature: float = 0.1) -> _LLMResponse:
    try:
        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
        r = requests.post(OLLAMA_URL, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        txt = (data.get("message", {}) or {}).get("content", "") or data.get("content", "") or ""
        return _LLMResponse(txt)
    except Exception as e:
        logger.warning(f"Ollama invoke failed for {model}: {e}")
        return _LLMResponse("")

def llm_call(task: str, messages: List[Dict[str, str]], max_tries: int = 2) -> _LLMResponse:
    if AGENT_DISABLE_LLM:
        if task == "sql":
            stub = (
                "```sql\n"
                "SELECT symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta "
                "FROM companies ORDER BY market_cap DESC LIMIT 20;\n"
                "```\n"
                "```sql\n"
                "SELECT symbol, name, sector, pe_ratio, beta FROM companies "
                "WHERE pe_ratio > 0 AND pe_ratio < 25 AND beta < 1.3 "
                "ORDER BY pe_ratio ASC LIMIT 20;\n"
                "```\n"
            )
            return _LLMResponse(stub)
        return _LLMResponse("—")

    models = MODEL_CATALOG.get(task, MODEL_CATALOG["default"])
    last = _LLMResponse("")
    for m in models[:max_tries]:
        resp = _ollama_invoke(messages, model=m)
        txt = resp.content.strip()
        if task == "sql":
            if re.search(r"```sql", txt, re.IGNORECASE):
                return resp
        else:
            if len(txt) > 5:
                return resp
        last = resp

    if task == "sql":
        stub = (
            "```sql\n"
            "SELECT symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta "
            "FROM companies ORDER BY market_cap DESC LIMIT 20;\n"
            "```\n"
            "```sql\n"
            "SELECT symbol, name, sector, pe_ratio, beta FROM companies "
            "WHERE pe_ratio > 0 AND pe_ratio < 25 AND beta < 1.3 "
            "ORDER BY pe_ratio ASC LIMIT 20;\n"
            "```\n"
        )
        return _LLMResponse(stub)
    return _LLMResponse("—")

# ======================================================================
# =================== PRICE HISTORY (YF → STOOQ → SYNTHETIC) ===========
# ======================================================================

# Small fundamentals fallback (to print nicer details in bullets)
FUND_FALLBACK = {
    "MMM": {"longName": "3M", "sector": "Industrials — Conglomerates", "trailingPE": 15.8,
            "marketCap": 52_000_000_000, "dividendYield": 0.052, "beta": 0.85},
    "AOS": {"longName": "A. O. Smith", "sector": "Industrials — Building Products", "trailingPE": 22.4,
            "marketCap": 16_000_000_000, "dividendYield": 0.019, "beta": 0.95},
    "ABT": {"longName": "Abbott Laboratories", "sector": "Healthcare — Medical Devices", "trailingPE": 23.9,
            "marketCap": 190_000_000_000, "dividendYield": 0.019, "beta": 0.90},
}

def fetch_history(symbol: str, days: int = 160) -> pd.DataFrame:
    """
    Returns a DataFrame with a 'Close' column.
    Priority:
      1) yfinance (ONLY if SHIM_YF == "0")
      2) Stooq via pandas-datareader (AAPL -> AAPL.US)
      3) Synthetic series (last-resort, offline)
    """
    # 1) yfinance live (only if you explicitly turned shim off)
    try:
        if os.getenv("SHIM_YF", "1") == "0":
            h = yf.Ticker(symbol).history(period="6mo", interval="1d", auto_adjust=True)
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                return h[["Close"]].copy()
    except Exception:
        pass

    # 2) Stooq (robust free daily source)
    try:
        from pandas_datareader import data as pdr
        stooq_symbol = symbol
        if not stooq_symbol.startswith("^") and "." not in stooq_symbol:
            stooq_symbol = f"{stooq_symbol}.US"
        df = pdr.DataReader(stooq_symbol, "stooq")  # daily OHLCV
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            df = df.sort_index()
            cutoff = pd.Timestamp(date.today() - timedelta(days=days*2))
            df = df.loc[df.index >= cutoff]
            return df[["Close"]].copy()
    except Exception:
        pass

    # 3) Synthetic (keeps the app alive in true offline)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    rets = 0.0004 + 0.02 * rng.standard_normal(days) / (252 ** 0.5)
    prices = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range(end=date.today(), periods=days)
    return pd.DataFrame({"Close": prices}, index=idx)

def calculate_rsi(prices: pd.Series, periods: int = 14) -> float:
    if len(prices) < periods:
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0

def _spy_series(period: str = "6mo") -> pd.Series:
    try:
        s = fetch_history("SPY", days=160)["Close"].dropna()
        return s
    except Exception:
        return pd.Series(dtype=float)

def _compute_beta(sym_close: pd.Series, mkt_close: pd.Series) -> float:
    try:
        df = pd.DataFrame({"s": sym_close.pct_change(), "m": mkt_close.pct_change()}).dropna()
        if len(df) < 30:
            return 1.0
        cov = np.cov(df["s"], df["m"])[0, 1]
        var = np.var(df["m"])
        if var <= 0:
            return 1.0
        return float(cov / var)
    except Exception:
        return 1.0

# ---------- DDG with retry (added) ----------
@retry(wait=wait_exponential(min=0.2, max=2.0), stop=stop_after_attempt(3))
def _safe_ddg_text(query: str, max_results: int = 5):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))

def _headlines_for_symbol(sym: str, max_n: int = 6) -> List[dict]:
    """Use DuckDuckGo (shimmed in demo) for headlines."""
    items: List[dict] = []
    try:
        q = f'{sym} stock news site:reuters.com OR site:barrons.com OR site:bloomberg.com OR site:investors.com OR site:cnbc.com'
        for r in _safe_ddg_text(q, max_n):
            t = (r.get("title") or "").strip()
            u = (r.get("href") or "").strip()
            if t and u and u.startswith("http"):
                items.append({"title": t, "url": u, "publisher": "", "date": ""})
    except Exception:
        pass
    return items[:max_n]

def _download_closes(symbols: List[str]) -> dict[str, pd.Series]:
    """Uniformly get close series via fast multi-ticker path when live, else robust fallback."""
    syms = list(dict.fromkeys([s for s in symbols if s]))  # dedupe/preserve order
    out: dict[str, pd.Series] = {}

    # Fast path: multi-ticker yf.download if shim is OFF
    try:
        if os.getenv("SHIM_YF", "1") == "0" and syms:
            df = yf.download(
                syms, period="6mo", interval="1d",
                auto_adjust=True, threads=True, progress=False
            )
            # MultiIndex (Close, TICKER) when multiple; plain 'Close' for single
            if isinstance(df.columns, pd.MultiIndex):
                if "Close" in df.columns.get_level_values(0):
                    for s in syms:
                        try:
                            series = df[("Close", s)].dropna()
                            if not series.empty:
                                out[s] = series
                        except Exception:
                            continue
            elif "Close" in df:
                s0 = syms[0]
                series = df["Close"].dropna()
                if not series.empty:
                    out[s0] = series
    except Exception:
        pass

    if out:
        return out

    # Fallback: per-symbol robust path via fetch_history (includes Stooq/synthetic)
    for sym in syms:
        try:
            df = fetch_history(sym, days=160)
            s = df["Close"].dropna()
            if not s.empty:
                out[sym] = s
        except Exception:
            continue
    return out

def get_real_stock_data_parallel(symbols: List[str]) -> List[Dict[str, Any]]:
    """Close prices via Stooq (or YF if allowed), indicators + FinBERT/VADER sentiment."""
    syms = list(dict.fromkeys([s for s in symbols if s]))[:max(1, YF_MAX_SYMBOLS)]
    out: List[Dict[str, Any]] = []
    if not syms:
        return out

    closes_map = _download_closes(syms)
    spy = _spy_series()

    for sym in syms:
        try:
            close = closes_map.get(sym, pd.Series(dtype=float))
            if close.empty:
                continue
            price_now = float(close.iloc[-1])
            # daily std over full sample -> annualized vol
            vol_ann = float(close.pct_change().std() * np.sqrt(252)) if len(close) >= 30 else 0.0
            sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price_now
            rsi = float(calculate_rsi(close)) if len(close) >= 14 else 50.0
            beta = _compute_beta(close, spy) if not spy.empty else 1.0
            risk = "Low" if (beta < 0.9 and vol_ann < 0.25) else ("High" if (beta > 1.2 or vol_ann > 0.40) else "Medium")
            headlines = _headlines_for_symbol(sym)
            sent = _sent_score([h["title"] for h in headlines])

            # fundamentals (fallback, if available)
            meta = FUND_FALLBACK.get(sym.upper(), {})
            sector = meta.get("sector", "Unknown")
            pe = meta.get("trailingPE", 0.0)
            mcap = meta.get("marketCap", 0.0)
            dy = meta.get("dividendYield", None)

            out.append({
                "name": meta.get("longName", sym),
                "symbol": sym,
                "sector": sector,
                "market_cap": mcap,
                "pe_ratio": pe,
                "dividend_yield": dy,
                "price": round(price_now, 2),
                "risk": risk,
                "beta": round(beta, 2),
                "volatility": round(vol_ann, 3),
                "sma_20": round(sma20, 2),
                "rsi": round(rsi, 1),
                "trend": "Bullish" if price_now > sma20 else "Bearish",
                "sentiment": round(sent, 3),
                "provenance": [{"type": "news", **h} for h in headlines],
            })
        except Exception:
            continue

    return out

def normalize_company_keys(c: dict) -> dict:
    return {
        "name": c.get("name") or c.get("longName") or c.get("symbol"),
        "symbol": c.get("symbol"),
        "sector": c.get("sector") or "Unknown",
        "market_cap": c.get("market_cap") or c.get("marketCap"),
        "pe_ratio": c.get("pe_ratio") or c.get("trailingPE"),
        "price": c.get("price") or c.get("current_price"),
        "risk": c.get("risk") or c.get("risk_level") or "Medium",
        "beta": c.get("beta") or 1.0,
        "volatility": c.get("volatility") or 0.0,
        "sma_20": c.get("sma_20") or c.get("sma20"),
        "rsi": c.get("rsi") or 50.0,
        "trend": c.get("trend") or "",
        "sentiment": c.get("sentiment") or 0.0,
        "dividend_yield": c.get("dividend_yield"),
        "provenance": c.get("provenance") or [],
    }

# --------------------------- Sector prefilter -----------------------------
def get_sp500_tickers() -> List[str]:
    try:
        if os.path.exists(SP500_CACHE_FILE):
            age = time.time() - os.path.getmtime(SP500_CACHE_FILE)
            if age < SP500_CACHE_TTL_HOURS * 3600:
                return pd.read_csv(SP500_CACHE_FILE)["Symbol"].tolist()
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        pd.DataFrame({"Symbol": tickers}).to_csv(SP500_CACHE_FILE, index=False)
        return tickers
    except Exception as e:
        logger.warning(f"Failed to get S&P 500 list live: {e}")
    if os.path.exists(SP500_CACHE_FILE):
        try:
            return pd.read_csv(SP500_CACHE_FILE)["Symbol"].tolist()
        except Exception:
            pass
    return ["AAPL","MSFT","GOOGL","AMZN","TSLA","META","NVDA","JPM","JNJ","V"]

def get_sp500_table_cached() -> pd.DataFrame:
    try:
        if os.path.exists(SP500_FULL_CACHE_FILE):
            age = time.time() - os.path.getmtime(SP500_FULL_CACHE_FILE)
            if age < SP500_CACHE_TTL_HOURS * 3600:
                return pd.read_csv(SP500_FULL_CACHE_FILE)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df.rename(columns={"GICS Sector": "Sector"}, inplace=True)
        df.to_csv(SP500_FULL_CACHE_FILE, index=False)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch full S&P500 table live: {e}")
    if os.path.exists(SP500_FULL_CACHE_FILE):
        try:
            return pd.read_csv(SP500_FULL_CACHE_FILE)
        except Exception:
            pass
    tickers = get_sp500_tickers()
    return pd.DataFrame({"Symbol": tickers, "Sector": ["Unknown"] * len(tickers)})


def prefilter_symbols_by_query(query_lower: str, spx_df: pd.DataFrame, max_candidates: int = 30) -> List[str]:
    sector_aliases = {
        "Information Technology": ["tech", "technology", "software", "semis", "semiconductor", "it"],
        "Health Care": ["health", "healthcare", "medical", "pharma", "biotech"],
        "Financials": ["finance", "bank", "financial", "insurance", "broker"],
        "Energy": ["energy", "oil", "gas", "o&g"],
        "Consumer Discretionary": ["consumer discretionary", "retail", "ecommerce", "auto"],
        "Consumer Staples": ["consumer staples", "staples", "grocery", "beverage", "food"],
        "Industrials": ["industrial", "industrials", "manufacturing", "aerospace", "defense"],
        "Utilities": ["utility", "utilities", "power", "electric"],
        "Real Estate": ["reit", "real estate", "property"],
        "Communication Services": ["communication", "telecom", "media", "social"],
        "Materials": ["materials", "mining", "chemicals", "paper"],
    }
    wanted = {sector for sector, keys in sector_aliases.items() if any(k in query_lower for k in keys)}
    df = spx_df[spx_df["Sector"].isin(sorted(wanted))] if wanted else spx_df
    return df["Symbol"].tolist()[:max_candidates]

def get_real_stock_data(symbols: List[str]) -> List[Dict[str, Any]]:
    return get_real_stock_data_parallel(symbols)

def get_stocks_by_real_criteria(query: str) -> List[Dict[str, Any]]:
    query_lower = query.lower()
    spx_df = get_sp500_table_cached()
    candidates = prefilter_symbols_by_query(query_lower, spx_df, max_candidates=30) or get_sp500_tickers()[:30]
    real_companies = get_real_stock_data_parallel(candidates[:YF_MAX_SYMBOLS])
    # minimal filters
    filtered = real_companies
    tmp = filtered
    if any(w in query_lower for w in ["tech", "technology", "software"]):
        t = [c for c in tmp if "technology" in c.get("sector", "").lower()]
        if t: tmp = t
    if any(w in query_lower for w in ["safe", "conservative", "stable", "low risk", "low-risk"]):
        t = [c for c in tmp if (c.get("beta",1.0) < 1.0 and 0 < (c.get("pe_ratio") or 0) < 25 and (c.get("market_cap") or 0) > 1e10)]
        if t: tmp = t
    if "under 50" in query_lower or "under $50" in query_lower:
        t = [c for c in tmp if (c.get("price") or 999) < 50]
        if t: tmp = t
    filtered = tmp
    return filtered[:10] or real_companies[:10]

# ----------------------------- Rules & MCDM -------------------------------
def check_constraints(company: dict, user_profile: dict) -> dict:
    flags = []
    risk_tol = (user_profile.get("risk_tolerance") or "").lower()
    beta = company.get("beta") or 1.0
    vol = company.get("volatility") or 0.3
    pe = company.get("pe_ratio") if company.get("pe_ratio") is not None else 0.0
    if risk_tol in ("conservative","low"):
        if beta > 1.3 or vol > 0.45:
            flags.append("risk_exceeds_profile")
    if (pe or 0) < 0 and not company.get("eps"):
        flags.append("pe_negative_unclear")
    return {"allowed": len(flags) == 0, "flags": flags}

RULES = [
    {"id":"conservative_block_high_beta","desc":"Block high beta for conservative","when": lambda c,p: (p.get("risk_tolerance","").lower() in {"conservative","low"}) and (c.get("beta",1.0) > 1.3),"action":"block"},
    {"id":"flag_litigation_terms","desc":"Flag if litigation/regulatory terms appear","when": lambda c,p: any(k in (c.get("logical_facts") or {}) for k in ["litigation","regulatory"]),"action":"flag"},
    {"id":"boost_profitable","desc":"Boost if profitable (eps>0 & PE>0)","when": lambda c,p: (c.get("eps") or 0) > 0 and (c.get("pe_ratio") or 0) > 0,"action":"boost:0.03"},
]

def apply_rules(company: dict, profile: dict) -> dict:
    flags = []; blocked=False; boost=0.0
    for r in RULES:
        try:
            if r["when"](company, profile or {}):
                if r["action"]=="block": blocked=True
                elif r["action"].startswith("boost:"):
                    try: boost += float(r["action"].split(":")[1])
                    except Exception: pass
                flags.append(r["id"])
        except Exception:
            continue
    return {"blocked":blocked, "flags":flags, "boost":boost}

def numeric_verification(c: dict) -> dict:
    try:
        price = float(c.get("price") or 0)
        pe = float(c.get("pe_ratio") or 0)
        eps = c.get("eps")
        if eps is not None:
            eps = float(eps)
            if eps != 0:
                pe_calc = price / max(1e-9, eps)
                if pe > 0 and 0.4 <= pe_calc/pe <= 2.5:
                    return {"numeric_ok": True, "numeric_err": ""}
                else:
                    return {"numeric_ok": False, "numeric_err": f"PE sanity mismatch (calc≈{pe_calc:.1f}, pe={pe:.1f})"}
        if pe >= 0:
            return {"numeric_ok": True, "numeric_err": ""}
        return {"numeric_ok": False, "numeric_err": "negative PE without EPS context"}
    except Exception as e:
        return {"numeric_ok": False, "numeric_err": f"exception {e}"}

# ---- Provenance gate (added) ----
def _meets_provenance_gate(c: dict) -> bool:
    p = c.get("provenance", []) or []
    return any((x or {}).get("type") in {"news", "qual", "logic", "numeric"} for x in p)

def _norm01(x: float, lo: float, hi: float) -> float:
    try:
        if hi <= lo: return 0.0
        v = (float(x) - lo) / (hi - lo)
        return max(0.0, min(1.0, v))
    except Exception:
        return 0.0

def mcdm_score(c: dict, w: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    pe = float(c.get("pe_ratio") or 0.0)
    beta = float(c.get("beta") or 1.0)
    vol = float(c.get("volatility") or 0.3)
    price = float(c.get("price") or 0.0)
    sma20 = float(c.get("sma_20") or price)
    rsi = float(c.get("rsi") or 50.0)
    mcap = float(c.get("market_cap") or 0.0)
    sent = float(c.get("sentiment") or 0.0)

    valuation = 0.0
    if pe <= 0: valuation = 0.6
    elif pe < 12: valuation = 1.0
    elif pe < 18: valuation = 0.8
    elif pe < 25: valuation = 0.6
    elif pe < 35: valuation = 0.4
    else: valuation = 0.2

    risk = 0.0
    risk += (1.0 - _norm01(beta, 0.6, 1.6)) * 0.5
    risk += (1.0 - _norm01(vol, 0.15, 0.6)) * 0.5
    risk = max(0.0, min(1.0, risk))

    momentum = 0.0
    if price and sma20:
        if price > sma20: momentum += 0.6
        if 45 <= rsi <= 65: momentum += 0.3
        elif 35 <= rsi < 45 or 65 < rsi <= 75: momentum += 0.15
    momentum = max(0.0, min(1.0, momentum))

    size = _norm01(mcap, 5e9, 300e9)
    sentiment = _norm01(sent, -0.6, 0.6)

    feats = {"valuation": round(valuation,3), "risk": round(risk,3), "momentum": round(momentum,3),
             "size": round(size,3), "sentiment": round(sentiment,3)}
    score = sum(feats[k] * float(w.get(k, 0.0)) for k in feats)
    return float(max(0.0, min(1.0, score))), feats

# -------------------------- Translation utils ----------------------------
SUPPORTED_LANGUAGES = {"bn":"Bengali","es":"Spanish","de":"German","fr":"French","it":"Italian","pt":"Portuguese","en":"English"}

@lru_cache(maxsize=20)
def get_translator(source_lang: str, target_lang: str):
    if pipeline is None:
        return None
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        return pipeline("translation", model=model_name, max_length=512)
    except Exception as e:
        logger.warning(f"Translation model loading failed: {e}")
        return None

def translate_input_to_english(text: str) -> tuple[str, str]:
    if detect is None:
        return text, "en"
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text, "en"
        elif detected_lang in SUPPORTED_LANGUAGES:
            translator = get_translator(detected_lang, "en")
            if translator:
                translated = translator(text, max_length=512)[0]["translation_text"]
                return translated, detected_lang
        return text, "en"
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text, "en"

def translate_output_from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    try:
        translator = get_translator("en", target_lang)
        if translator:
            translated = translator(text, max_length=512)[0]["translation_text"]
            return translated
    except Exception as e:
        logger.warning(f"Output translation failed: {e}")
    return text

# --------------------------- PDF / Web helpers ---------------------------
def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text[:3000]
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return f"Error reading PDF: {str(e)}"

def web_search_ddg(query: str, max_results: int = 3) -> str:
    try:
        results = _safe_ddg_text(query, max_results=max_results)
        if not results:
            return "No web search results found."
        out = "Recent web search results:\n\n"
        for i, r in enumerate(results, 1):
            out += f"{i}. {r.get('title','')}\n"
            out += f"   {(r.get('body','')[:200])}...\n"
            out += f"   Source: {r.get('href','')}\n\n"
        return out
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return "Web search temporarily unavailable."

# ----------------------------- DB: Portfolios ----------------------------
class SecurePortfolioManager:
    def __init__(self):
        self.db_path = "secure_portfolios.db"
        self.init_database()

    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def init_database(self):
        try:
            with self.get_db_connection() as conn:
                conn.execute("""
CREATE TABLE IF NOT EXISTS portfolios (
id INTEGER PRIMARY KEY AUTOINCREMENT,
user_id TEXT NOT NULL,
symbol TEXT NOT NULL,
shares REAL NOT NULL CHECK (shares > 0),
purchase_price REAL NOT NULL CHECK (purchase_price > 0),
purchase_date TEXT NOT NULL,
risk_tolerance TEXT CHECK (risk_tolerance IN ('Low','Medium','High')),
investment_goals TEXT,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
                conn.execute("""
CREATE TABLE IF NOT EXISTS user_profiles (
user_id TEXT PRIMARY KEY,
age INTEGER CHECK (age >= 18 AND age <= 100),
risk_tolerance TEXT CHECK (risk_tolerance IN ('Conservative','Moderate','Aggressive')),
investment_timeline TEXT,
income_level TEXT,
experience_level TEXT,
financial_goals TEXT,
created_date TEXT,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
                conn.execute("CREATE TABLE IF NOT EXISTS companies (symbol TEXT PRIMARY KEY, name TEXT, sector TEXT, market_cap REAL, pe_ratio REAL, price REAL, risk_level TEXT, beta REAL)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_symbol ON portfolios(symbol)")
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def add_to_portfolio(self, user_id: str, symbol: str, shares: float, price: float) -> bool:
        try:
            user_id = SecurityManager.validate_user_id(user_id)
            symbol = re.sub(r"[^A-Z.\-]", "", symbol.upper())
            if shares <= 0 or price <= 0:
                raise ValueError("Shares and price must be positive")
            with self.get_db_connection() as conn:
                conn.execute("""
INSERT INTO portfolios
(user_id, symbol, shares, purchase_price, purchase_date, risk_tolerance, investment_goals)
VALUES (?, ?, ?, ?, date('now'), ?, ?)
""", (user_id, symbol, shares, price, "Medium", "Growth"))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Portfolio add failed: {e}")
            return False

    def get_portfolio_performance(self, user_id: str) -> dict:
        try:
            user_id = SecurityManager.validate_user_id(user_id)
            with self.get_db_connection() as conn:
                positions = conn.execute("""
SELECT symbol, shares, purchase_price FROM portfolios WHERE user_id = ? ORDER BY created_at DESC
""", (user_id,)).fetchall()
                total_value=0.0; total_cost=0.0; performance=[]
                for symbol, shares, purchase_price in positions:
                    try:
                        close = fetch_history(symbol, days=7)["Close"].dropna()
                        current_price = float(close.iloc[-1]) if len(close) else None
                        if current_price is None:
                            continue
                        position_value = shares * current_price
                        position_cost = shares * purchase_price
                        pnl = position_value - position_cost
                        pnl_pct = (pnl / position_cost) * 100 if position_cost > 0 else 0
                        performance.append({
                            "symbol": symbol, "shares": shares, "current_price": round(current_price,2),
                            "purchase_price": purchase_price, "current_value": round(position_value,2),
                            "pnl": round(pnl,2), "pnl_pct": round(pnl_pct,2)
                        })
                        total_value += position_value; total_cost += position_cost
                    except Exception as e:
                        logger.warning(f"Failed to get price for {symbol}: {e}")
                        continue
                return {
                    "positions": performance, "total_value": round(total_value,2), "total_cost": round(total_cost,2),
                    "total_pnl": round(total_value - total_cost,2),
                    "total_return_pct": round(((total_value - total_cost)/total_cost*100),2) if total_cost>0 else 0
                }
        except Exception as e:
            logger.error(f"Portfolio performance calculation failed: {e}")
            return {"positions": [], "total_value": 0, "total_cost": 0, "total_pnl": 0, "total_return_pct": 0}

# ----------------------------- Agent state -------------------------------
class FinancialAgentState(TypedDict):
    query: str
    user_profile: Dict
    user_requirements: Dict
    decomposed: Dict
    sql_queries: List[str]
    company_data: List[Dict]
    market_analysis: str
    fundamental_analysis: str
    technical_analysis: List[Dict]
    recommendations: List[Dict]
    final_response: str
    budget_t0: float  # added for time budget

# --------------------------- Query Decomposition -------------------------
def query_decomposition_agent(state: FinancialAgentState):
    q = state["query"]
    if not AGENT_DISABLE_LLM:
        prompt = (
            "Analyze the user's intent for stock selection. "
            "Return STRICT JSON with keys: risk_profile (conservative|moderate|aggressive), "
            "sectors (list of strings), keywords (list of strings). "
            f'User query: "{q}"'
        )
        try:
            resp = llm_call("analysis", [{"role": "user", "content": prompt}], max_tries=1).content.strip()
            m = re.search(r"\{.*\}", resp, re.S)
            if m:
                parsed = json.loads(m.group(0))
                state["decomposed"] = {
                    "risk_profile": str(parsed.get("risk_profile","moderate")).lower(),
                    "sectors": [str(s) for s in parsed.get("sectors", [])],
                    "keywords": [str(s) for s in parsed.get("keywords", [])],
                }
                return state
        except Exception:
            pass

    ql = q.lower()
    risk = "conservative" if any(w in ql for w in ["safe","low risk","conservative","stable"]) else (
           "aggressive" if any(w in ql for w in ["aggressive","high risk","growth"]) else "moderate")
    sectors = []
    for s, ks in {
        "technology": ["tech","technology","software","semis","semiconductor","it"],
        "healthcare": ["health","healthcare","medical","pharma","biotech"],
        "finance": ["bank","finance","financial","insurance"],
        "energy": ["energy","oil","gas"],
        "consumer": ["consumer","retail","ecommerce","auto","staples"],
        "industrials": ["industrial","manufacturing","aerospace","defense"],
        "utilities": ["utility","utilities","power","electric"],
        "materials": ["materials","mining","chemicals"],
        "real estate": ["reit","real estate","property"],
        "communication": ["telecom","media","social","communication"],
    }.items():
        if any(k in ql for k in ks):
            sectors.append(s)
    keywords = re.findall(r"[a-zA-Z\-]{3,}", ql)
    state["decomposed"] = {"risk_profile": risk, "sectors": sectors, "keywords": keywords[:10]}
    return state

# ------------------------------ LLM agents --------------------------------
def generate_sql_agent(state: FinancialAgentState):
    query = state["query"]
    user_profile = state.get("user_profile", {})
    sql_prompt = f"""
Generate 2-3 read-only SQL queries for financial analysis (SQLite syntax).

Context:

Query: "{query}"
User Profile: Risk: {user_profile.get('risk_tolerance','Moderate')}, Timeline: {user_profile.get('investment_timeline','Medium-term')}
Schema:

companies(symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta)
portfolios(user_id, symbol, shares, purchase_price, purchase_date)
Constraints:

SINGLE SELECT only
No JOINs outside listed tables
No comments
Return queries fenced in sql blocks.
""".strip()
    try:
        resp = llm_call("sql", [{"role":"user","content":sql_prompt}], max_tries=2)
        content = resp.content or ""
        sql_queries = re.findall(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if not sql_queries:
            sql_queries = ["SELECT * FROM companies ORDER BY market_cap DESC LIMIT 10"]
        state["sql_queries"] = sql_queries[:3]
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        state["sql_queries"] = ["SELECT * FROM companies ORDER BY market_cap DESC LIMIT 10"]
    return state

def data_integration_agent(state: FinancialAgentState):
    query = state["query"]; sql_queries = state.get("sql_queries", [])
    db_path = "secure_portfolios.db"; all_results: List[pd.DataFrame] = []
    try:
        with sqlite3.connect(db_path) as conn:
            for sql_query in sql_queries:
                try:
                    safe_query = validate_sql_ast(sql_query)
                    df = pd.read_sql_query(safe_query, conn)
                    if not df.empty: all_results.append(df)
                except Exception as e:
                    logger.error(f"SQL execution failed: {e}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    if all_results:
        final_df = pd.concat(all_results).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        company_data = [normalize_company_keys(r) for r in final_df.to_dict("records")]
    else:
        logger.warning("SQL yielded no results, falling back to live screening.")
        company_data = [normalize_company_keys(x) for x in get_stocks_by_real_criteria(query)]

    # enrich with live metrics (already batched inside)
    syms = [c["symbol"] for c in company_data][:10]
    live = {c["symbol"]: c for c in get_real_stock_data_parallel(syms)}
    company_data = [{**c, **(live.get(c["symbol"], {}))} for c in company_data]

    # Market narrative (best-effort)
    analysis_prompt = f"""
Provide a concise (2-3 sentences) market snapshot for these companies:
{[{'name': c['name'], 'sector': c['sector'], 'pe': c.get('pe_ratio','N/A'), 'mcap': c.get('market_cap',0)} for c in company_data[:3]]}
Discuss sector trends and risk.
"""
    try:
        analysis_response = llm_call("analysis", [{"role":"user","content":analysis_prompt}], max_tries=1)
        state["market_analysis"] = analysis_response.content.strip() or "—"
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        state["market_analysis"] = "—"

    state["company_data"] = company_data[:6]
    return state

# ------------------------------- RAG agents -------------------------------
def qualitative_retriever_agent(state: FinancialAgentState):
    # skip if budget exceeded (added)
    if _over_budget(state):
        return state

    syms = [c["symbol"] for c in state.get("company_data", [])][:5]
    urls = set()
    decomp = state.get("decomposed", {})
    kw = " ".join(decomp.get("keywords", [])[:5]) if decomp else ""
    logger.info(f"QUAL RAG: Starting search for symbols: {syms}")

    for s in syms:
        try:
            for r in _safe_ddg_text(f"{s} investment thesis analysis {kw}", 5):
                if (r.get("href") or "").startswith("http"):
                    urls.add(r["href"])
        except Exception as e:
            logger.warning(f"QUAL RAG DDG failed for {s}: {e}")
            continue
    logger.info(f"QUAL RAG: Found {len(urls)} unique URLs to process.")

    try:
        add_from_urls_text(list(urls)[:12])
    except Exception as e:
        logger.error(f"QUAL RAG: Failed to add URLs. Error: {e}")

    enriched = []
    for c in state.get("company_data", []):
        prov = list(c.get("provenance", []))
        try:
            hits = retrieve_text(f"{c.get('name') or c['symbol']} investment thesis news", k=3)
            if not hits:
                prov.append({"type": "qual", "title": "No qualitative documents found", "url": "", "date": ""})
            for h in hits[:2]:
                prov.append({"type":"qual","title":h["title"],"url":h["url"],"date":""})
            enriched.append({**c, "provenance": prov})
        except Exception as e:
            logger.warning(f"QUAL RAG retrieve failed for {c.get('symbol')}: {e}")
            enriched.append({**c, "provenance": prov})
    state["company_data"] = enriched
    return state

def logical_retriever_agent(state: FinancialAgentState):
    # skip if budget exceeded (added)
    if _over_budget(state):
        return state

    syms = [c["symbol"] for c in state.get("company_data", [])][:5]
    urls = set()
    logger.info(f"LOGIC RAG: Starting search for symbols: {syms}")

    for s in syms:
        queries = [
            f"{s} press release acquisition OR merger OR financing",
            f"{s} site:sec.gov 8-K OR 10-K OR 10-Q",
        ]
        for q in queries:
            try:
                for r in _safe_ddg_text(q, 5):
                    href = (r.get("href") or "").strip()
                    if href.startswith("http"):
                        urls.add(href)
            except Exception as e:
                logger.warning(f"LOGIC RAG DDG failed for {s}: {e}")
                continue

    logger.info(f"LOGIC RAG: Found {len(urls)} unique URLs to process.")
    try:
        add_from_urls_logic(list(urls)[:15])
    except Exception as e:
        logger.error(f"LOGIC RAG add failed: {e}")

    enriched = []
    for c in state.get("company_data", []):
        prov = list(c.get("provenance", []))
        try:
            name = c.get("name") or c["symbol"]
            hits = retrieve_logic(f"{name} merger acquisition financing litigation regulatory", k=5)
            snippets = [h.get("snippet", "") for h in hits]
            logical = extract_logical_from_hits(snippets)
            if not hits:
                prov.append({"type":"logic","title":"No regulatory/logical docs found","url":"","date":""})
            for h in hits[:2]:
                prov.append({"type":"logic","title":h.get("title",""),"url":h.get("url",""),"date":""})
            enriched.append({**c, "logical_facts": logical, "provenance": prov})
        except Exception as e:
            logger.warning(f"LOGIC RAG retrieve failed for {c.get('symbol')}: {e}")
            enriched.append({**c, "provenance": prov})
    state["company_data"] = enriched
    return state


def numeric_retriever_agent(state: FinancialAgentState):
    # skip if budget exceeded (added)
    if _over_budget(state):
        return state

    syms = [c["symbol"] for c in state.get("company_data", [])][:5]
    try:
        ingest_numeric_for_symbols(syms)
    except Exception as e:
        logger.warning(f"NUM RAG ingest failed: {e}")

    enriched = []
    for c in state.get("company_data", []):
        prov2 = list(c.get("provenance", []))
        try:
            num_hits = numeric_retrieve("revenue eps free cash flow debt", [c["symbol"]], topk=5)
            if not num_hits:
                prov2.append({"type":"numeric","title":"No numeric facts found","url":"","date":""})
            numeric_facts=[]
            for nh in num_hits:
                numeric_facts.append({"metric": nh["metric"], "period": nh["period"], "value": nh["value"]})
                prov2.append({"type":"numeric","title": nh.get("title",""), "url": nh.get("url",""), "date": nh.get("period","")})
            eps_guess = next((f["value"] for f in numeric_facts if "eps" in f["metric"]), None)
            enriched.append({**c, "numeric_facts": numeric_facts, "eps": c.get("eps") or eps_guess, "provenance": prov2})
        except Exception as e:
            logger.warning(f"NUM RAG retrieve failed for {c.get('symbol')}: {e}")
            enriched.append({**c, "provenance": prov2})
    state["company_data"] = enriched
    return state

# -------------------------- Verifier / Recommender -----------------------
def verifier_agent(state: FinancialAgentState):
    enriched=[]
    for c in state.get("company_data", []):
        vr = numeric_verification(c)
        prov = c.get("provenance", [])
        news_n = sum(1 for p in prov if p.get("type")=="news")
        qual_n = sum(1 for p in prov if p.get("type")=="qual")
        logic_n= sum(1 for p in prov if p.get("type")=="logic")
        num_n = sum(1 for p in prov if p.get("type")=="numeric")
        conf = (
            0.25*(1.0 if vr["numeric_ok"] else 0.3)
            + 0.25*min(1.0, news_n/4.0)
            + 0.25*min(1.0, (qual_n+logic_n)/4.0)
            + 0.25*min(1.0, num_n/3.0)
        )
        enriched.append({**c, "verify_numeric_ok": vr["numeric_ok"], "verify_error": vr["numeric_err"], "confidence": round(conf,2)})
    state["company_data"] = enriched
    return state

class Recommendation(BaseModel):
    company: str
    symbol: str
    sector: str
    price: float = Field(ge=0)
    pe_ratio: float | None = None
    market_cap: float | None = None
    risk: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    features: Dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    flags: List[str] = []
    rationale: str
    sources: List[Dict[str, str]] = []

def validate_recommendations(items: List[dict]) -> List[dict]:
    out=[]
    for it in items:
        try:
            v = Recommendation(**it)
            out.append(v.dict())
        except ValidationError:
            continue
    return out

def _defensive_reason(c: dict) -> str:
    bits=[]
    if (c.get("beta") or 1.0) < 1.0: bits.append("below-market β")
    dy = c.get("dividend_yield")
    if isinstance(dy, (int,float)) and dy > 0.015: bits.append("dividend cushion")
    if c.get("trend") == "Bullish": bits.append("supportive trend")
    pe = c.get("pe_ratio")
    if pe is not None and 0 < pe < 25: bits.append("non-stretched multiple")
    return (", ".join(bits) or "balanced risk profile") + "."

def recommendation_agent(state: FinancialAgentState):
    companies = state["company_data"]
    query = state["query"]
    weights = state.get("user_requirements", {}).get("weights") or {"valuation":0.25,"risk":0.30,"momentum":0.20,"size":0.15,"sentiment":0.10}
    s = sum(weights.values()) or 1.0
    weights = {k: float(v)/s for k,v in weights.items()}
    profile = state.get("user_profile", {})
    risk_label = (profile.get("risk_tolerance") or state.get("decomposed", {}).get("risk_profile", "Moderate")).title()

    ranked=[]; excluded=[]
    for c in companies[:8]:
        prov = c.get("provenance", [])
        gate = check_constraints(c, profile)
        rules = apply_rules(c, profile)
        if rules["blocked"]:
            excluded.append({"symbol": c.get("symbol"), "reason": "policy_block", "flags": rules["flags"]})
            continue

        score, feats = mcdm_score(c, weights)
        score = max(0.0, min(1.0, score + rules["boost"] - (0.10 if not gate["allowed"] else 0.0)))

        news = [p for p in prov if p.get("type")=="news"][:2]
        qual = [p for p in prov if p.get("type")=="qual"][:1]
        logic= [p for p in prov if p.get("type")=="logic"][:1]
        num  = [p for p in prov if p.get("type")=="numeric"][:1]
        sources = news + qual + logic + num

        # provenance gate (added): downweight/flag if no evidence
        gate_ok = _meets_provenance_gate(c)
        if not gate_ok:
            score = max(0.0, score - 0.10)
            rules_flags = list(set((gate["flags"] or []) + rules["flags"] + ["low_provenance"]))
        else:
            rules_flags = list(set((gate["flags"] or []) + rules["flags"]))

        rationale = (
            f"PE={c.get('pe_ratio')} • Beta={c.get('beta')} • Vol≈{c.get('volatility')} • "
            f"RSI={c.get('rsi')} • Trend={c.get('trend')} • Sent={c.get('sentiment',0):+.2f} • "
            f"Conf={c.get('confidence',0):.2f}"
        )
        ranked.append({
            "company": c["name"], "symbol": c["symbol"], "sector": c["sector"],
            "price": c["price"], "pe_ratio": c.get("pe_ratio"), "market_cap": c.get("market_cap"),
            "risk": c.get("risk"), "score": score, "features": feats, "confidence": c.get("confidence",0.0),
            "flags": rules_flags, "rationale": rationale,
            "dividend_yield": c.get("dividend_yield"),
            "rsi": c.get("rsi"), "trend": c.get("trend"), "beta": c.get("beta"), "volatility": c.get("volatility"),
            "sources": [{"title":s.get("title",""),"url":s.get("url",""),"date":s.get("date","")} for s in sources if s.get("url","") or s.get("title","")],
        })

    ranked = validate_recommendations(ranked)
    ranked.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
    top = ranked[:3]

    # Friendly, investor-style rendering (no heavy scaffolding)
    lines = [f"### Safe stock suggestions ({risk_label})", ""]
    for r in top:
        cap = r.get("market_cap") or 0
        cap_str = f"${cap/1e9:.0f}B" if cap else "—"
        pe = r.get("pe_ratio")
        pe_str = f"{pe:.1f}x" if isinstance(pe,(int,float)) and pe>0 else "—"
        dy = r.get("dividend_yield")
        dy_str = f"{dy*100:.1f}%" if isinstance(dy,(int,float)) and dy>0 else "—"
        beta = r.get("beta") or 1.0
        sigma20d = (r.get("volatility") or 0.0) * math.sqrt(20/252) * 100.0

        lines.append(f"    {r['company']} ({r['symbol']}) — ${r['price']:.2f} · score {r['score']:.3f} · conf {r['confidence']:.2f}")
        lines.append(f"        Profile: {r.get('sector','—')} • β≈{beta:.2f} • DY≈{dy_str}")
        lines.append(f"        Valuation: P/E≈{pe_str} • Cap≈{cap_str}")
        lines.append(f"        Momentum & risk: RSI {r.get('rsi',50)} • Trend {r.get('trend','—')} • σ20d≈{sigma20d:.1f}%")
        lines.append(f"        Why it screens as defensive: {_defensive_reason(r)}")
        lines.append("")  # blank line between entries

    state["recommendations"] = top
    state["final_response"] = "\n".join(lines).rstrip()
    return state

# ----------------------------- Workflow graph ----------------------------
from langgraph.graph import StateGraph, END

def create_financial_workflow():
    workflow = StateGraph(FinancialAgentState)
    workflow.add_node("decompose", query_decomposition_agent)
    workflow.add_node("sql_agent", generate_sql_agent)
    workflow.add_node("data_agent", data_integration_agent)
    workflow.add_node("qual_retriever", qualitative_retriever_agent)
    workflow.add_node("logic_retriever", logical_retriever_agent)
    workflow.add_node("numeric_retriever", numeric_retriever_agent)
    workflow.add_node("verifier_agent", verifier_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)

    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "sql_agent")
    workflow.add_edge("sql_agent", "data_agent")
    workflow.add_edge("data_agent", "qual_retriever")
    workflow.add_edge("qual_retriever", "logic_retriever")
    workflow.add_edge("logic_retriever", "numeric_retriever")
    workflow.add_edge("numeric_retriever", "verifier_agent")
    workflow.add_edge("verifier_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    return workflow.compile()

# --------------------------- Intent & formatting --------------------------
def is_financial_query(query: str) -> bool:
    q = query.lower()
    financial_keywords = ["stock","portfolio","market","buy","sell","recommendation","risk","return","shares","p/e","market cap","dividend","investment","ticker"]
    if any(k in q for k in financial_keywords): return True
    if re.search(r"\b[A-Z]{1,5}\b", query): return True
    if "company" in q or "safe" in q:
        return any(k in q for k in ["stock","investment","p/e","market"])
    return False

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["buy","add","purchase"]):
        if re.search(r"buy\s+([\d.,]+)\s+shares?\s+of\s+([A-Za-z.\-]{1,7})", query, re.IGNORECASE):
            return "add_to_portfolio"
    if re.search(r"\b((my\s+)?portfolio|holdings|performance)\b", q):
        return "view_portfolio"
    if is_financial_query(query): return "financial_analysis"
    return "general_query"

def format_portfolio_response(perf: dict) -> str:
    if not perf or not perf.get("positions"):
        return "Your portfolio is currently empty. Try: **buy 10 shares of AAPL**."
    lines = ["### Your Portfolio Performance", ""]
    lines.append(f"**Total Value:** ${perf['total_value']:,.2f}")
    lines.append(f"**Total Cost Basis:** ${perf['total_cost']:,.2f}")
    lines.append(f"**Total P&L:** ${perf['total_pnl']:,.2f} ({perf['total_return_pct']:+.2f}%)")
    lines.append("")
    lines.append("#### Individual Holdings")
    for p in perf["positions"]:
        pnl_emoji = "📈" if p["pnl"] > 0 else ("📉" if p["pnl"] < 0 else "➖")
        cp = p.get("current_price")
        cp_str = f"{cp:.2f}" if cp is not None else "—"
        lines.append(f"**{p['symbol']}** — {p['shares']} shares")
        lines.append(f"• Current: ${cp_str} | Purchase: ${p['purchase_price']:.2f}")
        lines.append(f"• Value: ${p['current_value']:,.2f} | P&L: {pnl_emoji} ${p['pnl']:+,.2f} ({p['pnl_pct']:+.2f}%)")
        lines.append("")
    return "\n".join(lines)

def validate_query(query: str) -> str:
    if not query or len(query.strip()) == 0: raise ValueError("Query cannot be empty")
    if len(query) > 2000: raise ValueError("Query too long (max 2000 characters)")
    q = query.strip()
    q = re.sub(r"<[^>]+>", "", q)
    q = re.sub(r"javascript:", "", q, flags=re.IGNORECASE)
    return q

def handle_general_query_with_search(query: str, context: str, chat_history: List[Dict[str, str]], file_path: Optional[str] = None) -> str:
    should_search = any(k in query.lower() for k in ["company","startup","recent","news","current","today","latest","what is","who is","tell me about"])
    web_results = web_search_ddg(query, max_results=3) if should_search else ""
    history_text = "\n".join(f"User: {m['user']}\nAgent: {m['agent']}" for m in chat_history[-3:])
    file_summary = extract_text_from_pdf(file_path) if file_path else ""

    context_str = f"Previous conversation context:\n{context}\n" if context else ""
    history_str = f"Recent chat history:\n{history_text}\n" if history_text else ""
    file_str = f"File content summary:\n{file_summary}\n" if file_summary else ""
    web_str = f"Current web search results:\n{web_results}\n" if web_results else ""

    prompt = f"""
You are a helpful AI assistant. Provide clear, accurate responses grounded in sources if given.

{context_str}{history_str}{file_str}{web_str}
User's question: {query}

Instructions:

Use search results if available and cite with URLs.
Be direct and concise.
Answer:
""".strip()
    try:
        return llm_call("analysis", [{"role":"user","content":prompt}], max_tries=1).content
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return "I’m having trouble answering that right now."

# ----------------------------- Public entrypoint -------------------------
portfolio_manager = SecurePortfolioManager()

def initialize_global_instances():
    global rate_limiter, conversation_memory
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    if conversation_memory is None:
        conversation_memory = ConversationMemory()

def run_customer_support(
    query: str,
    user_profile: Optional[Dict] = None,
    user_requirements: Optional[Dict] = None,
    force_language: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    file_path: Optional[str] = None,
    user_id: str = "anonymous_user",
) -> Dict[str, Any]:
    if chat_history is None: chat_history = []
    if user_profile is None: user_profile = {}
    if user_requirements is None: user_requirements = {}

    initialize_global_instances()
    allowed, remaining = rate_limiter.is_allowed(user_id)
    if not allowed:
        return {"original_language": "English","response":"⚠️ Rate limit exceeded. Please try again later.","raw_response":"Rate limit exceeded"}

    try:
        query = validate_query(query)
    except ValueError as e:
        return {"original_language":"English","response":f"❌ Invalid input: {str(e)}","raw_response":str(e)}

    if detect is None:
        translated_query, detected_lang = (query, "en")
    else:
        translated_query, detected_lang = translate_input_to_english(query)
    context = conversation_memory.get_context_summary(user_id)
    intent = detect_intent(translated_query)

    if intent == "add_to_portfolio":
        m = re.search(r"buy\s+([\d.,]+)\s+shares?\s+of\s+([A-Za-z.\-]{1,7})", translated_query, re.IGNORECASE)
        if m:
            shares_str, symbol = m.groups()
            try:
                close = fetch_history(symbol.upper(), days=7)["Close"].dropna()
                current_price = float(close.iloc[-1]) if len(close) else None
                if current_price is None: raise ValueError("price unavailable")
                shares = float(shares_str.replace(",", ""))
                success = portfolio_manager.add_to_portfolio(user_id, symbol.upper(), shares, current_price)
                response = f"✅ Added {shares:g} shares of {symbol.upper()} at ${current_price:.2f} per share." if success else "❌ Failed to add to portfolio. Please try again."
            except Exception:
                response = f"❌ Error: Could not fetch a reliable price for '{symbol}'."
        else:
            response = "Please use: **buy [number] shares of [SYMBOL]** (e.g., *buy 10 shares of AAPL*)."

    elif intent == "view_portfolio":
        performance_data = portfolio_manager.get_portfolio_performance(user_id)
        response = format_portfolio_response(performance_data)

    elif intent == "financial_analysis":
        workflow = create_financial_workflow()
        initial_state: FinancialAgentState = {
            "query": translated_query,
            "user_profile": user_profile,
            "user_requirements": user_requirements,
            "decomposed": {},
            "sql_queries": [], "company_data": [], "market_analysis": "", "fundamental_analysis": "",
            "technical_analysis": [], "recommendations": [], "final_response": "",
            "budget_t0": time.time(),  # start overall budget timer (added)
        }
        try:
            result = workflow.invoke(initial_state)
            response = result.get("final_response", "Analysis complete, but no response was generated.")
        except Exception as e:
            logger.error(f"Financial workflow failed: {e}")
            response = "Sorry, the      financial analysis system is temporarily unavailable. Please try again later."
    else:
        response = handle_general_query_with_search(translated_query, context, chat_history, file_path)

    conversation_memory.add_exchange(user_id, query, response)
    target_lang = force_language if force_language else detected_lang
    translated_response = translate_output_from_english(response, target_lang)

    return {"original_language": SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
            "response": translated_response, "raw_response": response, "remaining_requests": remaining}
