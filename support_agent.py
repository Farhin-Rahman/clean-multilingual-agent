from typing import TypedDict, Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate  # (kept for future use)
from functools import lru_cache
from transformers import pipeline
from langdetect import detect
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import fitz  # PyMuPDF
import os
import re
import yfinance as yf
import pandas as pd
import sqlite3
import requests  # optional; handy for future endpoints

# optional providers; keep app running even if not installed
try:
    import finnhub  # type: ignore
except Exception:
    finnhub = None

try:
    from fredapi import Fred  # type: ignore
except Exception:
    Fred = None

from datetime import datetime, timedelta
import numpy as np
import logging
import hashlib
import time
from contextlib import contextmanager
from duckduckgo_search import DDGS

load_dotenv()

# ─────────────────────────── Constants / logging ───────────────────────────
PRICE_BUCKET_SECONDS = 600
SP500_CACHE_FILE = "sp500_cache.csv"
SP500_CACHE_TTL_HOURS = 24
SP500_FULL_CACHE_FILE = "sp500_full_cache.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate API keys
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Optional API keys (graceful degradation)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

SUPPORTED_LANGUAGES = {
    "bn": "Bengali",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "en": "English",
}

# Globals (initialized later)
rate_limiter = None
conversation_memory = None
portfolio_manager = None

# ───────────────────────────── Security / utils ────────────────────────────
class SecurityManager:
    """Production-grade security management"""

    @staticmethod
    def allowlist_sql(query: str) -> str:
        """Read-only SQL allow-list: SELECT only, no comments, single statement."""
        q = (query or "").strip()
        if re.search(r"--|/\*|\*/", q):
            raise ValueError("SQL comments are not allowed.")
        q = q.replace(";", "")
        if not re.match(r"^\s*select\b", q, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed.")
        return q

    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate and sanitize user ID"""
        if not user_id or len(user_id) < 1:
            return "anonymous_user"
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", user_id)[:50]
        return sanitized or "anonymous_user"

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive information"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class RateLimiter:
    """Production-grade rate limiting"""

    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.limits = {"free": {"requests": 100, "window": 3600}}  # 100/hr

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        now = time.time()
        user_key = f"{user_id}_free"  # Always use free tier
        if user_key not in self.requests:
            self.requests[user_key] = []
        window = self.limits["free"]["window"]
        self.requests[user_key] = [
            t for t in self.requests[user_key] if now - t < window
        ]
        current_count = len(self.requests[user_key])
        limit = self.limits["free"]["requests"]
        if current_count < limit:
            self.requests[user_key].append(now)
            return True, limit - current_count - 1
        return False, 0


class ConversationMemory:
    """Enhanced conversation memory management"""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}

    def add_exchange(
        self, user_id: str, user_msg: str, agent_msg: str, context: dict | None = None
    ):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "agent": agent_msg,
            "context": context or {},
            "tokens": len(user_msg.split()) + len(agent_msg.split()),
        }
        self.conversations[user_id].append(exchange)
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][
                -self.max_history :
            ]

    def get_context_summary(self, user_id: str, max_tokens: int = 1000) -> str:
        if user_id not in self.conversations:
            return ""
        recent_exchanges = self.conversations[user_id][-5:]
        total_tokens = 0
        parts: List[str] = []
        for ex in reversed(recent_exchanges):
            if total_tokens + ex["tokens"] > max_tokens:
                break
            parts.append(
                f"User: {ex['user']}\nAssistant: {ex['agent'][:200]}..."
            )
            total_tokens += ex["tokens"]
        parts.reverse()
        fin = self.extract_financial_preferences(user_id)
        if fin:
            parts.append(f"\nUser Financial Profile: {fin}")
        return "\n---\n".join(parts)

    def extract_financial_preferences(self, user_id: str) -> str:
        if user_id not in self.conversations:
            return ""
        prefs = {"risk_tolerance": None, "sectors": []}
        all_text = " ".join(
            [ex["user"] + " " + ex["agent"] for ex in self.conversations[user_id]]
        ).lower()
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


class RealFinancialData:
    def __init__(self):
        self.finnhub_client = (
            finnhub.Client(api_key=FINNHUB_API_KEY)
            if (finnhub and FINNHUB_API_KEY)
            else None
        )
        self.fred = Fred(api_key=FRED_API_KEY) if (Fred and FRED_API_KEY) else None

    def get_comprehensive_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        try:
            if self.finnhub_client:
                data["financials"] = self.finnhub_client.company_basic_financials(
                    symbol, "all"
                )
                data["news"] = self.finnhub_client.company_news(
                    symbol,
                    _from=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                    to=datetime.now().strftime("%Y-%m-%d"),
                )
                data["recommendations"] = self.finnhub_client.recommendation_trends(
                    symbol
                )
                data["earnings"] = self.finnhub_client.company_earnings(
                    symbol, limit=4
                )
            if self.fred:
                end = datetime.today().date()
                start = end - timedelta(days=365 * 2)
                fed_data = self.fred.get_series(
                    "FEDFUNDS", observation_start=start, observation_end=end
                )
                cpi = self.fred.get_series(
                    "CPIAUCSL", observation_start=start, observation_end=end
                )
                data["interest_rates"] = (
                    float(fed_data.dropna().iloc[-1])
                    if fed_data is not None and len(fed_data.dropna())
                    else 5.25
                )
                data["inflation"] = (
                    float(cpi.pct_change().dropna().iloc[-1] * 100)
                    if cpi is not None and len(cpi)
                    else 3.2
                )
        except Exception as e:
            logger.warning(f"API data fetch failed for {symbol}: {e}")
            data = {"error": str(e)}
        return data


class SecurePortfolioManager:
    """Production-grade portfolio management"""

    def __init__(self):
        self.db_path = "secure_portfolios.db"
        self.security = SecurityManager()
        self.init_database()

    @contextmanager
    def get_db_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def init_database(self):
        try:
            with self.get_db_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        shares REAL NOT NULL CHECK (shares > 0),
                        purchase_price REAL NOT NULL CHECK (purchase_price > 0),
                        purchase_date TEXT NOT NULL,
                        risk_tolerance TEXT CHECK (risk_tolerance IN ('Low', 'Medium', 'High')),
                        investment_goals TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, symbol)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        age INTEGER CHECK (age >= 18 AND age <= 100),
                        risk_tolerance TEXT CHECK (risk_tolerance IN ('Conservative', 'Moderate', 'Aggressive')),
                        investment_timeline TEXT,
                        income_level TEXT,
                        experience_level TEXT,
                        financial_goals TEXT,
                        created_date TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_portfolios_symbol ON portfolios(symbol)"
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def add_to_portfolio(
        self, user_id: str, symbol: str, shares: float, price: float
    ) -> bool:
        try:
            user_id = self.security.validate_user_id(user_id)
            symbol = re.sub(r"[^A-Z.\-]", "", symbol.upper())
            if shares <= 0 or price <= 0:
                raise ValueError("Shares and price must be positive")
            with self.get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO portfolios
                    (user_id, symbol, shares, purchase_price, purchase_date, risk_tolerance, investment_goals)
                    VALUES (?, ?, ?, ?, date('now'), ?, ?)
                    """,
                    (user_id, symbol, shares, price, "Medium", "Growth"),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Portfolio add failed: {e}")
            return False

    def get_portfolio_performance(self, user_id: str) -> dict:
        try:
            user_id = self.security.validate_user_id(user_id)
            with self.get_db_connection() as conn:
                positions = conn.execute(
                    """
                    SELECT symbol, shares, purchase_price
                    FROM portfolios
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                ).fetchall()

            total_value = 0.0
            total_cost = 0.0
            performance = []

            for symbol, shares, purchase_price in positions:
                try:
                    current_price = safe_latest_price(symbol)
                    if current_price is None:
                        continue
                    position_value = shares * current_price
                    position_cost = shares * purchase_price
                    pnl = position_value - position_cost
                    pnl_pct = (pnl / position_cost) * 100 if position_cost > 0 else 0
                    performance.append(
                        {
                            "symbol": symbol,
                            "shares": shares,
                            "current_price": round(current_price, 2),
                            "purchase_price": purchase_price,
                            "current_value": round(position_value, 2),
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                        }
                    )
                    total_value += position_value
                    total_cost += position_cost
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
                    continue

            return {
                "positions": performance,
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "total_pnl": round(total_value - total_cost, 2),
                "total_return_pct": round(
                    ((total_value - total_cost) / total_cost * 100), 2
                )
                if total_cost > 0
                else 0,
            }
        except Exception as e:
            logger.error(f"Portfolio performance calculation failed: {e}")
            return {
                "positions": [],
                "total_value": 0,
                "total_cost": 0,
                "total_pnl": 0,
                "total_return_pct": 0,
            }


# Initialize globals after classes exist
def initialize_global_instances():
    global rate_limiter, conversation_memory, portfolio_manager
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    if conversation_memory is None:
        conversation_memory = ConversationMemory()
    if portfolio_manager is None:
        portfolio_manager = SecurePortfolioManager()


# ─────────────────────── Stock data helpers / caching ──────────────────────
def calculate_rsi(prices: pd.Series, periods: int = 14) -> float:
    if len(prices) < periods:
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


@lru_cache(maxsize=512)
def _get_ticker(symbol: str):
    return yf.Ticker(symbol)


def _price_bucket():
    return int(time.time() // PRICE_BUCKET_SECONDS)


@lru_cache(maxsize=4096)
def _safe_latest_price_cached(symbol: str, bucket: int):
    try:
        t = _get_ticker(symbol)
        fi = getattr(t, "fast_info", {}) or {}
        p = fi.get("last_price") or fi.get("last_close") or fi.get("last_price_raw")
        if p:
            return float(p)
        for period in ["5d", "1mo"]:
            h = t.history(period=period)["Close"].dropna()
            if len(h):
                return float(h.iloc[-1])
    except Exception:
        pass
    return None


def safe_latest_price(symbol: str) -> Optional[float]:
    return _safe_latest_price_cached(symbol, _price_bucket())


@lru_cache(maxsize=2048)
def _close_series(symbol: str) -> pd.Series:
    try:
        return _get_ticker(symbol).history(period="6mo")["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


@lru_cache(maxsize=1024)
def safe_info_and_vol(symbol: str) -> tuple[dict, float]:
    info, vol = {}, 0.0
    try:
        info = _get_ticker(symbol).info or {}
    except Exception:
        info = {}
    close = _close_series(symbol)
    if len(close) >= 5:
        vol = float(close.pct_change().std() * np.sqrt(252))
    return info, vol


def safe_sma_rsi(symbol: str, price_now: float) -> tuple[float, float]:
    close = _close_series(symbol)
    if close.empty:
        return float(price_now), 50.0
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(
        price_now
    )
    rsi = float(calculate_rsi(close)) if len(close) >= 14 else 50.0
    return sma20, rsi


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
    }


def get_real_stock_data(
    symbols: List[str] = ["AAPL", "MSFT", "GOOGL", "JNJ", "JPM", "PFE", "XOM", "NVDA", "TSLA", "V"]
) -> List[Dict[str, Any]]:
    real_companies: List[Dict[str, Any]] = []
    for symbol in symbols[:10]:
        try:
            price_now = safe_latest_price(symbol)
            if price_now is None:
                continue
            info, vol = safe_info_and_vol(symbol)

            market_cap = info.get("marketCap", 0)
            pe_ratio = info.get("trailingPE", 0)
            sector = info.get("sector", "Unknown")
            name = info.get("longName", symbol)
            beta = (info.get("beta") or 1.0)

            if (beta and beta < 0.8) and (vol and vol < 0.25):
                risk = "Low"
            elif (beta and beta > 1.2) or (vol and vol > 0.40):
                risk = "High"
            else:
                risk = "Medium"

            sma20, rsi = safe_sma_rsi(symbol, price_now)

            real_companies.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "sector": sector,
                    "market_cap": market_cap,
                    "pe_ratio": round(pe_ratio, 2) if pe_ratio else 0,
                    "price": round(price_now, 2),
                    "risk": risk,
                    "beta": round(beta, 2) if beta else 1.0,
                    "volatility": round(vol, 3) if vol else 0,
                    "sma_20": round(sma20, 2),
                    "rsi": round(rsi, 1),
                    "trend": "Bullish" if price_now > sma20 else "Bearish",
                }
            )
        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            continue
    return real_companies


def get_sp500_tickers() -> List[str]:
    try:
        if os.path.exists(SP500_CACHE_FILE):
            age = time.time() - os.path.getmtime(SP500_CACHE_FILE)
            if age < SP500_CACHE_TTL_HOURS * 3600:
                return pd.read_csv(SP500_CACHE_FILE)["Symbol"].tolist()
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
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
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "JNJ",
            "V",
            "WMT",
            "PG",
            "UNH",
            "HD",
            "CVX",
            "XOM",
            "BAC",
            "ABBV",
            "KO",
            "PFE",
        ]


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


def prefilter_symbols_by_query(
    query_lower: str, spx_df: pd.DataFrame, max_candidates: int = 80
) -> List[str]:
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


def get_stocks_by_real_criteria(query: str) -> List[Dict[str, Any]]:
    """Light prefilter by sector list, then fetch real metrics, then filter."""
    query_lower = query.lower()
    spx_df = get_sp500_table_cached()
    candidates = prefilter_symbols_by_query(query_lower, spx_df, max_candidates=80)
    if not candidates:
        candidates = get_sp500_tickers()[:80]
    real_companies = get_real_stock_data(candidates[:30])  # fast
    filtered = filter_stocks_by_query(real_companies, query_lower)
    return filtered[:10] if filtered else real_companies[:10]


def filter_stocks_by_query(companies: List[dict], query_lower: str) -> List[dict]:
    filtered = companies
    if any(w in query_lower for w in ["tech", "technology", "software"]):
        filtered = [c for c in filtered if "technology" in c.get("sector", "").lower()]
    elif any(w in query_lower for w in ["health", "healthcare", "medical", "pharma"]):
        filtered = [
            c
            for c in filtered
            if any(x in c.get("sector", "").lower() for x in ["health", "pharma", "biotech"])
        ]
    elif any(w in query_lower for w in ["finance", "bank", "financial"]):
        filtered = [c for c in filtered if "financial" in c.get("sector", "").lower()]
    elif any(w in query_lower for w in ["energy", "oil", "gas"]):
        filtered = [c for c in filtered if "energy" in c.get("sector", "").lower()]
    elif any(w in query_lower for w in ["consumer", "retail"]):
        filtered = [
            c
            for c in filtered
            if "consumer" in c.get("sector", "").lower() or "retail" in c.get("sector", "").lower()
        ]

    if any(w in query_lower for w in ["safe", "conservative", "stable"]):
        filtered = [
            c
            for c in filtered
            if (c.get("beta") is not None and c["beta"] < 1.0)
            and (c.get("pe_ratio") and 0 < c["pe_ratio"] < 25)
            and (c.get("market_cap") and c["market_cap"] > 10_000_000_000)
        ]

    if any(p in query_lower for p in ["under $50", "below 50", "under 50"]):
        filtered = [c for c in filtered if c.get("price") and c["price"] < 50]
    elif any(p in query_lower for p in ["under $100", "below 100", "under 100"]):
        filtered = [c for c in filtered if c.get("price") and c["price"] < 100]

    if any(p in query_lower for p in ["undervalued", "low pe", "value"]):
        filtered = [c for c in filtered if c.get("pe_ratio") and 0 < c["pe_ratio"] < 15]

    if any(w in query_lower for w in ["startup", "growth", "new", "young"]):
        filtered = [
            c
            for c in filtered
            if (c.get("beta") and c["beta"] > 1.0) or (c.get("pe_ratio") and c["pe_ratio"] > 25)
        ]
    return filtered


# ────────────────────── Translation / PDF / Web search ─────────────────────
@lru_cache(maxsize=20)
def get_translator(source_lang: str, target_lang: str):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        return pipeline("translation", model=model_name, max_length=512)
    except Exception as e:
        logger.warning(f"Translation model loading failed: {e}")
        return None


def translate_input_to_english(text: str) -> tuple[str, str]:
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
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web search results found."
        out = "Recent web search results:\n\n"
        for i, r in enumerate(results, 1):
            out += f"{i}. **{r['title']}**\n"
            out += f"   {r['body'][:200]}...\n"
            out += f"   Source: {r['href']}\n\n"
        return out
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return "Web search temporarily unavailable."


# ───────────────────────────── LLM + agents ────────────────────────────────
class FinancialAgentState(TypedDict):
    query: str
    user_profile: Dict
    user_requirements: Dict
    sql_queries: List[str]
    company_data: List[Dict]
    market_analysis: str
    fundamental_analysis: str
    technical_analysis: List[Dict]
    recommendations: List[Dict]
    final_response: str


llm = ChatGroq(
    temperature=0.1, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192"
)


def generate_sql_agent(state: FinancialAgentState):
    query = state["query"]
    user_profile = state.get("user_profile", {})
    sql_prompt = f"""
    Generate SQL queries for financial analysis:
    Query: "{query}"
    User Profile: Risk: {user_profile.get('risk_tolerance', 'Moderate')}, Timeline: {user_profile.get('investment_timeline', 'Medium-term')}

    Schema:
    - companies (symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta)
    - portfolios (user_id, symbol, shares, purchase_price, purchase_date)

    Generate 2-3 relevant SQL queries:
    """
    try:
        response = llm.invoke([{"role": "user", "content": sql_prompt}])
        sql_queries = re.findall(r"```sql\n(.*?)\n```", response.content, re.DOTALL)
        if not sql_queries:
            sql_queries = [
                "SELECT * FROM companies WHERE sector LIKE '%technology%' ORDER BY market_cap DESC;"
            ]
        state["sql_queries"] = sql_queries
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        state["sql_queries"] = [
            "SELECT * FROM companies ORDER BY market_cap DESC LIMIT 10;"
        ]
    return state


def data_integration_agent(state: FinancialAgentState):
    query = state["query"]
    sql_queries = state.get("sql_queries", [])
    db_path = "secure_portfolios.db"
    all_results: List[pd.DataFrame] = []

    try:
        with sqlite3.connect(db_path) as conn:
            for sql_query in sql_queries:
                try:
                    safe_query = SecurityManager.allowlist_sql(sql_query)
                    df = pd.read_sql_query(safe_query, conn)
                    if not df.empty:
                        all_results.append(df)
                except Exception as e:
                    logger.error(f"SQL execution failed for query '{sql_query}': {e}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    if all_results:
        final_df = (
            pd.concat(all_results).drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        )
        company_data = [
            normalize_company_keys(r) for r in final_df.to_dict("records")
        ]
    else:
        logger.warning("SQL queries yielded no results. Falling back to live yfinance.")
        company_data = [
            normalize_company_keys(x) for x in get_stocks_by_real_criteria(query)
        ]

    state["company_data"] = company_data[:5]

    analysis_prompt = f"""
    Analyze these companies: {[{
        'name': c['name'],
        'sector': c['sector'],
        'market_cap': f"${c.get('market_cap', 0)/1000000000:.1f}B",
        'pe_ratio': c.get('pe_ratio', 'N/A'),
        'price': c.get('price', 'N/A')
    } for c in state["company_data"][:3]]}

    Provide a brief market analysis (2-3 sentences) covering sector trends and general risk.
    """
    try:
        analysis_response = llm.invoke([{"role": "user", "content": analysis_prompt}])
        state["market_analysis"] = analysis_response.content
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        state["market_analysis"] = "Market analysis is currently unavailable."
    return state


def _safe(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def score_company(c: dict) -> dict:
    pe = _safe(c.get("pe_ratio"))
    beta = _safe(c.get("beta"), 1.0)
    vol = _safe(c.get("volatility"))
    price = _safe(c.get("price"))
    sma20 = _safe(c.get("sma_20"), price)
    rsi = _safe(c.get("rsi"), 50.0)
    mcap = _safe(c.get("market_cap")) / 1e9

    fx: Dict[str, float] = {}
    if pe <= 0:
        fx["valuation"] = 10
    else:
        if pe < 12:
            val = 35
        elif pe < 18:
            val = 28
        elif pe < 25:
            val = 22
        elif pe < 35:
            val = 12
        else:
            val = 5
        fx["valuation"] = val

    mom = 0
    if price and sma20:
        if price > sma20:
            mom += 12
        if 45 <= rsi <= 65:
            mom += 10
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            mom += 6
    fx["momentum"] = mom

    risk_pts = 0
    if beta < 0.9:
        risk_pts += 12
    elif beta < 1.1:
        risk_pts += 8
    elif beta < 1.3:
        risk_pts += 4
    if vol:
        if vol < 0.25:
            risk_pts += 13
        elif vol < 0.35:
            risk_pts += 8
        elif vol < 0.5:
            risk_pts += 3
    fx["risk"] = risk_pts

    size = 0
    if mcap >= 200:
        size = 10
    elif mcap >= 50:
        size = 7
    elif mcap >= 10:
        size = 4
    fx["size"] = size

    score = round(sum(fx.values()), 1)
    return {"score": score, "factors": {k: round(v, 1) for k, v in fx.items()}}


def softmax_weights(scores: List[float], cap: float = 0.6) -> List[float]:
    import math

    if not scores:
        return []
    mx = max(scores) or 1.0
    exps = [math.exp(s / mx) for s in scores]
    ssum = sum(exps) or 1.0
    w = [e / ssum for e in exps]
    w = [min(x, cap) for x in w]
    ssum = sum(w) or 1.0
    return [round(x / ssum, 4) for x in w]


def recommendation_agent(state: FinancialAgentState):
    companies = state["company_data"]
    analysis = state["market_analysis"]
    query = state["query"]

    scored = []
    for c in companies[:5]:
        sc = score_company(c)
        scored.append({**c, **sc})

    weights = softmax_weights([x["score"] for x in scored], cap=0.6)
    for i, w in enumerate(weights):
        scored[i]["weight"] = w

    recs = []
    for x in scored[:3]:
        alloc_pct = f"{round(x['weight'] * 100):d}%"
        rationale = (
            f"PE={x.get('pe_ratio')} • Beta={x.get('beta')} • "
            f"Vol≈{x.get('volatility')} • RSI={x.get('rsi')} • "
            f"Trend={x.get('trend')}"
        )
        why = ", ".join(
            f"{k}:{v}"
            for k, v in sorted(x["factors"].items(), key=lambda kv: -kv[1])
            if v > 0
        )
        recs.append(
            {
                "company": x["name"],
                "symbol": x["symbol"],
                "sector": x["sector"],
                "allocation": alloc_pct,
                "price": x["price"],
                "risk": x["risk"],
                "pe_ratio": x["pe_ratio"],
                "market_cap": x["market_cap"],
                "score": x["score"],
                "explain": x["factors"],
                "rationale": rationale,
                "why": why,
            }
        )

    lines = [f"**Market Analysis:** {analysis}\n", f"**Query:** {query}\n", "**Top Picks (data-backed):**"]
    for r in recs:
        lines.append(
            f"- **{r['company']}** ({r['symbol']}) — {r['allocation']} "
            f"| ${r['price']:.2f} | PE {r['pe_ratio']} | "
            f"MktCap ${r['market_cap']/1e9:.1f}B | Risk {r['risk']} | Score {r['score']}/100"
        )
        lines.append(f"  • Why: {r['why']}")
        lines.append(f"  • Stats: {r['rationale']}")
    final = "\n".join(lines)

    state["recommendations"] = recs
    state["final_response"] = final
    return state


def create_financial_workflow():
    workflow = StateGraph(FinancialAgentState)
    workflow.add_node("sql_agent", generate_sql_agent)
    workflow.add_node("data_agent", data_integration_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)
    workflow.set_entry_point("sql_agent")
    workflow.add_edge("sql_agent", "data_agent")
    workflow.add_edge("data_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    return workflow.compile()


def is_financial_query(query: str) -> bool:
    q = query.lower()
    financial_keywords = [
        "stock",
        "portfolio",
        "market",
        "buy",
        "sell",
        "recommendation",
        "risk",
        "return",
        "shares",
        "p/e",
        "market cap",
        "dividend",
        "investment",
        "ticker",
    ]
    if any(k in q for k in financial_keywords):
        return True
    if re.search(r"\b[A-Z]{1,5}\b", query):
        return True
    if "company" in q or "safe" in q:
        return any(k in q for k in ["stock", "investment", "p/e", "market"])
    return False


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["buy", "add", "purchase"]):
        if re.search(r"buy\s+([\d.,]+)\s+shares?\s+of\s+([A-Za-z.\-]{1,7})", query, re.IGNORECASE):
            return "add_to_portfolio"
    if re.search(r"\b((my\s+)?portfolio|holdings|performance)\b", q):
        return "view_portfolio"
    if is_financial_query(query):
        return "financial_analysis"
    return "general_query"


def format_portfolio_response(perf: dict) -> str:
    if not perf or not perf.get("positions"):
        return "Your portfolio is currently empty. Try: *buy 10 shares of AAPL*."
    response = "### Your Portfolio Performance\n\n"
    response += f"**Total Portfolio Value:** ${perf['total_value']:,.2f}\n"
    response += f"**Total Cost Basis:** ${perf['total_cost']:,.2f}\n"
    response += f"**Total P&L:** ${perf['total_pnl']:,.2f} ({perf['total_return_pct']:+.2f}%)\n\n"
    response += "#### Individual Holdings:\n"
    for p in perf["positions"]:
        pnl_emoji = "📈" if p["pnl"] > 0 else "📉" if p["pnl"] < 0 else "➖"
        cp = p.get("current_price")
        cp_str = f"${cp:.2f}" if cp is not None else "—"
        response += f"**{p['symbol']}** - {p['shares']} shares\n"
        response += f"  • Current: {cp_str} | Purchase: ${p['purchase_price']:.2f}\n"
        response += f"  • Value: ${p['current_value']:,.2f} | P&L: {pnl_emoji} ${p['pnl']:+,.2f} ({p['pnl_pct']:+.2f}%)\n\n"
    return response


def validate_query(query: str) -> str:
    if not query or len(query.strip()) == 0:
        raise ValueError("Query cannot be empty")
    if len(query) > 2000:
        raise ValueError("Query too long (max 2000 characters)")
    q = query.strip()
    q = re.sub(r"<[^>]+>", "", q)
    q = re.sub(r"javascript:", "", q, flags=re.IGNORECASE)
    return q


def handle_general_query_with_search(
    query: str, context: str, chat_history: List[Dict[str, str]], file_path: Optional[str] = None
) -> str:
    should_search = any(
        k in query.lower()
        for k in ["company", "startup", "recent", "news", "current", "today", "latest", "what is", "who is", "tell me about"]
    )
    web_results = web_search_ddg(query, max_results=3) if should_search else ""
    history_text = "\n".join(
        f"User: {m['user']}\nAgent: {m['agent']}" for m in chat_history[-3:]
    )
    file_summary = extract_text_from_pdf(file_path) if file_path else ""
    full_prompt = f"""
    You are a professional, helpful AI assistant. Provide clear and accurate responses.

    {'Previous conversation context:\n' + context + '\n' if context else ''}
    {'Recent chat history:\n' + history_text + '\n' if history_text else ''}
    {'File content summary:\n' + file_summary + '\n' if file_summary else ''}
    {'Current web search results:\n' + web_results + '\n' if web_results else ''}

    User's question: {query}

    Instructions:
    1. Use the web search results if available to provide accurate, up-to-date information
    2. If asking about a specific company/organization, prioritize the web search results
    3. Be direct and specific, not generic
    4. If the web results are relevant, cite them in your response
    5. Keep response concise and focused

    Answer:
    """
    try:
        return llm.invoke([{"role": "user", "content": full_prompt}]).content
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."


# ────────────────────────── Public entrypoint ──────────────────────────────
def run_customer_support(
    query: str,
    force_language: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    file_path: Optional[str] = None,
    user_id: str = "anonymous_user",
) -> Dict[str, Any]:
    if chat_history is None:
        chat_history = []

    # Initialize global singletons
    initialize_global_instances()

    # Rate limit
    allowed, remaining = rate_limiter.is_allowed(user_id)
    if not allowed:
        return {
            "original_language": "English",
            "response": "⚠️ Rate limit exceeded. Please try again later or upgrade to premium.",
            "raw_response": "Rate limit exceeded",
        }

    # Validate + translate input
    try:
        query = validate_query(query)
    except ValueError as e:
        return {
            "original_language": "English",
            "response": f"❌ Invalid input: {str(e)}",
            "raw_response": str(e),
        }

    translated_query, detected_lang = translate_input_to_english(query)

    # Context
    context = conversation_memory.get_context_summary(user_id)

    # Intent
    intent = detect_intent(translated_query)

    if intent == "add_to_portfolio":
        m = re.search(
            r"buy\s+([\d.,]+)\s+shares?\s+of\s+([A-Za-z.\-]{1,7})",
            translated_query,
            re.IGNORECASE,
        )
        if m:
            shares_str, symbol = m.groups()
            try:
                current_price = safe_latest_price(symbol.upper())
                if current_price is None:
                    raise ValueError("price unavailable")
                shares = float(shares_str.replace(",", ""))
                success = portfolio_manager.add_to_portfolio(
                    user_id, symbol.upper(), shares, current_price
                )
                if success:
                    response = f"✅ Added {shares:g} shares of {symbol.upper()} at ${current_price:.2f} per share."
                else:
                    response = "❌ Failed to add to portfolio. Please try again."
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
            "user_profile": {},
            "user_requirements": {},
            "sql_queries": [],
            "company_data": [],
            "market_analysis": "",
            "fundamental_analysis": "",
            "technical_analysis": [],
            "recommendations": [],
            "final_response": "",
        }
        try:
            result = workflow.invoke(initial_state)
            response = result["final_response"]
            demo_info = (
                "\n\n🔧 **Live Financial Analysis System:**\n"
                f"**SQL Queries Generated:** {len(result['sql_queries'])}\n"
                "**Real-time Data Sources:** Yahoo Finance, Live Market Data\n"
                f"**Companies Analyzed:** {len(result['company_data'])}\n"
                f"**Recommendations:** {len(result['recommendations'])}\n\n"
                "**Live Stock Recommendations:**\n"
            )
            for i, rec in enumerate(result["recommendations"], 1):
                demo_info += (
                    f"{i}. **{rec['company']}** ({rec['symbol']}) - ${rec['price']}/share\n"
                    f"   • {rec['allocation']} allocation, {rec['risk']} risk\n"
                    f"   • {rec['rationale']}\n\n"
                )
            response += demo_info
        except Exception as e:
            logger.error(f"Financial workflow failed: {e}")
            response = "Sorry, the financial analysis system is temporarily unavailable. Please try again later."
    else:
        response = handle_general_query_with_search(
            translated_query, context, chat_history, file_path
        )

    # Memory + translation
    conversation_memory.add_exchange(user_id, query, response)
    target_lang = force_language if force_language else detected_lang
    translated_response = translate_output_from_english(response, target_lang)

    return {
        "original_language": SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "response": translated_response,
        "raw_response": response,
        "remaining_requests": remaining,
    }
