# sitecustomize.py — selective offline shims (free-tier friendly)
import os, datetime as dt
import numpy as np, pandas as pd

FREE = os.environ.get("FREE_DEMO","0") == "1"
SHIM_YF   = os.environ.get("SHIM_YF","1")   == "1"
SHIM_DDG  = os.environ.get("SHIM_DDG","1")  == "1"
SHIM_TRAFI= os.environ.get("SHIM_TRAFI","1")== "1"

# Cache HTTP to reduce rate-limit hits (yfinance, etc.)
try:
    import requests_cache
    expire = int(os.environ.get("YF_CACHE_SECS","86400"))
    requests_cache.install_cache("yfdemo_cache", expire_after=expire)
    print(f"[sitecustomize] requests-cache ON (expire={expire}s)")
except Exception as e:
    print("[sitecustomize] requests-cache OFF:", e)

if not FREE:
    print("[sitecustomize] FREE_DEMO not set - live mode")
else:
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    print("[sitecustomize] FREE_DEMO=1 -> offline shims framework active")

def _demo_price_series(days=160, start=100.0, drift=0.0004, vol=0.02, seed=42):
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(days) / (252 ** 0.5)
    prices = start * np.cumprod(1 + rets)
    idx = pd.date_range(end=dt.date.today(), periods=days, freq="B")
    return pd.DataFrame({"Close": prices}, index=idx)

_LONG = {
    "MMM": {"longName":"3M","sector":"Industrials","industry":"Conglomerates",
            "trailingPE":15.8,"marketCap":52_000_000_000,"dividendYield":0.052,"beta":0.85},
    "AOS": {"longName":"A. O. Smith","sector":"Industrials","industry":"Building Products",
            "trailingPE":22.4,"marketCap":16_000_000_000,"dividendYield":0.019,"beta":0.95},
    "ABT": {"longName":"Abbott Laboratories","sector":"Healthcare","industry":"Medical Devices",
            "trailingPE":23.9,"marketCap":190_000_000_000,"dividendYield":0.019,"beta":0.90},
}

class _DemoTicker:
    def __init__(self, symbol, session=None):
        self.ticker = (symbol or "DEMO").upper()
        seed = abs(hash(self.ticker)) % (2**32)
        self._hist = _demo_price_series(seed=seed)
        meta = _LONG.get(self.ticker, {})
        self.info = {
            "symbol": self.ticker,
            "longName": meta.get("longName", self.ticker),
            "sector": meta.get("sector","Misc"),
            "industry": meta.get("industry","Misc"),
            "trailingEps": 3.0,
            "trailingPE": meta.get("trailingPE", 20.0),
            "marketCap": meta.get("marketCap", 10_000_000_000),
            "dividendYield": meta.get("dividendYield", 0.02),
            "ebitda": 1_200_000_000.0,
            "totalDebt": 5_000_000_000.0,
            "operatingCashflow": 600_000_000.0,
            "freeCashflow": 500_000_000.0,
            "beta": meta.get("beta", 0.95),
        }
    def history(self, period="6mo", interval="1d", **kwargs):
        return self._hist.copy()

def _patch_yfinance():
    try:
        import yfinance as yf
        def _download(tickers, *args, **kwargs):
            if isinstance(tickers, str): tickers = [tickers]
            frames = []
            for t in tickers or []:
                df = _DemoTicker(t).history()
                df.columns = pd.MultiIndex.from_product([["Close"], [t]])
                frames.append(df)
            if not frames: return pd.DataFrame()
            return pd.concat(frames, axis=1)
        yf.Ticker = _DemoTicker
        yf.download = _download
        print("[sitecustomize] yfinance SHIM active")
    except Exception as e:
        print("[sitecustomize] yfinance shim skipped:", e)

def _patch_ddg():
    try:
        import duckduckgo_search as dds
        class _DemoDDGS:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def text(self, q, max_results=5):
                base = [
                    {"title":"Market roundup: defensives lead","href":"https://example.com/defensives"},
                    {"title":"Earnings preview: stable cash flows","href":"https://example.com/earnings"},
                    {"title":"Regulatory watch: no adverse items","href":"https://example.com/regulatory"},
                ]
                return base[:max_results]
        dds.DDGS = _DemoDDGS
        print("[sitecustomize] DDG SHIM active")
    except Exception as e:
        print("[sitecustomize] DDG shim skipped:", e)

def _patch_trafilatura():
    try:
        import trafilatura as _tr
        _tr.fetch_url = lambda *a, **k: None
        print("[sitecustomize] trafilatura SHIM active")
    except Exception as e:
        print("[sitecustomize] trafilatura shim skipped:", e)

if FREE:
    if SHIM_YF:    _patch_yfinance()
    if SHIM_DDG:   _patch_ddg()
    if SHIM_TRAFI: _patch_trafilatura()
