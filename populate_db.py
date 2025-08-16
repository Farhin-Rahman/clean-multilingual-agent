# populate_db.py — seed SQLite companies table from S&P 500 via yfinance
import pandas as pd, yfinance as yf, sqlite3, numpy as np, logging, os, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("populate_db")
SP500_CACHE_FILE = "sp500_cache.csv"
SP500_CACHE_TTL_HOURS = 24

def get_sp500_tickers():
    try:
        if os.path.exists(SP500_CACHE_FILE):
            age = time.time() - os.path.getmtime(SP500_CACHE_FILE)
            if age < SP500_CACHE_TTL_HOURS * 3600:
                return pd.read_csv(SP500_CACHE_FILE)["Symbol"].tolist()
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = [ticker.replace('.', '-') for ticker in sp500_table['Symbol'].tolist()]
        pd.DataFrame({"Symbol": tickers}).to_csv(SP500_CACHE_FILE, index=False)
        return tickers
    except Exception as e:
        logger.error(f"Failed to get S&P 500 list live: {e}")
        if os.path.exists(SP500_CACHE_FILE):
            try:
                return pd.read_csv(SP500_CACHE_FILE)["Symbol"].tolist()
            except Exception:
                pass
        return ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","JNJ","V"]

def safe_latest_price(symbol: str):
    t = yf.Ticker(symbol)
    try:
        fi = getattr(t, "fast_info", {}) or {}
        p = fi.get("last_price") or fi.get("last_close")
        if p: return float(p)
    except Exception:
        pass
    for period in ["5d", "1mo"]:
        try:
            h = t.history(period=period)["Close"].dropna()
            if len(h): return float(h.iloc[-1])
        except Exception:
            pass
    return None

def populate_company_data():
    tickers = get_sp500_tickers()
    if not tickers:
        logger.error("No tickers found. Aborting."); return

    db_path = "secure_portfolios.db"
    conn = sqlite3.connect(db_path); cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            market_cap REAL,
            pe_ratio REAL,
            price REAL,
            risk_level TEXT,
            beta REAL
        )
    ''')

    logger.info(f"Fetching data for {len(tickers)} companies. This may take a while...")
    for symbol in tickers:
        try:
            price = safe_latest_price(symbol)
            info = {}
            try: info = yf.Ticker(symbol).info or {}
            except Exception: info = {}
            if price is not None:
                name = info.get('longName', symbol)
                sector = info.get('sector', 'Unknown')
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 0)
                beta = (info.get('beta') or 1.0)
                risk = "Low" if beta < 0.8 else "High" if beta > 1.2 else "Medium"
                cursor.execute("""
                    INSERT OR REPLACE INTO companies (symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, name, sector, market_cap, pe_ratio, price, risk, beta))
                logger.info(f"Added/updated {symbol}")
        except Exception as e:
            logger.warning(f"Could not fetch data for {symbol}: {e}")
            continue

    conn.commit(); conn.close()
    logger.info("Database population complete.")

if __name__ == "__main__":
    populate_company_data()
