# populate_db.py — LIGHT MODE: seed SQLite 'companies' table from S&P 500 (Wikipedia only)
import pandas as pd
import sqlite3, logging, os, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("populate_db")

SP500_CACHE_FILE = "sp500_cache.csv"
SP500_CACHE_TTL_HOURS = 24

def get_sp500_table():
    """
    Returns a DataFrame with at least columns: Symbol, Security, GICS Sector.
    Uses a 24h CSV cache to avoid repeated network calls.
    """
    try:
        if os.path.exists(SP500_CACHE_FILE):
            age = time.time() - os.path.getmtime(SP500_CACHE_FILE)
            if age < SP500_CACHE_TTL_HOURS * 3600:
                df = pd.read_csv(SP500_CACHE_FILE)
                if {"Symbol", "Security"}.issubset(df.columns):
                    return df

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        # Normalize ticker format for Yahoo
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df.to_csv(SP500_CACHE_FILE, index=False)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 table: {e}")
        # tiny fallback
        return pd.DataFrame({
            "Symbol": ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","JNJ","V"],
            "Security": ["Apple","Microsoft","Alphabet Class A","Amazon","NVIDIA","Tesla","Meta Platforms","JPMorgan Chase","Johnson & Johnson","Visa"],
            "GICS Sector": ["Information Technology","Information Technology","Communication Services","Consumer Discretionary","Information Technology","Consumer Discretionary","Communication Services","Financials","Health Care","Financials"],
        })

def populate_company_data():
    df = get_sp500_table()
    if df.empty:
        logger.error("No tickers found. Aborting.")
        return

    db_path = "secure_portfolios.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
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
    """)

    # LIGHT seed: only static fields. Dynamic metrics are fetched later on-demand.
    rows = 0
    for _, r in df.iterrows():
        symbol = str(r["Symbol"])
        name = str(r.get("Security", symbol))
        sector = str(r.get("GICS Sector", "Unknown"))

        # placeholders — live enrichment will overwrite these for shortlisted symbols
        market_cap = 0.0
        pe_ratio = 0.0
        price = 0.0
        beta = 1.0
        risk = "Medium"

        cur.execute(
            """
            INSERT OR REPLACE INTO companies
            (symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, name, sector, market_cap, pe_ratio, price, risk, beta),
        )
        rows += 1

    conn.commit()
    conn.close()
    logger.info(f"Database population complete. Seeded {rows} companies (light mode).")

if __name__ == "__main__":
    populate_company_data()
