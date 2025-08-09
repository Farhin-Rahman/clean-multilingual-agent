from typing import TypedDict, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache
from transformers import pipeline
from langdetect import detect
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import json
import re
import yfinance as yf
import requests
import pandas as pd
import sqlite3
import finnhub
from datetime import datetime, timedelta
import numpy as np
from fredapi import Fred 
import logging
import hashlib
import time
from contextlib import contextmanager

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate API keys
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Optional API keys (graceful degradation)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

SUPPORTED_LANGUAGES = {
    "bn": "Bengali", "es": "Spanish", "de": "German", "fr": "French", 
    "it": "Italian", "pt": "Portuguese", "en": "English"
}

# Global instances
rate_limiter = None
conversation_memory = None
portfolio_manager = None

def initialize_global_instances():
    """Initialize global instances"""
    global rate_limiter, conversation_memory, portfolio_manager
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    if conversation_memory is None:
        conversation_memory = ConversationMemory()
    if portfolio_manager is None:
        portfolio_manager = SecurePortfolioManager()

class SecurityManager:
    """Production-grade security management"""
    
    @staticmethod
    def sanitize_sql_input(query: str) -> str:
        """Prevent SQL injection"""
        dangerous_patterns = [
            r'\bdrop\b', r'\bdelete\b', r'\bupdate\b', r'\binsert\b',
            r'\bcreate\b', r'\balter\b', r'\btruncate\b', r'\bexec\b',
            r'\beval\b', r'--', r'/\*', r'\*/', r';'
        ]
        
        cleaned = query
        for pattern in dangerous_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate and sanitize user ID"""
        if not user_id or len(user_id) < 1:
            return "anonymous_user"
        
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', user_id)[:50]
        return sanitized or "anonymous_user"
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive information"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class RateLimiter:
    """Production-grade rate limiting"""
    
    def __init__(self):
        self.requests = {}
        self.limits = {
            'free': {'requests': 100, 'window': 3600}  # 100 per hour - generous for portfolio demo
            }
    
    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """Check if request is allowed"""
        now = time.time()
        user_key = f"{user_id}_free"  # Always use free tier
        
        if user_key not in self.requests:
            self.requests[user_key] = []
        window = self.limits['free']['window']
        self.requests[user_key] = [req_time for req_time in self.requests[user_key] 
                                       if now - req_time < window]
        
        current_count = len(self.requests[user_key])
        limit = self.limits['free']['requests']

        if current_count < limit:
            self.requests[user_key].append(now)
            return True, limit - current_count - 1
        
        return False, 0

class ConversationMemory:
    """Enhanced conversation memory management"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.conversations = {}
    
    def add_exchange(self, user_id: str, user_msg: str, agent_msg: str, context: dict = None):
        """Add conversation exchange with context"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'agent': agent_msg,
            'context': context or {},
            'tokens': len(user_msg.split()) + len(agent_msg.split())
        }
        
        self.conversations[user_id].append(exchange)
        
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
    
    def get_context_summary(self, user_id: str, max_tokens: int = 1000) -> str:
        """Get intelligent context summary for continuity"""
        if user_id not in self.conversations:
            return ""
        
        recent_exchanges = self.conversations[user_id][-5:]
        total_tokens = 0
        context_parts = []
        
        for exchange in reversed(recent_exchanges):
            if total_tokens + exchange['tokens'] > max_tokens:
                break
            
            context_parts.append(f"User: {exchange['user']}\nAssistant: {exchange['agent'][:200]}...")
            total_tokens += exchange['tokens']
        
        context_parts.reverse()
        
        financial_context = self.extract_financial_preferences(user_id)
        if financial_context:
            context_parts.append(f"\nUser Financial Profile: {financial_context}")
        
        return "\n---\n".join(context_parts)
    
    def extract_financial_preferences(self, user_id: str) -> str:
        """Extract financial preferences from conversation history"""
        if user_id not in self.conversations:
            return ""
        
        preferences = {
            'risk_tolerance': None,
            'sectors': [],
            'investment_amount': None,
            'timeline': None
        }
        
        all_text = " ".join([ex['user'] + " " + ex['agent'] 
                            for ex in self.conversations[user_id]])
        
        if any(word in all_text.lower() for word in ['conservative', 'safe', 'low risk']):
            preferences['risk_tolerance'] = 'Conservative'
        elif any(word in all_text.lower() for word in ['aggressive', 'high risk', 'growth']):
            preferences['risk_tolerance'] = 'Aggressive'
        
        sectors = ['technology', 'healthcare', 'finance', 'energy', 'consumer']
        for sector in sectors:
            if sector in all_text.lower():
                preferences['sectors'].append(sector)
        
        summary_parts = []
        if preferences['risk_tolerance']:
            summary_parts.append(f"Risk: {preferences['risk_tolerance']}")
        if preferences['sectors']:
            summary_parts.append(f"Interested sectors: {', '.join(preferences['sectors'])}")
        
        return "; ".join(summary_parts)

class RealFinancialData:
    def __init__(self):
        self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) if (finnhub and FINNHUB_API_KEY) else None
        self.fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        
    def get_comprehensive_stock_analysis(self, symbol):
        """Get real fundamental + technical analysis"""
        data = {}
        
        try:
            if self.finnhub_client:
                data['financials'] = self.finnhub_client.company_basic_financials(symbol, 'all')
                data['news'] = self.finnhub_client.company_news(symbol, 
                    _from=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                    to=datetime.now().strftime('%Y-%m-%d'))
                data['recommendations'] = self.finnhub_client.recommendation_trends(symbol)
                data['earnings'] = self.finnhub_client.company_earnings(symbol, limit=4)
            
            if self.fred:
                fed_data = self.fred.get_series('FEDFUNDS', limit=12)
                cpi_data = self.fred.get_series('CPIAUCSL', limit=12)
                data['interest_rates'] = fed_data.iloc[-1] if fed_data is not None else 5.25
                data['inflation'] = cpi_data.pct_change().iloc[-1] * 100 if cpi_data is not None else 3.2
            
        except Exception as e:
            logger.warning(f"API data fetch failed for {symbol}: {e}")
            data = {'error': str(e)}
        
        return data

class SecurePortfolioManager:
    """Production-grade portfolio management"""
    
    def __init__(self):
        self.db_path = "secure_portfolios.db"
        self.security = SecurityManager()
        self.init_database()
    
    @contextmanager
    def get_db_connection(self):
        """Secure database connection"""
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
        """Create secure database"""
        try:
            with self.get_db_connection() as conn:
                conn.execute('''
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
                ''')
                
                conn.execute('''
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
                ''')
                
                conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_symbol ON portfolios(symbol)")
                
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def add_to_portfolio(self, user_id: str, symbol: str, shares: float, price: float) -> bool:
        """Securely add portfolio position"""
        try:
            user_id = self.security.validate_user_id(user_id)
            symbol = re.sub(r'[^A-Z]', '', symbol.upper())
            
            if shares <= 0 or price <= 0:
                raise ValueError("Shares and price must be positive")
            
            with self.get_db_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO portfolios (user_id, symbol, shares, purchase_price, purchase_date, risk_tolerance, investment_goals) VALUES (?, ?, ?, ?, date('now'), ?, ?)",
                    (user_id, symbol, shares, price, "Medium", "Growth")
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Portfolio add failed: {e}")
            return False
    
    def get_portfolio_performance(self, user_id: str) -> dict:
        """Get portfolio performance"""
        try:
            user_id = self.security.validate_user_id(user_id)
            
            with self.get_db_connection() as conn:
                positions = conn.execute(
                    "SELECT symbol, shares, purchase_price FROM portfolios WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,)
                ).fetchall()
            
            total_value = 0
            total_cost = 0
            performance = []
            
            for symbol, shares, purchase_price in positions:
                try:
                    current_data = yf.Ticker(symbol).history(period="1d")
                    if not current_data.empty:
                        current_price = current_data['Close'].iloc[-1]
                        position_value = shares * current_price
                        position_cost = shares * purchase_price
                        pnl = position_value - position_cost
                        pnl_pct = (pnl / position_cost) * 100 if position_cost > 0 else 0
                        
                        performance.append({
                            'symbol': symbol,
                            'shares': shares,
                            'current_price': round(current_price, 2),
                            'purchase_price': purchase_price,
                            'current_value': round(position_value, 2),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round(pnl_pct, 2)
                        })
                        
                        total_value += position_value
                        total_cost += position_cost
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")
                    continue
            
            return {
                'positions': performance,
                'total_value': round(total_value, 2),
                'total_cost': round(total_cost, 2),
                'total_pnl': round(total_value - total_cost, 2),
                'total_return_pct': round(((total_value - total_cost) / total_cost * 100), 2) if total_cost > 0 else 0
            }
        except Exception as e:
            logger.error(f"Portfolio performance calculation failed: {e}")
            return {'positions': [], 'total_value': 0, 'total_cost': 0, 'total_pnl': 0, 'total_return_pct': 0}

def validate_query(query: str) -> str:
    """Enhanced input validation"""
    if not query or len(query.strip()) == 0:
        raise ValueError("Query cannot be empty")
    if len(query) > 2000:
        raise ValueError("Query too long (max 2000 characters)")
    
    query = query.strip()
    query = re.sub(r'<[^>]+>', '', query)
    query = re.sub(r'javascript:', '', query, flags=re.IGNORECASE)
    
    return query

def get_real_stock_data(symbols=['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM', 'PFE', 'XOM', 'NVDA', 'TSLA', 'V']):
    """Enhanced stock data retrieval"""
    real_companies = []
    
    for symbol in symbols[:10]:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")
            
            if not hist.empty and info:
                current_price = round(hist['Close'].iloc[-1], 2)
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 0)
                sector = info.get('sector', 'Unknown')
                name = info.get('longName', symbol)
                revenue_growth = info.get('revenueGrowth', 0)
                profit_margins = info.get('profitMargins', 0)
                debt_to_equity = info.get('debtToEquity', 0)
                
                beta = info.get('beta', 1.0)
                volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                
                if beta and beta < 0.8 and volatility < 0.25:
                    risk = "Low"
                elif beta and beta > 1.2 or volatility > 0.40:
                    risk = "High" 
                else:
                    risk = "Medium"
                
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
                rsi = calculate_rsi(hist['Close']) if len(hist) >= 14 else 50
                
                real_companies.append({
                    "name": name,
                    "symbol": symbol,
                    "sector": sector,
                    "market_cap": market_cap,
                    "pe_ratio": round(pe_ratio, 2) if pe_ratio else 0,
                    "price": current_price,
                    "risk": risk,
                    "beta": round(beta, 2) if beta else 1.0,
                    "revenue_growth": round(revenue_growth * 100, 1) if revenue_growth else 0,
                    "profit_margins": round(profit_margins * 100, 1) if profit_margins else 0,
                    "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity else 0,
                    "volatility": round(volatility, 3) if not np.isnan(volatility) else 0,
                    "sma_20": round(sma_20, 2),
                    "rsi": round(rsi, 1),
                    "trend": "Bullish" if current_price > sma_20 else "Bearish"
                })
        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            continue
    
    return real_companies

def get_sector_stocks(sector_query):
    """Get stocks dynamically based on real market data"""
    return get_stocks_by_real_criteria(sector_query)

def get_stocks_by_real_criteria(query):
    """Dynamic stock filtering using real market data"""
    
    query_lower = query.lower()
    
    # Get broader market data first
    sp500_tickers = get_sp500_tickers()  # Real S&P 500 list
    all_stock_data = []
    
    for ticker in sp500_tickers[:50]:  # Reasonable limit for performance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                continue
                
            # Real-time classification based on actual data
            stock_data = {
                'symbol': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'price': info.get('currentPrice', 0),
                'beta': info.get('beta', 1.0),
                'founded_year': 2010  # Simplified for now
            }
            
            all_stock_data.append(stock_data)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            continue
    
    # Dynamic filtering based on query
    filtered_stocks = filter_stocks_by_query(all_stock_data, query_lower)
    
    return get_real_stock_data([stock['symbol'] for stock in filtered_stocks[:10]])

def filter_stocks_by_query(stocks, query_lower):
    """Filter stocks based on real criteria, not hardcoded lists"""
    
    filtered = stocks
    
    # Real sector filtering
    if any(word in query_lower for word in ['tech', 'technology', 'software']):
        filtered = [s for s in filtered if 'technology' in s['sector'].lower()]
    elif any(word in query_lower for word in ['health', 'healthcare', 'medical', 'pharma']):
        filtered = [s for s in filtered if any(word in s['sector'].lower() 
                   for word in ['healthcare', 'pharmaceutical', 'biotechnology'])]
    elif any(word in query_lower for word in ['finance', 'bank', 'financial']):
        filtered = [s for s in filtered if 'financial' in s['sector'].lower()]
    elif any(word in query_lower for word in ['energy', 'oil', 'gas']):
        filtered = [s for s in filtered if 'energy' in s['sector'].lower()]
    elif any(word in query_lower for word in ['consumer', 'retail']):
        filtered = [s for s in filtered if any(word in s['sector'].lower() 
                   for word in ['consumer', 'retail'])]
    
    # Real growth/startup filtering based on actual metrics
    if any(word in query_lower for word in ['startup', 'growth', 'new', 'young']):
        # Filter by real criteria: high growth, smaller market cap
        filtered = [s for s in filtered if (
            s['revenue_growth'] and s['revenue_growth'] > 0.10 and  # 10%+ revenue growth
            s['market_cap'] < 100_000_000_000 and  # Under $100B market cap
            s['beta'] and s['beta'] > 1.0  # Higher volatility/growth
        )]
    
    # Conservative filtering
    if any(word in query_lower for word in ['safe', 'conservative', 'stable']):
        filtered = [s for s in filtered if (
            s['beta'] and s['beta'] < 1.0 and
            s['pe_ratio'] and s['pe_ratio'] > 0 and s['pe_ratio'] < 25 and
            s['market_cap'] > 10_000_000_000  # Larger, more stable companies
        )]
    
    # Price filtering
    if any(phrase in query_lower for phrase in ['under $100', 'below 100', 'under 100']):
        filtered = [s for s in filtered if s['price'] and s['price'] < 100]
    elif any(phrase in query_lower for phrase in ['under $50', 'below 50', 'under 50']):
        filtered = [s for s in filtered if s['price'] and s['price'] < 50]
    
    # Value filtering
    if any(phrase in query_lower for phrase in ['undervalued', 'low pe', 'value']):
        filtered = [s for s in filtered if s['pe_ratio'] and s['pe_ratio'] > 0 and s['pe_ratio'] < 15]
    
    return filtered

def get_sp500_tickers():
    """Get real S&P 500 ticker list"""
    try:
        # Get real S&P 500 list from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        # Clean up any formatting issues
        return [ticker.replace('.', '-') for ticker in tickers]
    except Exception as e:
        logger.warning(f"Failed to get S&P 500 list: {e}")
        # Fallback to major companies
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 
                'WMT', 'PG', 'UNH', 'HD', 'CVX', 'XOM', 'BAC', 'ABBV', 'KO', 'PFE',
                'TMO', 'COST', 'AVGO', 'MRK', 'PEP', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'CSCO']

def calculate_rsi(prices, periods=14):
    """Calculate real RSI indicator"""
    if len(prices) < periods:
        return 50
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def calculate_investment_capacity(user_profile):
    """Calculate personalized investment allocation"""
    age = user_profile.get('age', 35)
    risk_tolerance = user_profile.get('risk_tolerance', 'Moderate')
    timeline = user_profile.get('investment_timeline', '3-5 years')
    
    base_equity = max(100 - age, 20)
    
    risk_multipliers = {"Conservative": 0.7, "Moderate": 1.0, "Aggressive": 1.3}
    equity_pct = min(base_equity * risk_multipliers[risk_tolerance], 90)
    
    if timeline == "< 1 year":
        equity_pct *= 0.5
    elif timeline == "5+ years":
        equity_pct *= 1.1
    
    bond_pct = min(100 - equity_pct, 80)
    cash_pct = max(100 - equity_pct - bond_pct, 10)
    
    return {
        'stocks': f"{equity_pct:.0f}%",
        'bonds': f"{bond_pct:.0f}%", 
        'cash': f"{cash_pct:.0f}%"
    }

@lru_cache(maxsize=20)
def get_translator(source_lang, target_lang):
    """Cached translator loading"""
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        return pipeline("translation", model=model_name, max_length=512)
    except Exception as e:
        logger.warning(f"Translation model loading failed: {e}")
        return None

def translate_input_to_english(text):
    """Translate input to English"""
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

def translate_output_from_english(text, target_lang):
    """Translate output from English"""
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
    """Extract text from PDF"""
    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text[:3000]
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return f"Error reading PDF: {str(e)}"

# Multi-Agent System States
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
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Enhanced Agents
def generate_sql_agent(state: FinancialAgentState):
    """SQL Query Generation Agent"""
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
        sql_queries = re.findall(r'```sql\n(.*?)\n```', response.content, re.DOTALL)
        if not sql_queries:
            sql_queries = [f"SELECT * FROM companies WHERE sector LIKE '%technology%' ORDER BY market_cap DESC;"]
        state["sql_queries"] = sql_queries
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        state["sql_queries"] = ["SELECT * FROM companies ORDER BY market_cap DESC LIMIT 10;"]
    
    return state

def data_integration_agent(state: FinancialAgentState):
    """Data Integration Agent"""
    query = state["query"]
    
    all_companies = get_sector_stocks(query)
    relevant_companies = []
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['safe', 'low risk', 'stable', 'conservative']):
        relevant_companies = [c for c in all_companies if c['risk'] == 'Low']
    elif any(word in query_lower for word in ['growth', 'high return', 'aggressive']):
        relevant_companies = [c for c in all_companies if c['risk'] == 'High']
    else:
        relevant_companies = all_companies
    
    if any(word in query_lower for word in ['under $100', 'below 100', 'cheap']):
        relevant_companies = [c for c in relevant_companies if c['price'] < 100]
    elif any(word in query_lower for word in ['under $50', 'below 50']):
        relevant_companies = [c for c in relevant_companies if c['price'] < 50]
    
    if any(word in query_lower for word in ['low pe', 'undervalued']):
        relevant_companies = [c for c in relevant_companies if c['pe_ratio'] and c['pe_ratio'] < 20]
    
    if not relevant_companies:
        relevant_companies = sorted(all_companies, key=lambda x: x.get('market_cap', 0), reverse=True)
    
    state["company_data"] = relevant_companies[:5]
    
    analysis_prompt = f"""
    Analyze these real companies: {[{
        'name': c['name'], 
        'sector': c['sector'], 
        'market_cap': f"${c['market_cap']/1000000000:.1f}B" if c['market_cap'] > 1000000000 else f"${c['market_cap']/1000000:.1f}M",
        'pe_ratio': c['pe_ratio'],
        'price': c['price']
    } for c in relevant_companies[:3]]}
    
    Provide market analysis (2-3 sentences):
    1. Current sector trends
    2. Risk assessment
    3. Growth potential
    """
    
    try:
        analysis_response = llm.invoke([{"role": "user", "content": analysis_prompt}])
        state["market_analysis"] = analysis_response.content
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        state["market_analysis"] = "Market analysis unavailable due to technical issues."
    
    return state

def recommendation_agent(state: FinancialAgentState):
    """Investment Recommendation Agent"""
    companies = state["company_data"]
    analysis = state["market_analysis"]
    query = state["query"]
    
    rec_prompt = f"""
    User Query: "{query}"
    Market Analysis: {analysis}
    
    Real Company Data:
    {json.dumps([{
        'name': c['name'],
        'symbol': c['symbol'], 
        'sector': c['sector'],
        'market_cap': c['market_cap'],
        'pe_ratio': c['pe_ratio'],
        'current_price': c['price'],
        'risk_level': c['risk'],
        'beta': c['beta']
    } for c in companies], indent=2)}
    
    Generate 3 investment recommendations using REAL data:
    1. Company name and rationale
    2. Risk/return assessment with real metrics
    3. Recommended allocation
    4. Key financial metrics
    5. Investment thesis
    
    Format as actionable advice with real numbers.
    """
    
    try:
        response = llm.invoke([{"role": "user", "content": rec_prompt}])
        
        recommendations = []
        for i, company in enumerate(companies[:3]):
            recommendations.append({
                "company": company["name"],
                "symbol": company["symbol"],
                "sector": company["sector"],
                "allocation": f"{40-i*10}%",
                "risk": company["risk"],
                "price": company["price"],
                "market_cap": company["market_cap"],
                "pe_ratio": company["pe_ratio"],
                "rationale": f"P/E: {company['pe_ratio']}, Market cap: ${company['market_cap']/1000000000:.1f}B, Beta: {company['beta']}"
            })
        
        state["recommendations"] = recommendations
        state["final_response"] = response.content
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        state["final_response"] = "Unable to generate recommendations due to technical issues."
        state["recommendations"] = []
    
    return state

def create_financial_workflow():
    """Create financial agent workflow"""
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
    """Check if query is financial"""
    financial_keywords = ['invest', 'stock', 'portfolio', 'finance', 'company', 'market', 'buy', 'sell', 'recommendation', 'risk', 'return', 'shares']
    return any(keyword in query.lower() for keyword in financial_keywords)

def run_customer_support(query: str,
                        force_language=None,
                        chat_history: List[Dict[str, str]] = [],
                        file_path: Optional[str] = None,
                        user_id: str = "anonymous_user") -> Dict[str, str]:
    
    # Initialize global instances
    initialize_global_instances()
    
    # Rate limiting
    allowed, remaining = rate_limiter.is_allowed(user_id)
    if not allowed:
        return {
            "original_language": "English",
            "response": "⚠️ Rate limit exceeded. Please try again later or upgrade to premium.",
            "raw_response": "Rate limit exceeded"
        }
    
    # Input validation
    try:
        query = validate_query(query)
    except ValueError as e:
        return {
            "original_language": "English", 
            "response": f"❌ Invalid input: {str(e)}",
            "raw_response": str(e)
        }
    
    translated_query, detected_lang = translate_input_to_english(query)
    
    # Get conversation context
    context = conversation_memory.get_context_summary(user_id)
    
    if is_financial_query(translated_query):
        # Financial workflow
        workflow = create_financial_workflow()
        
        initial_state = {
            "query": translated_query,
            "user_profile": {},
            "user_requirements": {},
            "sql_queries": [],
            "company_data": [],
            "market_analysis": "",
            "fundamental_analysis": "",
            "technical_analysis": [],
            "recommendations": [],
            "final_response": ""
        }
        
        try:
            result = workflow.invoke(initial_state)
            response = result["final_response"]
            
            # Add demo information
            demo_info = f"\n\n🔧 **Live Financial Analysis System:**\n"
            demo_info += f"**SQL Queries Generated:** {len(result['sql_queries'])}\n"
            demo_info += f"**Real-time Data Sources:** Yahoo Finance, Live Market Data\n"
            demo_info += f"**Companies Analyzed:** {len(result['company_data'])}\n"
            demo_info += f"**Recommendations:** {len(result['recommendations'])}\n\n"
            
            demo_info += "**Live Stock Recommendations:**\n"
            for i, rec in enumerate(result['recommendations'], 1):
                demo_info += f"{i}. **{rec['company']}** ({rec['symbol']}) - ${rec['price']}/share\n"
                demo_info += f"   • {rec['allocation']} allocation, {rec['risk']} risk\n"
                demo_info += f"   • {rec['rationale']}\n\n"
            
            response += demo_info
            
        except Exception as e:
            logger.error(f"Financial workflow failed: {e}")
            response = "Sorry, the financial analysis system is temporarily unavailable. Please try again later."
    
    else:
        # Regular support
        history_text = "\n".join(f"User: {msg['user']}\nAgent: {msg['agent']}" for msg in chat_history[-3:])
        file_summary = extract_text_from_pdf(file_path) if file_path else ""
        
        full_prompt = f"""
        You are a professional, helpful AI assistant. Provide clear and accurate responses.
        
        {f'Previous conversation context:\n{context}\n' if context else ''}
        {f'Recent chat history:\n{history_text}\n' if history_text else ''}
        {f'File content summary:\n{file_summary}\n' if file_summary else ''}
        
        Current query: {translated_query}
        
        Provide a helpful, professional response. If this relates to previous conversation, acknowledge the context.
        """
        
        try:
            response = llm.invoke([{"role": "user", "content": full_prompt}]).content
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            response = "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."
    
    # Store conversation
    conversation_memory.add_exchange(user_id, query, response)
    
    # Translate response
    target_lang = force_language if force_language else detected_lang
    translated_response = translate_output_from_english(response, target_lang)
    
    return {
        "original_language": SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "response": translated_response,
        "raw_response": response,
        "remaining_requests": remaining
    }