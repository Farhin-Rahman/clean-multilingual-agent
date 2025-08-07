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

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env or Streamlit secrets.")

SUPPORTED_LANGUAGES = {
    "bn": "Bengali", "es": "Spanish", "de": "German", "fr": "French", 
    "it": "Italian", "pt": "Portuguese", "en": "English"
}

# REAL DATA FUNCTIONS
def get_real_stock_data(symbols=['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'JPM', 'PFE', 'XOM', 'NVDA', 'TSLA', 'V']):
    """Get real stock data from Yahoo Finance"""
    real_companies = []
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty and info:
                current_price = round(hist['Close'].iloc[-1], 2)
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 0)
                sector = info.get('sector', 'Unknown')
                name = info.get('longName', symbol)
                
                # Simple risk calculation based on beta
                beta = info.get('beta', 1.0)
                if beta and beta < 0.8:
                    risk = "Low"
                elif beta and beta > 1.2:
                    risk = "High" 
                else:
                    risk = "Medium"
                
                real_companies.append({
                    "name": name,
                    "symbol": symbol,
                    "sector": sector,
                    "market_cap": market_cap,
                    "pe_ratio": round(pe_ratio, 2) if pe_ratio else 0,
                    "price": current_price,
                    "risk": risk,
                    "beta": beta if beta else 1.0
                })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    
    return real_companies

def get_sector_stocks(sector_query):
    """Get stocks based on sector"""
    sector_symbols = {
        'technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA'],
        'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO'],
        'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
        'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
        'consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP']
    }
    
    query_lower = sector_query.lower()
    symbols = []
    
    if any(word in query_lower for word in ['tech', 'technology', 'software']):
        symbols = sector_symbols['technology']
    elif any(word in query_lower for word in ['health', 'healthcare', 'medical', 'pharma']):
        symbols = sector_symbols['healthcare']  
    elif any(word in query_lower for word in ['finance', 'bank', 'financial']):
        symbols = sector_symbols['finance']
    elif any(word in query_lower for word in ['energy', 'oil', 'gas']):
        symbols = sector_symbols['energy']
    elif any(word in query_lower for word in ['consumer', 'retail']):
        symbols = sector_symbols['consumer']
    else:
        # Default mix
        symbols = ['AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM']
    
    return get_real_stock_data(symbols)

@lru_cache(maxsize=20)
def get_translator(source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        return pipeline("translation", model=model_name)
    except:
        return None

def translate_input_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text, "en"
        elif detected_lang in SUPPORTED_LANGUAGES:
            translator = get_translator(detected_lang, "en")
            if translator:
                translated = translator(text, max_length=512)
                return translated[0]["translation_text"], detected_lang
        return text, "en"
    except:
        return text, "en"

def translate_output_from_english(text, target_lang):
    if target_lang == "en":
        return text
    translator = get_translator("en", target_lang)
    if translator:
        translated = translator(text, max_length=512)
        return translated[0]["translation_text"]
    return text

def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        return text[:2000]
    except Exception as e:
        return "Error reading PDF: " + str(e)

# Multi-Agent System States
class FinancialAgentState(TypedDict):
    query: str
    user_requirements: Dict
    sql_queries: List[str]
    company_data: List[Dict]
    market_analysis: str
    recommendations: List[Dict]
    final_response: str

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Agent 1: SQL Query Generation (A3)
def generate_sql_agent(state: FinancialAgentState):
    query = state["query"]
    
    sql_prompt = f"""
    You are a SQL expert. Convert this natural language query to SQL:
    Query: "{query}"
    
    Available database schema:
    - companies (symbol, name, sector, market_cap, pe_ratio, price, risk_level, beta)
    - user_preferences (user_id, risk_tolerance, preferred_sectors, investment_amount)
    - financial_metrics (symbol, revenue, profit_margin, debt_ratio, roe, dividend_yield)
    
    Generate 2-3 relevant SQL queries that would help answer this query:
    """
    
    response = llm.invoke([{"role": "user", "content": sql_prompt}])
    
    # Extract SQL queries from response
    sql_queries = re.findall(r'```sql\n(.*?)\n```', response.content, re.DOTALL)
    if not sql_queries:
        sql_queries = [f"SELECT * FROM companies WHERE sector LIKE '%technology%' AND pe_ratio < 25 ORDER BY market_cap DESC;"]
    
    state["sql_queries"] = sql_queries
    return state

# Agent 2: Data Integration (A1, A2)
def data_integration_agent(state: FinancialAgentState):
    query = state["query"]
    
    # Get real data based on query
    all_companies = get_sector_stocks(query)
    relevant_companies = []
    
    query_lower = query.lower()
    
    # Risk-based filtering
    if any(word in query_lower for word in ['safe', 'low risk', 'stable', 'conservative']):
        relevant_companies = [c for c in all_companies if c['risk'] == 'Low']
    elif any(word in query_lower for word in ['growth', 'high return', 'aggressive']):
        relevant_companies = [c for c in all_companies if c['risk'] == 'High']
    else:
        relevant_companies = all_companies
    
    # Price filtering
    if any(word in query_lower for word in ['under $100', 'below 100', 'cheap']):
        relevant_companies = [c for c in relevant_companies if c['price'] < 100]
    elif any(word in query_lower for word in ['under $50', 'below 50']):
        relevant_companies = [c for c in relevant_companies if c['price'] < 50]
    
    # P/E ratio filtering
    if any(word in query_lower for word in ['low pe', 'undervalued']):
        relevant_companies = [c for c in relevant_companies if c['pe_ratio'] and c['pe_ratio'] < 20]
    
    # If no specific filtering, return top companies by market cap
    if not relevant_companies:
        relevant_companies = sorted(all_companies, key=lambda x: x.get('market_cap', 0), reverse=True)
    
    state["company_data"] = relevant_companies[:5]
    
    # Market analysis integration
    analysis_prompt = f"""
    Based on the following real company data: {[{
        'name': c['name'], 
        'sector': c['sector'], 
        'market_cap': f"${c['market_cap']/1000000000:.1f}B" if c['market_cap'] > 1000000000 else f"${c['market_cap']/1000000:.1f}M",
        'pe_ratio': c['pe_ratio'],
        'price': c['price']
    } for c in relevant_companies[:3]]}
    
    Provide a brief market analysis focusing on:
    1. Current sector trends
    2. Risk assessment based on real metrics
    3. Growth potential
    
    Keep it concise (2-3 sentences):
    """
    
    analysis_response = llm.invoke([{"role": "user", "content": analysis_prompt}])
    state["market_analysis"] = analysis_response.content
    
    return state

# Agent 3: Investment Recommendation (A4)
def recommendation_agent(state: FinancialAgentState):
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
    
    Generate 3 personalized investment recommendations using REAL data. For each:
    1. Company name (symbol) and why it fits user needs
    2. Risk level and expected returns based on real metrics
    3. Recommended allocation percentage
    4. Key financial metrics (real P/E, market cap, price)
    5. Current market position
    
    Format as a clear, actionable response with real financial data:
    """
    
    response = llm.invoke([{"role": "user", "content": rec_prompt}])
    
    # Parse recommendations with real data
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
            "rationale": f"Real P/E ratio of {company['pe_ratio']}, Market cap: ${company['market_cap']/1000000000:.1f}B"
        })
    
    state["recommendations"] = recommendations
    state["final_response"] = response.content
    
    return state

# Agent Orchestration
def create_financial_workflow():
    workflow = StateGraph(FinancialAgentState)
    
    # Add agents
    workflow.add_node("sql_agent", generate_sql_agent)
    workflow.add_node("data_agent", data_integration_agent)  
    workflow.add_node("recommendation_agent", recommendation_agent)
    
    # Define flow
    workflow.set_entry_point("sql_agent")
    workflow.add_edge("sql_agent", "data_agent")
    workflow.add_edge("data_agent", "recommendation_agent")
    workflow.add_edge("recommendation_agent", END)
    
    return workflow.compile()

def is_financial_query(query: str) -> bool:
    financial_keywords = ['invest', 'stock', 'portfolio', 'finance', 'company', 'market', 'buy', 'sell', 'recommendation', 'risk', 'return', 'shares']
    return any(keyword in query.lower() for keyword in financial_keywords)

def run_customer_support(query: str,
                        force_language=None,
                        chat_history: List[Dict[str, str]] = [],
                        file_path: Optional[str] = None) -> Dict[str, str]:
    
    translated_query, detected_lang = translate_input_to_english(query)
    
    # Check if it's a financial query
    if is_financial_query(translated_query):
        # Run multi-agent financial system
        workflow = create_financial_workflow()
        
        initial_state = {
            "query": translated_query,
            "user_requirements": {},
            "sql_queries": [],
            "company_data": [],
            "market_analysis": "",
            "recommendations": [],
            "final_response": ""
        }
        
        result = workflow.invoke(initial_state)
        response = result["final_response"]
        
        # Add technical details for demo with REAL data
        demo_info = f"\n\n🔧 **Live Technical Demo - Real Market Data:**\n"
        demo_info += f"**Generated SQL Queries:** {len(result['sql_queries'])} queries\n"
        demo_info += f"**Live Data Sources:** Yahoo Finance API, Real-time Stock Data\n"
        demo_info += f"**Companies Analyzed:** {len(result['company_data'])} (Live Market Data)\n"
        demo_info += f"**Recommendations Generated:** {len(result['recommendations'])}\n\n"
        
        demo_info += "**Real Stock Recommendations:**\n"
        for i, rec in enumerate(result['recommendations'], 1):
            demo_info += f"{i}. **{rec['company']}** ({rec['symbol']}) - ${rec['price']}/share\n"
            demo_info += f"   • {rec['allocation']} allocation, {rec['risk']} risk\n"
            demo_info += f"   • Market Cap: ${rec['market_cap']/1000000000:.1f}B, P/E: {rec['pe_ratio']}\n\n"
        
        response += demo_info
        
    else:
        # Regular support flow
        history_text = "\n".join(f"User: {msg['user']}\nAgent: {msg['agent']}" for msg in chat_history[-3:])
        file_summary = extract_text_from_pdf(file_path) if file_path else ""
        
        full_prompt = f"""
        You are a friendly, intelligent assistant. Keep responses clear and helpful.
        {f'Conversation history:\n{history_text}\n' if history_text else ''}
        {f'File summary:\n{file_summary}\n' if file_summary else ''}
        
        Current query: {translated_query}
        """
        
        response = llm.invoke([{"role": "user", "content": full_prompt}]).content
    
    target_lang = force_language if force_language else detected_lang
    translated_response = translate_output_from_english(response, target_lang)
    
    return {
        "original_language": SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "response": translated_response,
        "raw_response": response
    }