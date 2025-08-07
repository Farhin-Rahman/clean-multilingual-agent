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

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env or Streamlit secrets.")

SUPPORTED_LANGUAGES = {
    "bn": "Bengali", "es": "Spanish", "de": "German", "fr": "French", 
    "it": "Italian", "pt": "Portuguese", "en": "English"
}

# Sample financial data for demo
SAMPLE_COMPANIES = [
    {"name": "TechCorp", "sector": "Technology", "market_cap": 50000, "pe_ratio": 15, "price": 120, "risk": "Medium"},
    {"name": "HealthPlus", "sector": "Healthcare", "market_cap": 30000, "pe_ratio": 18, "price": 85, "risk": "Low"},
    {"name": "GreenEnergy", "sector": "Energy", "market_cap": 25000, "pe_ratio": 12, "price": 45, "risk": "High"},
    {"name": "FinanceFirst", "sector": "Finance", "market_cap": 40000, "pe_ratio": 14, "price": 95, "risk": "Low"},
    {"name": "InnovateTech", "sector": "Technology", "market_cap": 60000, "pe_ratio": 22, "price": 150, "risk": "High"}
]

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
    - companies (id, name, sector, market_cap, pe_ratio, price, risk_level)
    - user_preferences (user_id, risk_tolerance, preferred_sectors, investment_amount)
    - financial_metrics (company_id, revenue, profit_margin, debt_ratio, roe)
    
    Generate 2-3 relevant SQL queries that would help answer this query:
    """
    
    response = llm.invoke([{"role": "user", "content": sql_prompt}])
    
    # Extract SQL queries from response
    sql_queries = re.findall(r'```sql\n(.*?)\n```', response.content, re.DOTALL)
    if not sql_queries:
        sql_queries = [f"SELECT * FROM companies WHERE sector LIKE '%tech%' AND pe_ratio < 20;"]
    
    state["sql_queries"] = sql_queries
    return state

# Agent 2: Data Integration (A1, A2)
def data_integration_agent(state: FinancialAgentState):
    query = state["query"]
    
    # Simulate database query results with sample data
    relevant_companies = []
    
    # Simple keyword matching for demo
    query_lower = query.lower()
    for company in SAMPLE_COMPANIES:
        if (any(keyword in query_lower for keyword in ['tech', 'technology']) and company['sector'] == 'Technology') or \
           (any(keyword in query_lower for keyword in ['health', 'healthcare']) and company['sector'] == 'Healthcare') or \
           (any(keyword in query_lower for keyword in ['safe', 'low risk', 'stable']) and company['risk'] == 'Low') or \
           len(relevant_companies) < 3:
            relevant_companies.append(company)
    
    state["company_data"] = relevant_companies[:5]
    
    # Market analysis integration
    analysis_prompt = f"""
    Based on the following company data: {relevant_companies}
    
    Provide a brief market analysis focusing on:
    1. Sector trends
    2. Risk assessment
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
    Available Companies: {json.dumps(companies, indent=2)}
    
    Generate 3 personalized investment recommendations. For each recommendation, provide:
    1. Company name and why it fits the user's needs
    2. Risk level and expected returns
    3. Recommended allocation percentage
    4. Key financial metrics
    
    Format as a clear, actionable response:
    """
    
    response = llm.invoke([{"role": "user", "content": rec_prompt}])
    
    # Parse recommendations (simplified for demo)
    recommendations = []
    for i, company in enumerate(companies[:3]):
        recommendations.append({
            "company": company["name"],
            "sector": company["sector"],
            "allocation": f"{30-i*5}%",
            "risk": company["risk"],
            "rationale": f"Strong fundamentals with P/E ratio of {company['pe_ratio']}"
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
    financial_keywords = ['invest', 'stock', 'portfolio', 'finance', 'company', 'market', 'buy', 'sell', 'recommendation', 'risk', 'return']
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
        
        # Add technical details for demo
        demo_info = f"\n\n🔧 **Technical Demo Info:**\n"
        demo_info += f"**Generated SQL Queries:** {len(result['sql_queries'])} queries\n"
        demo_info += f"**Data Sources Integrated:** Company DB, Market Data, Financial Metrics\n"
        demo_info += f"**Companies Analyzed:** {len(result['company_data'])}\n"
        demo_info += f"**Recommendations Generated:** {len(result['recommendations'])}\n\n"
        
        for i, rec in enumerate(result['recommendations'], 1):
            demo_info += f"{i}. **{rec['company']}** ({rec['sector']}) - {rec['allocation']} allocation, {rec['risk']} risk\n"
        
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