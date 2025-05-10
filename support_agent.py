from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache
from transformers import pipeline
from langdetect import detect
import os
from dotenv import load_dotenv  # Add this line
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv() 
# After load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
# Supported languages
SUPPORTED_LANGUAGES = {
    "bn": "Bengali",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "en": "English"
}

# Translator cache
@lru_cache(maxsize=20)
def get_translator(source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        return pipeline("translation", model=model_name)
    except:
        return None

def translate_output_from_english(text, target_lang):
    if target_lang == "en":
        return text
    translator = get_translator("en", target_lang)
    if translator:
        translated = translator(text, max_length=512)
        return translated[0]["translation_text"]
    return text

# State structure
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# LLM from Groq
llm = ChatGroq(
    temperature=0,
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"  # Changed from "llama-3-70b-8192" to "llama3-70b-8192"
)
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
    except Exception as e:
        print(f"Translation error: {e}")
        return text, "en"

# Node functions
def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: Technical, Billing, General.\nQuery: {query}"
    )
    category = (prompt | llm).invoke({"query": state["query"]}).content
    return {"category": category.strip()}

def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following query. Respond with 'Positive', 'Neutral', or 'Negative'.\nQuery: {query}"
    )
    sentiment = (prompt | llm).invoke({"query": state["query"]}).content
    return {"sentiment": sentiment.strip()}

def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to this query: {query}"
    )
    response = (prompt | llm).invoke({"query": state["query"]}).content
    return {"response": response.strip()}

def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to this query: {query}"
    )
    response = (prompt | llm).invoke({"query": state["query"]}).content
    return {"response": response.strip()}

def handle_general(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a general customer support response to this query: {query}"
    )
    response = (prompt | llm).invoke({"query": state["query"]}).content
    return {"response": response.strip()}

def escalate(state: State) -> State:
    return {"response": "This query has been escalated to a human agent due to negative sentiment."}

def route_query(state: State) -> str:
    if state["sentiment"] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Build LangGraph workflow
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges("analyze_sentiment", route_query, {
    "handle_technical": "handle_technical",
    "handle_billing": "handle_billing",
    "handle_general": "handle_general",
    "escalate": "escalate"
})

workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)

workflow.set_entry_point("categorize")
app = workflow.compile()

# Final callable function
# Final callable function
def run_customer_support(query: str, force_language=None) -> Dict[str, str]:
    # Translate input to English for processing
    translated_query, detected_lang = translate_input_to_english(query)
    
    # Process with LLM
    results = app.invoke({"query": translated_query})
    
    # Use forced language if specified, otherwise use detected language
    target_lang = force_language if force_language else detected_lang
    translated_response = translate_output_from_english(results["response"], target_lang)

    return {
        "original_language": SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": translated_response
    }
