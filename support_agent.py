# Updated support_agent.py with context-aware support and file summarization

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

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env or Streamlit secrets.")

SUPPORTED_LANGUAGES = {
    "bn": "Bengali",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "en": "English"
}

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
        return text[:2000]  # Truncate for LLM input limit
    except Exception as e:
        return "Error reading PDF: " + str(e)

class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str
    history: Optional[str]
    file_summary: Optional[str]

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

def build_context_prompt(query: str, history: str = "", file_summary: str = "") -> str:
    return f"""
You are a friendly, intelligent assistant. Keep responses clear and helpful. Avoid email-like phrases like 'Dear user' or 'Best regards'. Use a conversational tone like ChatGPT.
{f'Here is the conversation history:\n{history}\n' if history else ''}
{f'This is a summary of the user-uploaded file:\n{file_summary}\n' if file_summary else ''}

Now respond to the current query: {query}
"""

def run_customer_support(query: str, 
                         force_language=None, 
                         chat_history: List[Dict[str, str]] = [],
                         file_path: Optional[str] = None) -> Dict[str, str]:

    translated_query, detected_lang = translate_input_to_english(query)

    # Prepare history
    history_text = "\n".join(f"User: {msg['user']}\nAgent: {msg['agent']}" for msg in chat_history[-3:])

    # Prepare file summary
    file_summary = extract_text_from_pdf(file_path) if file_path else ""

    full_prompt = build_context_prompt(translated_query, history_text, file_summary)

    prompt = ChatPromptTemplate.from_template(full_prompt)
    response = (prompt | llm).invoke({}).content.strip()

    target_lang = force_language if force_language else detected_lang
    translated_response = translate_output_from_english(response, target_lang)

    return {
        "original_language": SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
        "response": translated_response,
        "raw_response": response
    }