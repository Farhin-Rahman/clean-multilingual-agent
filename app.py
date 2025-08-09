import streamlit as st
from support_agent import run_customer_support
from dotenv import load_dotenv
import os
import base64
from gtts import gTTS
import tempfile
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from datetime import datetime
import hashlib

# Load environment variables
load_dotenv()
st.set_page_config(
    page_title="AI Financial Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

# Custom CSS for ChatGPT-like styling
# Custom CSS for actual ChatGPT-like styling
st.markdown("""
<style>
/* Import ChatGPT font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Remove Streamlit branding and styling */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global styling - ChatGPT colors */
.main {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ChatGPT-like background gradient */
.main > div {
    background: linear-gradient(180deg, #f7f7f8 0%, #ffffff 100%);
    min-height: 100vh;
}

/* Hide Streamlit header */
.main > div > div > div > div > section > div {
    padding-top: 1rem;
}

/* Clean header - no flashy colors */
.chat-header {
    background: #ffffff;
    padding: 2rem 0;
    text-align: center;
    margin-bottom: 2rem;
    border-bottom: 1px solid #e5e5e5;
}

.chat-header h1 {
    color: #2d3748;
    font-weight: 600;
    font-size: 2rem;
    margin: 0;
}

.chat-header p {
    color: #6b7280;
    font-size: 0.95rem;
    margin: 0.5rem 0 0 0;
    font-weight: 400;
}

/* ChatGPT message styling */
.user-message {
    background-color: #2d3748;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 18px;
    margin: 1.5rem 0;
    max-width: 75%;
    margin-left: auto;
    margin-right: 0;
    font-size: 0.95rem;
    line-height: 1.5;
}

.assistant-message {
    background-color: #f1f3f4;
    color: #2d3748;
    padding: 1rem 1.5rem;
    border-radius: 18px;
    margin: 1.5rem 0;
    max-width: 75%;
    margin-right: auto;
    margin-left: 0;
    font-size: 0.95rem;
    line-height: 1.5;
    border: 1px solid #e5e7eb;
}

/* Clean input styling */
.stTextArea textarea {
    border: 1px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 1rem 1.5rem !important;
    font-size: 0.95rem !important;
    background-color: white !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}

.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    outline: none !important;
}

/* Simple button - no gradients */
.stButton button {
    background-color: #2d3748 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    transition: background-color 0.2s ease !important;
}

.stButton button:hover {
    background-color: #1a202c !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Clean sidebar */
.sidebar .sidebar-content {
    background-color: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}

/* Minimal info boxes */
.stInfo {
    background-color: #f0f9ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px !important;
    color: #1e40af !important;
}

/* Clean demo indicator */
.demo-active {
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.9rem;
}

/* Simple voice recorder section */
.voice-recorder {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.voice-recorder h3 {
    color: #374151;
    font-size: 1.1rem;
    margin-bottom: 1rem;
}

/* Audio player */
audio {
    width: 100%;
    margin: 1rem 0;
    height: 40px;
}

/* Clean footer */
.footer {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    color: #6b7280;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
}

/* Remove animations and transitions */
* {
    animation: none !important;
    transition: background-color 0.2s ease, border-color 0.2s ease, opacity 0.2s ease !important;
}

/* Clean selectbox */
.stSelectbox > div > div {
    background-color: white !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
}

/* Clean file uploader */
.stFileUploader {
    border: 1px dashed #d1d5db !important;
    border-radius: 8px !important;
    background-color: #fafafa !important;
}

/* Clean checkbox */
.stCheckbox {
    color: #374151 !important;
}

/* Clean number input */
.stNumberInput > div > div > input {
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    background-color: white !important;
}

/* Clean multiselect */
.stMultiSelect > div > div {
    background-color: white !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #ffffff !important;
}

.css-1d391kg h3 {
    color: #1f2937 !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# Whisper model loading (only once)
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", compute_type="auto")

def transcribe_audio(audio_bytes, sample_rate):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name

    model = load_whisper_model()
    segments, _ = model.transcribe(tmp_path)
    transcription = "".join([segment.text for segment in segments])
    return transcription.strip()

def generate_tts_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        with open(tmp.name, 'rb') as audio_file:
            audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
        <audio controls style="width: 100%; margin: 0.5rem 0;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """

def show_financial_demo():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏦 Financial Advisory")
    st.sidebar.markdown("*Real-time market analysis system*")
    
    demo_enabled = st.sidebar.checkbox("Enable Financial Mode", value=False)
    
    if demo_enabled:
        st.sidebar.markdown("#### Investment Preferences")
        risk_level = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        sectors = st.sidebar.multiselect("Preferred Sectors", 
            ["Technology", "Healthcare", "Finance", "Energy", "Consumer"], 
            default=["Technology"])
        investment_amount = st.sidebar.number_input("Investment Amount ($)", 
            min_value=1000, max_value=100000, value=25000)
        
        st.sidebar.markdown("**📋 Try these queries:**")
        example_queries = [
            "Find safe technology stocks with low P/E ratios",
            "Show me high-growth companies under $50 per share", 
            "What are undervalued healthcare stocks with good margins?",
            "Recommend stable dividend stocks in the energy sector",
            "Find growth companies with revenue growth over 15%"
        ]
        
        for i, example in enumerate(example_queries, 1):
            if st.sidebar.button(f"💡 Example {i}", key=f"ex_{i}"):
                st.session_state.query = example
                st.rerun()
        
        return True
    return False

# Session state initialization
if "query" not in st.session_state:
    st.session_state.query = ""
if "clear_query" not in st.session_state:
    st.session_state.clear_query = False
if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{hash(str(datetime.now()))}"[:12]

# Sidebar settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    display_language = st.selectbox("Response language:", [
        "Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"
    ])
    uploaded_file = st.file_uploader("📎 Upload file (PDF, TXT)", type=["pdf", "txt"])
    
    # Add financial demo
    is_financial_mode = show_financial_demo()
    
    language_mapping = {
        "Auto (detect)": None,
        "English": "en",
        "Spanish": "es", 
        "French": "fr",
        "German": "de",
        "Bengali": "bn",
        "Italian": "it",
        "Portuguese": "pt"
    }
    
    if st.button("🔄 Start New Conversation"):
        st.session_state.clear()
        st.rerun()

# Header
# Header
st.markdown("""
<div class="chat-header">
    <h1>AI Financial Assistant</h1>
    <p>Intelligent financial analysis with real-time market data</p>
</div>
""", unsafe_allow_html=True)

# Show demo status
if any('invest' in msg.get('content', '').lower() or 'stock' in msg.get('content', '').lower() 
       or 'company' in msg.get('content', '').lower() or 'finance' in msg.get('content', '').lower()
       for msg in st.session_state.messages):
    st.markdown("""
    <div class="demo-active">
        🤖 <strong>Real-Time Financial Analysis Active</strong> - Using live S&P 500 data, dynamic sector filtering, and real market metrics
    </div>
    """, unsafe_allow_html=True)

# Chat display with improved styling
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            # Audio playback
            st.markdown(generate_tts_audio(msg["content"]), unsafe_allow_html=True)

# Voice input section
st.markdown('<div class="voice-recorder">', unsafe_allow_html=True)
st.markdown("### 🎙️ Voice Input")
audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹ Stop", just_once=True)

if audio:
    st.audio(audio['bytes'], format="audio/wav")
    transcribed = transcribe_audio(audio['bytes'], audio['sample_rate'])
    st.session_state.query = transcribed
    st.success("✅ Voice transcription ready!")
st.markdown('</div>', unsafe_allow_html=True)

# Text input
query = st.text_area("💬 Your message:", value=st.session_state.query, height=100, key="query", 
                    placeholder="Ask me anything about investments, stocks, or financial advice...")

# Submit button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🚀 Send Message", use_container_width=True):
        if query.strip():
            st.session_state.messages.append({"role": "user", "content": query})
            
            file_path = None
            if uploaded_file:
                suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    file_path = tmp.name

            chat_history = []
            for i in range(0, len(st.session_state.messages) - 1, 2):
                if i + 1 < len(st.session_state.messages):
                    chat_history.append({
                        "user": st.session_state.messages[i]["content"],
                        "agent": st.session_state.messages[i + 1]["content"]
                    })

            with st.spinner("🤖 Analyzing with real-time market data..."):
                result = run_customer_support(
                    query, 
                    language_mapping[display_language], 
                    chat_history, 
                    file_path,
                    user_id=st.session_state.user_id
                )
            
            st.session_state.messages.append({"role": "agent", "content": result["response"]})
            st.session_state.clear_query = True
            st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <strong>🚀 Live Features:</strong> Real S&P 500 data • Dynamic sector analysis • Live market screening • Real-time stock metrics • Professional investment recommendations
</div>
""", unsafe_allow_html=True)