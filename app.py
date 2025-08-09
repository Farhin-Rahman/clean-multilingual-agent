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
# ChatGPT-inspired clean styling
st.markdown("""
<style>
/* Import the exact font ChatGPT uses */
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600&display=swap');

/* Remove all Streamlit default styling */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}

/* ChatGPT exact background */
.main {
    background: linear-gradient(180deg, #f9f9f9 0%, #ffffff 50%, #f9f9f9 100%);
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
}

/* Remove padding */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 48rem;
}

/* Clean minimal header */
.chat-header {
    text-align: center;
    padding: 1.5rem 0;
    background: transparent;
}

.chat-header h1 {
    color: #343541;
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
}

.chat-header p {
    color: #8e8ea0;
    font-size: 0.875rem;
    margin: 0.25rem 0 0 0;
}

/* Exact ChatGPT message styling */
.user-message {
    background-color: #343541;
    color: #ffffff;
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    margin-left: 20%;
    position: relative;
    font-size: 0.9rem;
    line-height: 1.6;
}

.assistant-message {
    background-color: #f7f7f8;
    color: #343541;
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    margin-right: 20%;
    position: relative;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid #e5e5e5;
}

/* Clean input field - exactly like ChatGPT */
.stTextArea textarea {
    border: 1px solid #d9d9e3 !important;
    border-radius: 0.75rem !important;
    padding: 1rem 1rem !important;
    background-color: #ffffff !important;
    color: #343541 !important;
    font-size: 0.875rem !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    outline: none !important;
    border-color: #10a37f !important;
    box-shadow: 0 0 0 1px #10a37f !important;
}

/* ChatGPT send button */
.stButton > button {
    background-color: #10a37f !important;
    color: white !important;
    border: none !important;
    border-radius: 0.375rem !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease !important;
}

.stButton > button:hover {
    background-color: #0d9065 !important;
}

/* Sidebar exactly like ChatGPT */
.css-1d391kg {
    background-color: #171717 !important;
    color: #ffffff !important;
}

.sidebar .sidebar-content {
    background-color: #171717 !important;
    color: #ffffff !important;
}

.sidebar h3 {
    color: #ffffff !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
}

.sidebar .stSelectbox label {
    color: #ffffff !important;
    font-size: 0.875rem !important;
}

.sidebar .stCheckbox label {
    color: #ffffff !important;
    font-size: 0.875rem !important;
}

.sidebar .stButton > button {
    background-color: transparent !important;
    border: 1px solid #4d4d4f !important;
    color: #ffffff !important;
    width: 100% !important;
    margin: 0.25rem 0 !important;
}

.sidebar .stButton > button:hover {
    background-color: #2d2d30 !important;
}

/* Remove demo flashy styling */
.demo-active {
    background-color: #d1fae5;
    color: #065f46;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid #a7f3d0;
    font-size: 0.875rem;
    margin: 1rem 0;
}

/* Clean voice section */
.voice-recorder {
    background-color: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 0.75rem;
    padding: 1rem;
    margin: 1rem 0;
}

.voice-recorder h3 {
    color: #343541;
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

/* Audio controls */
audio {
    width: 100%;
    height: 32px;
}

/* Footer */
.footer {
    background-color: #f7f7f8;
    border: 1px solid #e5e5e5;
    color: #8e8ea0;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    text-align: center;
    font-size: 0.8125rem;
    margin-top: 2rem;
}

/* Remove all animations */
* {
    animation: none !important;
    transition: none !important;
}

.stButton > button,
.stTextArea textarea {
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
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
    try:
        # Limit text length for TTS (gTTS has limits)
        if len(text) > 500:
            text = text[:500] + "..."
        
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
    except Exception as e:
        return "<p><em>Audio generation temporarily unavailable</em></p>"

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