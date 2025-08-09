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
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styling */
.main {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main > div {
    padding-top: 2rem;
}

/* ChatGPT-like header */
.chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.chat-header h1 {
    margin: 0;
    font-weight: 600;
    font-size: 2.2rem;
}

.chat-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
    font-size: 1rem;
}

/* Chat messages */
.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 5px 18px;
    margin: 1rem 0;
    margin-left: 20%;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    animation: slideInRight 0.3s ease;
}

.assistant-message {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    color: #333;
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 18px 5px;
    margin: 1rem 0;
    margin-right: 20%;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #667eea;
    animation: slideInLeft 0.3s ease;
}

@keyframes slideInRight {
    from { transform: translateX(100px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes slideInLeft {
    from { transform: translateX(-100px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Input area styling */
.stTextArea textarea {
    border-radius: 25px;
    border: 2px solid #e9ecef;
    padding: 1rem 1.5rem;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: white;
}

.stTextArea textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Button styling */
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Info boxes */
.stInfo {
    background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
    color: white;
    border-radius: 10px;
    border: none;
}

/* Audio player styling */
audio {
    width: 100%;
    margin: 0.5rem 0;
}

/* Financial demo styling */
.demo-active {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* Footer styling */
.footer {
    background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 2rem;
}

/* Voice recorder styling */
.voice-recorder {
    background: linear-gradient(135deg, #fd7e14 0%, #e63946 100%);
    border-radius: 15px;
    padding: 1rem;
    margin: 1rem 0;
    color: white;
}

/* Spinner customization */
.stSpinner > div {
    border-top-color: #667eea !important;
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
st.markdown("""
<div class="chat-header">
    <h1>🤖 AI Financial Assistant</h1>
    <p>Your intelligent multilingual financial advisor powered by real-time market data</p>
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