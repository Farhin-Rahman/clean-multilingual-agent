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
# In app.py

# DELETE your entire old CSS block and REPLACE it with this one.
st.markdown("""
<style>
    /* Import a font that is close to the new ChatGPT font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Hide Streamlit's default elements */
    #MainMenu, .stDeployButton, footer, header {
        visibility: hidden;
    }

    /* Modern ChatGPT "ombre" background */
    .main {
        background: radial-gradient(circle at 20% 20%, rgba(200, 180, 255, 0.1), transparent 30%),
                    radial-gradient(circle at 80% 80%, rgba(180, 220, 255, 0.1), transparent 30%),
                    #FFFFFF; /* Fallback solid color */
        font-family: 'Inter', sans-serif;
    }

    /* Center the main chat content */
    .main .block-container {
        max-width: 52rem; /* Slightly wider for better look */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Chat header styling */
    .chat-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .chat-header h1 {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1f2937;
    }
    .chat-header p {
        font-size: 1rem;
        color: #6b7280;
    }
    
    /* Message container styling */
    .user-message, .assistant-message {
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .user-message {
        background-color: #F3F4F6; /* Light gray for user */
        color: #1f2937;
        margin-left: 20%;
    }

    .assistant-message {
        background-color: #FFFFFF;
        color: #111827;
        margin-right: 20%;
        border: 1px solid #E5E7EB;
    }
    
    /* Input area styling to match modern ChatGPT */
    .stTextArea textarea {
        border: 1px solid #D1D5DB !important;
        border-radius: 1rem !important;
        padding: 1rem !important;
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-size: 1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        resize: none !important;
    }
    .stTextArea textarea:focus {
        outline: none !important;
        border-color: #6366F1 !important; /* A nice purple/blue focus color */
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Send button styling */
    .stButton > button {
        background-color: #4F46E5 !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #4338CA !important;
    }

    /* Sidebar styling */
    .css-1d391kg { /* This is Streamlit's class for the sidebar */
         background-color: #F9FAFB !important; /* Very light gray sidebar */
         border-right: 1px solid #E5E7EB;
    }
    .sidebar .sidebar-content {
        background-color: #F9FAFB !important;
    }
    .sidebar h3, .sidebar .stSelectbox label, .sidebar .stCheckbox label, .sidebar .stButton > button {
        color: #374151 !important; /* Darker text for light background */
    }
    .sidebar .stButton > button {
        border: 1px solid #D1D5DB !important;
        background-color: #FFFFFF !important;
        width: 100% !important;
    }
    .sidebar .stButton > button:hover {
        background-color: #F3F4F6 !important;
    }

    /* Remove the "Drag and drop file here" box border for a cleaner look */
    .stFileUploader {
        border: none !important;
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    /* Demo status styling */
    .demo-active {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #a7f3d0;
        font-size: 0.875rem;
        margin: 1rem 0;
    }

    /* Footer styling */
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