# app.py — Streamlit UI for the Financial Support Agent

import os
import base64
import tempfile
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from support_agent import run_customer_support

# Optional deps (keep app usable even if they’re missing)
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    HAVE_WHISPER = True
except Exception:
    HAVE_WHISPER = False


# ───────────────────────────── Setup ─────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="AI Financial Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖",
)

# ───────────────────────────── Styles ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    #MainMenu, .stDeployButton, footer, header { visibility: hidden; }
    .main {
        background: radial-gradient(circle at 20% 20%, rgba(200,180,255,.1), transparent 30%),
                    radial-gradient(circle at 80% 80%, rgba(180,220,255,.1), transparent 30%),
                    #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    .main .block-container { max-width: 52rem; padding-top: 2rem; padding-bottom: 2rem; }
    .chat-header { text-align: center; padding: 1rem 0 2rem 0; }
    .chat-header h1 { font-size: 2.2rem; font-weight: 600; color: #1f2937; }
    .chat-header p { font-size: 1rem; color: #6b7280; }

    .user-message, .assistant-message {
        padding: 1rem 1.5rem; border-radius: 1rem; margin-bottom: 1rem;
        line-height: 1.6; font-size: .95rem; box-shadow: 0 2px 8px rgba(0,0,0,.05);
        word-break: break-word;
    }
    .user-message { background-color: #F3F4F6; color: #1f2937; margin-left: 20%; }
    .assistant-message { background-color: #FFFFFF; color: #111827; margin-right: 20%; border: 1px solid #E5E7EB; }

    .stTextArea textarea {
        border: 1px solid #D1D5DB !important; border-radius: 1rem !important; padding: 1rem !important;
        background-color: #FFFFFF !important; color: #111827 !important; font-size: 1rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,.05) !important; resize: none !important;
    }
    .stTextArea textarea:focus {
        outline: none !important; border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,.4) !important;
    }
    .stButton > button {
        background-color: #4F46E5 !important; color: white !important; border: none !important;
        border-radius: .5rem !important; padding: .6rem 1.2rem !important; font-size: 1rem !important;
        font-weight: 500 !important; cursor: pointer !important; transition: background-color .2s ease !important;
    }
    .stButton > button:hover { background-color: #4338CA !important; }

    .css-1d391kg { background-color: #F9FAFB !important; border-right: 1px solid #E5E7EB; }
    .sidebar .sidebar-content { background-color: #F9FAFB !important; }
    .sidebar h3, .sidebar .stSelectbox label, .sidebar .stCheckbox label, .sidebar .stButton > button { color: #374151 !important; }
    .sidebar .stButton > button { border: 1px solid #D1D5DB !important; background-color: #FFFFFF !important; width: 100% !important; }
    .sidebar .stButton > button:hover { background-color: #F3F4F6 !important; }

    .stFileUploader { border: none !important; background-color: #F9FAFB; padding: 1rem; border-radius: .75rem; }

    .demo-active {
        background-color: #d1fae5; color: #065f46; padding: .75rem 1rem; border-radius: .5rem;
        border: 1px solid #a7f3d0; font-size: .875rem; margin: 1rem 0;
    }
    .footer {
        background-color: #f7f7f8; border: 1px solid #e5e5e5; color: #8e8ea0;
        padding: .75rem 1rem; border-radius: .5rem; text-align: center; font-size: .8125rem; margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────── Utilities ─────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

if HAVE_WHISPER:
    @st.cache_resource
    def load_whisper_model():
        # you can tune these through env vars if you want
        size = os.getenv("WHISPER_MODEL", "base")
        compute = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
        return WhisperModel(size, compute_type=compute)

def transcribe_audio(audio_bytes, sample_rate):
    if not HAVE_WHISPER:
        return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name
    model = load_whisper_model()
    segments, _ = model.transcribe(tmp_path)
    transcription = "".join([seg.text for seg in segments])
    return transcription.strip()

def generate_tts_audio(text: str) -> str:
    if not HAVE_GTTS or not text:
        return ""
    try:
        # small guard for overly long TTS
        t = text if len(text) <= 800 else text[:800] + "..."
        tts = gTTS(t)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            with open(tmp.name, "rb") as audio_file:
                audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        return f"""
            <audio controls style="width: 100%; margin: 0.5rem 0;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
    except Exception:
        return "<p><em>Audio generation temporarily unavailable</em></p>"

def show_financial_demo():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏦 Financial Advisory")
    st.sidebar.markdown("*Real-time market analysis system*")
    demo_enabled = st.sidebar.checkbox("Enable Financial Mode", value=False)
    if demo_enabled:
        st.sidebar.markdown("#### Investment Preferences")
        st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"], key="risk_level_demo")
        st.sidebar.multiselect(
            "Preferred Sectors",
            ["Technology", "Healthcare", "Finance", "Energy", "Consumer"],
            default=["Technology"],
            key="pref_sectors_demo",
        )
        st.sidebar.number_input("Investment Amount ($)", min_value=1000, max_value=100000, value=25000, key="invest_amt_demo")
        st.sidebar.markdown("**📋 Try these queries:**")
        examples = [
            "Find safe technology stocks with low P/E ratios",
            "Show me high-growth companies under $50 per share",
            "What are undervalued healthcare stocks with good margins?",
            "Recommend stable dividend stocks in the energy sector",
            "Find growth companies with revenue growth over 15%",
        ]
        for i, ex in enumerate(examples, 1):
            if st.sidebar.button(f"💡 Example {i}", key=f"ex_{i}"):
                st.session_state.query = ex
                st.rerun()
        return True
    return False

# ───────────────────────── Session State ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "clear_query" not in st.session_state:
    st.session_state.clear_query = False
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_" + hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:12]

if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False

# ─────────────────────────── Sidebar ────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    display_language = st.selectbox(
        "Response language:",
        ["Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"],
    )

    # Keep PDF support for the backend; handle TXT locally
    uploaded_file = st.file_uploader("📎 Upload file (PDF or TXT)", type=["pdf", "txt"])

    is_financial_mode = show_financial_demo()

    lang_map = {
        "Auto (detect)": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Bengali": "bn",
        "Italian": "it",
        "Portuguese": "pt",
    }

    if st.button("🔄 Start New Conversation"):
        st.session_state.clear()
        st.rerun()

# ───────────────────────── Header ───────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>AI Financial Assistant</h1>
    <p>Intelligent financial analysis with real-time market data</p>
</div>
""", unsafe_allow_html=True)

# Demo status banner
if any(
    any(k in msg.get("content", "").lower() for k in ["invest", "stock", "company", "finance"])
    for msg in st.session_state.messages
):
    st.markdown("""
    <div class="demo-active">
        🤖 <strong>Real-Time Financial Analysis Active</strong> – Using live S&P 500 data, dynamic sector filtering, and real market metrics
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────── Chat Log ─────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
            # Inline TTS player (optional)
            html_audio = generate_tts_audio(msg["content"])
            if html_audio:
                st.markdown(html_audio, unsafe_allow_html=True)

# ───────────────────────── Voice Input ─────────────────────────
st.markdown("### 🎙️ Voice Input")
if MIC_AVAILABLE:
    audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹ Stop", just_once=True)
    if audio:
        st.audio(audio["bytes"], format="audio/wav")
        if HAVE_WHISPER:
            transcribed = transcribe_audio(audio["bytes"], audio["sample_rate"])
            if transcribed:
                st.session_state.query = transcribed
                st.success("✅ Voice transcription ready!")
        else:
            st.info("Install `faster-whisper` to enable automatic transcription.")
else:
    st.caption("Tip: install `streamlit-mic-recorder` to enable voice input.")

# ─────────────────────── Text Input + Submit ───────────────────
query = st.text_area(
    "💬 Your message:",
    value=st.session_state.query,
    height=100,
    key="query",
    placeholder="Ask me anything about investments, stocks, or financial advice…",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Send Message", use_container_width=True):
        if query.strip():
            st.session_state.messages.append({"role": "user", "content": query})

            # Handle file attachment:
            file_path = None
            appended_text = ""
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    # Save and pass to backend (it expects PDF path)
                    safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in ("-", "_", "."))
                    save_path = UPLOAD_DIR / safe_name
                    with save_path.open("wb") as f:
                        f.write(uploaded_file.read())
                    file_path = str(save_path.resolve())
                else:
                    # If it's a .txt, read locally and append to the user's query
                    txt = uploaded_file.read().decode(errors="ignore")
                    appended_text = f"\n\n[Attached note]\n{txt[:2000]}"

            # Build chat history as pairs for the agent
            chat_history = []
            last_user = None
            for m in st.session_state.messages[:-1]:  # exclude the current user message
                if m["role"] == "user":
                    last_user = m["content"]
                elif m["role"] != "user" and last_user is not None:
                    chat_history.append({"user": last_user, "agent": m["content"]})
                    last_user = None

            # Call backend
            with st.spinner("🤖 Analyzing with real-time market data…"):
                result = run_customer_support(
                    query + appended_text,
                    lang_map[display_language],
                    chat_history,
                    file_path,
                    user_id=st.session_state.user_id,
                )

            st.session_state.messages.append({"role": "agent", "content": result["response"]})
            st.session_state.clear_query = True
            st.rerun()

# ───────────────────────── Footer ──────────────────────────────
st.markdown("""
<div class="footer">
    <strong>🚀 Live Features:</strong> Real S&P 500 data • Dynamic sector analysis • Live market screening • Real-time stock metrics • Professional investment recommendations
</div>
""", unsafe_allow_html=True)
