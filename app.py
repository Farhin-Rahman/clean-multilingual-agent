# app.py — Streamlit UI for the Financial Support Agent (free-first)

import os
import base64
import tempfile
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from support_agent import run_customer_support, is_financial_query

# ---------- Optional voice/TTS dependencies (guarded) ----------
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


# ============================ Setup ============================
load_dotenv()
st.set_page_config(
    page_title="AI Financial Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖",
)

# ============================ Styles ===========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
#MainMenu, .stDeployButton, footer, header { visibility: hidden; }
.main { background: radial-gradient(circle at 20% 20%, rgba(200,180,255,.1), transparent 30%),
                    radial-gradient(circle at 80% 80%, rgba(180,220,255,.1), transparent 30%), #FFFFFF;
        font-family: 'Inter', sans-serif; }
.main .block-container { max-width: 52rem; padding-top: 2rem; padding-bottom: 2rem; }
.chat-header { text-align: center; padding: 1rem 0 2rem 0; }
.chat-header h1 { font-size: 2.2rem; font-weight: 600; color: #1f2937; }
.chat-header p { font-size: 1rem; color: #6b7280; }
.user-message, .assistant-message { padding: 1rem 1.5rem; border-radius: 1rem; margin-bottom: 1rem;
    line-height: 1.6; font-size: .95rem; box-shadow: 0 2px 8px rgba(0,0,0,.05); word-break: break-word; }
.user-message { background-color: #F3F4F6; color: #1f2937; margin-left: 20%; }
.assistant-message { background-color: #FFFFFF; color: #111827; margin-right: 20%; border: 1px solid #E5E7EB; }
.stTextArea textarea { border: 1px solid #D1D5DB !important; border-radius: 1rem !important; padding: 1rem !important;
    background-color: #FFFFFF !important; color: #111827 !important; font-size: 1rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.05) !important; resize: none !important; }
.stTextArea textarea:focus { outline: none !important; border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,.4) !important; }
.stButton > button { background-color: #4F46E5 !important; color: white !important; border: none !important;
    border-radius: .5rem !important; padding: .6rem 1.2rem !important; font-size: 1rem !important; font-weight: 500 !important;
    cursor: pointer !important; transition: background-color .2s ease !important; }
.stButton > button:hover { background-color: #4338CA !important; }
.demo-active { background-color: #d1fae5; color: #065f46; padding: .75rem 1rem; border-radius: .5rem;
    border: 1px solid #a7f3d0; font-size: .875rem; margin: 1rem 0; }
.footer { background-color: #f7f7f8; border: 1px solid #e5e5e5; color: #8e8ea0; padding: .75rem 1rem; border-radius: .5rem;
    text-align: center; font-size: .8125rem; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ========================== Utilities ==========================
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ----- Whisper (optional) -----
if HAVE_WHISPER:
    @st.cache_resource
    def load_whisper_model():
        size = os.getenv("WHISPER_MODEL", "base")
        compute = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
        return WhisperModel(size, compute_type=compute)

    def transcribe_audio(audio_bytes, sample_rate):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                tmp_path = tmp.name
            model = load_whisper_model()
            segments, _ = model.transcribe(tmp_path)
            return "".join([seg.text for seg in segments]).strip()
        except Exception:
            return ""
else:
    def transcribe_audio(audio_bytes, sample_rate):
        return ""


def generate_tts_audio(text: str) -> str:
    if not HAVE_GTTS or not text:
        return ""
    try:
        t = text if len(text) <= 800 else (text[:800] + " …")
        tts = gTTS(t)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
        with open(tmp.name, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        return f"""
        <audio controls style="width: 100%; margin: 0.5rem 0;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    except Exception:
        return "<p><em>Audio generation temporarily unavailable</em></p>"


# ======================= Session State =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: "user"|"agent", content: str}]

if "query" not in st.session_state:
    st.session_state.query = ""

if "clear_query" not in st.session_state:
    st.session_state.clear_query = False

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_" + hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:12]

if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False


# =========================== Sidebar ===========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    display_language = st.selectbox(
        "Response language:",
        ["Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"],
    )

    uploaded_file = st.file_uploader("📎 Upload file (PDF or TXT)", type=["pdf", "txt"])

    st.markdown("### 📊 Financial Analysis Settings")
    st.session_state.is_financial_mode = st.checkbox(
        "Enable Financial Analysis Mode",
        value=True,
        help="When enabled, your query will trigger the full agentic workflow for stock analysis.",
    )
    if st.session_state.is_financial_mode:
        st.session_state.risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate",
        )
        st.session_state.investment_timeline = st.select_slider(
            "Investment Timeline",
            options=["Short-term", "Medium-term", "Long-term"],
            value="Medium-term",
        )

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

    st.markdown("---")
    st.markdown("### 🛠️ Database Management")
    if st.button("Pre-load Company Database"):
        try:
            from populate_db import populate_company_data
            with st.spinner("Fetching data for ~500 S&P companies... This may take several minutes."):
                populate_company_data()
            st.success("Database populated successfully!")
        except Exception as e:
            st.error(f"Failed to populate database: {e}")


# ========================== Header ============================
st.markdown(
    """
<div class="chat-header">
  <h1>AI Financial Assistant</h1>
  <p>Agentic analysis with citations • Multi-RAG (quant/qual/logical) • Free-first (Ollama + FinBERT)</p>
</div>
""",
    unsafe_allow_html=True,
)

# Demo status banner
if any(
    any(k in (msg.get("content", "").lower()) for k in ["invest", "stock", "company", "finance"])
    for msg in st.session_state.messages
):
    st.markdown(
        """
<div class="demo-active">
  🤖 <strong>Agentic Pipeline Active</strong> – SQL → Live metrics → Text/Logic/Numeric RAG → Verification → MCDM + Rules
</div>
""",
        unsafe_allow_html=True,
    )

# ========================== Chat Log ==========================
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message"><strong>You:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message"><strong>🤖 Assistant:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            html_audio = generate_tts_audio(msg["content"])
            if html_audio:
                st.markdown(html_audio, unsafe_allow_html=True)

# ========================= Voice Input ========================
st.markdown("### 🎙️ Voice Input")
if MIC_AVAILABLE:
    audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹ Stop", just_once=True)
    if audio:
        st.audio(audio["bytes"], format="audio/wav")
        if HAVE_WHISPER:
            transcribed = transcribe_audio(audio["bytes"], audio.get("sample_rate"))
            if transcribed:
                st.session_state.query = transcribed
                st.success("✅ Voice transcription ready!")
        else:
            st.info("Install faster-whisper to enable automatic transcription.")
else:
    st.caption("Tip: install streamlit-mic-recorder to enable voice input.")


# ================== Text Input + Submit =======================
query = st.text_area(
    "💬 Your message:",
    value=st.session_state.query,
    height=120,
    key="query",
    placeholder="Ask about investments, companies, or screening…",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Send Message", use_container_width=True):
        if query.strip():
            st.session_state.messages.append({"role": "user", "content": query})

            # ----- File handling -----
            file_path = None
            appended_text = ""
            uploaded = st.session_state.get("uploaded_file_memo")  # not used; kept for clarity
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in ("-", "_", "."))
                    save_path = UPLOAD_DIR / safe_name
                    with save_path.open("wb") as f:
                        f.write(uploaded_file.read())
                    file_path = str(save_path.resolve())
                else:
                    try:
                        txt = uploaded_file.read().decode(errors="ignore")
                    except Exception:
                        txt = ""
                    if txt:
                        appended_text = f"\n\n[Attached note]\n{txt[:2000]}"

            # ----- Chat history pairs -----
            chat_history = []
            msgs = st.session_state.messages
            for i in range(0, len(msgs) - 1):
                if msgs[i]["role"] == "user":
                    agent_reply = msgs[i + 1]["content"] if i + 1 < len(msgs) and msgs[i + 1]["role"] != "user" else ""
                    chat_history.append({"user": msgs[i]["content"], "agent": agent_reply})

            # ----- Profile & weights -----
            user_profile = {}
            user_requirements = {}
            if st.session_state.get("is_financial_mode") and is_financial_query(query):
                user_profile = {
                    "risk_tolerance": st.session_state.get("risk_tolerance", "Moderate"),
                    "investment_timeline": st.session_state.get("investment_timeline", "Medium-term"),
                }
                risk = user_profile["risk_tolerance"]
                if risk == "Aggressive":
                    user_requirements["weights"] = {
                        "valuation": 0.15, "risk": 0.20, "momentum": 0.35, "size": 0.15, "sentiment": 0.15
                    }
                elif risk == "Conservative":
                    user_requirements["weights"] = {
                        "valuation": 0.30, "risk": 0.40, "momentum": 0.10, "size": 0.15, "sentiment": 0.05
                    }
                else:  # Moderate
                    user_requirements["weights"] = {
                        "valuation": 0.25, "risk": 0.30, "momentum": 0.20, "size": 0.15, "sentiment": 0.10
                    }

            # ----- Call agent -----
            with st.spinner("🤖 Analyzing with multi-RAG and live data…"):
                result = run_customer_support(
                    query=query + appended_text,
                    user_profile=user_profile,
                    user_requirements=user_requirements,
                    force_language=lang_map.get(display_language),  # safe mapping
                    chat_history=chat_history,
                    file_path=file_path,
                    user_id=st.session_state.user_id,
                )

            st.session_state.messages.append({"role": "agent", "content": result.get("response", "")})
            st.session_state.clear_query = True
            st.rerun()

# ===================== Export last answer ======================
if st.session_state.messages and st.session_state.messages[-1]["role"] == "agent":
    md_text = st.session_state.messages[-1]["content"]
    st.download_button(
        "⬇️ Download last answer (Markdown)",
        data=md_text.encode("utf-8"),
        file_name=f"recommendations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

# ============================ Footer ===========================
st.markdown(
    """
<div class="footer">
  <strong>Live features:</strong> S&amp;P500 screening • FinBERT sentiment • Ollama router (Llama↔︎Mistral) •
  Numeric/Logical/Text citations • SQL AST validation • Rule-based risk gating
</div>
""",
    unsafe_allow_html=True,
)
