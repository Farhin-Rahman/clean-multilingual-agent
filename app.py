# ========================== Early env load ==========================
from dotenv import load_dotenv, find_dotenv
import os, pathlib

# Load .env from the project root ASAP (before any other imports use env vars)
load_dotenv(find_dotenv(usecwd=True))

# Ensure SEC_USER_AGENT is set (quiet the SEC warning from downstream code)
os.environ.setdefault(
    "SEC_USER_AGENT",
    os.getenv("SEC_USER_AGENT", "Streamlit App nobody@example.com"),
)

# Quick probe so you can see what's happening in the terminal
print("CWD:", pathlib.Path().resolve())
print("Has GROQ_API_KEY?", bool(os.getenv("GROQ_API_KEY")))
print("Has SEC_USER_AGENT?", bool(os.getenv("SEC_USER_AGENT")))

# ============================= Imports ==============================
import base64
import tempfile
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
SAMPLE_IMAGE = ROOT / "uploads" / "images" / "analysis-of-financial-statements-3-1002x1024.png"

import streamlit as st

from support_agent import run_customer_support, is_financial_query
# SAFE import of the preset
try:
    from support_agent import ACADEMIC_PRESET
except Exception:
    ACADEMIC_PRESET = None
from services.provider_router import healthcheck, routing_mode
from services.charts import create_all_charts
from rag.attachments import AttachmentIndex, process_attachment_file
from services.files import extract_text_from_file, split_chunks
from services.ocr import extract_text_from_image, is_image_file

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
st.set_page_config(
    page_title="AI Financial Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed for everyone by default
    page_icon="🤖",
)

# Optional: set to True to hide sidebar completely (no hamburger toggle)
HIDE_SIDEBAR = False
if HIDE_SIDEBAR:
    st.markdown("""
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# ===================== Enrichment / Normalizer ======================
def _fmt_dy(dy):
    if not isinstance(dy, (int, float)) or dy <= 0:
        return "—"
    pct = dy*100 if dy < 1 else dy  # allow 0.017 or 1.7 to render as 1.7%
    pct = max(0.0, min(pct, 20.0))  # clamp to 20% to avoid absurd values
    return f"{pct:.1f}%"

def _enrich_recommendations(markdown_text: str) -> str:
    import re
    import numpy as np
    import pandas as pd
    import yfinance as yf  # shimmed by sitecustomize when FREE_DEMO=1

    # Fast exit: if academic sections or header already present, don't add another header
    if "Assumptions" in markdown_text or "Limitations" in markdown_text or "### Safe stock suggestions" in markdown_text:
        cleaned = "\n".join([ln for ln in markdown_text.splitlines() if not ln.strip().startswith("Note:")])
        return cleaned

    risk_label = st.session_state.get("risk_tolerance", "Moderate")

    def _rsi(series: "pd.Series", n: int = 14):
        if series is None or len(series) < n + 5:
            return None
        d = series.diff()
        up = d.clip(lower=0.0)
        down = -d.clip(upper=0.0)
        roll_up = up.rolling(n).mean()
        roll_down = down.rolling(n).mean()
        rs = roll_up / (roll_down + 1e-12)
        rsi = 100 - (100 / (1 + rs))
        try:
            return float(rsi.iloc[-1])
        except Exception:
            return None

    def _mk_pick_line(name, sym, price, score, conf):
        price_part = f"${price:,.2f}" if isinstance(price, (int, float)) else "$—"
        score_part = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
        conf_part = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        return f"- {name} ({sym}) — {price_part} · score {score_part} · conf {conf_part}"

    # -------- Path A: normalize the original “Market Analysis” block --------
    if "Top Picks" in markdown_text or "Market Analysis:" in markdown_text:
        lines = markdown_text.splitlines()
        picks = []

        for line in lines:
            if line.lstrip().startswith("- **"):
                m1 = re.match(r"\s*-\s*\*\*(.+?)\*\*\s+\(([A-Z.\-]{1,8})\)", line)
                if not m1:
                    continue
                name, sym = m1.group(1), m1.group(2)
                m_price = re.search(r"\$\s*([0-9][0-9,]*\.?[0-9]*)", line)
                price = float(m_price.group(1).replace(",", "")) if m_price else None
                m_score = re.search(r"score\s+([0-9.]+)", line, re.I)
                score = float(m_score.group(1)) if m_score else None
                m_conf = re.search(r"Conf\s+([0-9.]+)", line, re.I)
                conf = float(m_conf.group(1)) if m_conf else None
                picks.append((name, sym, price, score, conf))

        out = [f"### Safe stock suggestions ({risk_label})", ""]
        for name, sym, price, score, conf in picks:
            try:
                t = yf.Ticker(sym)
                info = getattr(t, "info", {}) or {}
                hist = t.history()
                close = hist["Close"] if "Close" in hist else None
            except Exception:
                info, close = {}, None

            def _rsi(series: "pd.Series", n: int = 14):
                if series is None or len(series) < n + 5:
                    return None
                d = series.diff()
                up = d.clip(lower=0.0)
                down = -d.clip(upper=0.0)
                roll_up = up.rolling(n).mean()
                roll_down = down.rolling(n).mean()
                rs = roll_up / (roll_down + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                try:
                    return float(rsi.iloc[-1])
                except Exception:
                    return None

            rsi = _rsi(close)
            sma50 = float(close.rolling(50).mean().iloc[-1]) if (close is not None and len(close) >= 50) else None
            trend = "Bullish" if (close is not None and sma50 is not None and close.iloc[-1] > sma50) else "Bearish"
            vol20 = (
                float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5))
                if (close is not None and len(close) >= 40)
                else None
            )

            sector = info.get("sector")
            industry = info.get("industry")
            beta = info.get("beta")
            dy = info.get("dividendYield")
            pe = info.get("trailingPE")
            mcap = info.get("marketCap")
            debt = info.get("totalDebt")
            ebitda = info.get("ebitda")
            debt_ebitda = (debt / ebitda) if isinstance(debt, (int, float)) and isinstance(ebitda, (int, float)) and ebitda else None

            out.append(_mk_pick_line(name, sym, price, score, conf))

            bullets = []
            prof = []
            if sector:
                prof.append(sector + (f" — {industry}" if industry else ""))
            if isinstance(beta, (int, float)):
                prof.append(f"β≈{beta:.2f}")
            if isinstance(dy, (int, float)) and dy > 0:
                prof.append(f"DY≈{_fmt_dy(dy)}")
            if prof:
                bullets.append("Profile: " + " • ".join(prof))

            val = []
            if isinstance(pe, (int, float)):
                val.append(f"P/E≈{pe:.1f}x")
            if isinstance(mcap, (int, float)):
                val.append(f"Cap≈${mcap/1e9:.0f}B")
            if val:
                bullets.append("Valuation: " + " • ".join(val))

            mom = []
            if rsi is not None:
                mom.append(f"RSI {rsi:.0f}")
            if sma50 is not None and close is not None:
                mom.append(f"Trend {trend}")
            if vol20 is not None:
                mom.append(f"σ20d≈{vol20*100:.1f}%")
            if mom:
                bullets.append("Momentum & risk: " + " • ".join(mom))

            if debt_ebitda is not None:
                bullets.append(f"Balance sheet: Debt/EBITDA≈{debt_ebitda:.1f}×")

            bullets.append(
                "Why it screens as *defensive*: below-market β and/or dividend cushion, "
                "neutral-to-positive trend, and non-stretched multiples (demo data)."
            )
            out.extend([f"    - {b}" for b in bullets])
            out.append("")

        cleaned = [ln for ln in out if not ln.strip().startswith("Note:")]
        return "\n".join(cleaned)

    # -------- Path B (fallback) --------
    try:
        import re
        import pandas as pd
        import yfinance as yf
        pick_re = re.compile(
            r"""\s*(?:\d+\.|\-|\u2022)?\s*
                ([A-Za-z0-9 .&'/-]+)\s+\(
                ([A-Z.\-]{1,8})\)\s+—\s+\$
                ([0-9.,]+).*?score\s+([0-9.]+).*?conf\s+([0-9.]+)""",
            re.VERBOSE | re.IGNORECASE,
        )
        lines = markdown_text.splitlines()
        out, any_match = [], False
        for ln in lines:
            if ln.strip().startswith("Note:"):
                continue
            m = pick_re.match(ln)
            if not m:
                out.append(ln)
                continue
            any_match = True
            name, sym, price_s, score_s, conf_s = m.groups()
            try:
                price = float(price_s.replace(",", ""))
            except Exception:
                price = None
            score = float(score_s)
            conf = float(conf_s)
            try:
                t = yf.Ticker(sym)
                info = getattr(t, "info", {}) or {}
                hist = t.history()
                close = hist["Close"] if "Close" in hist else None
            except Exception:
                info, close = {}, None

            def _rsi(series: "pd.Series", n: int = 14):
                if series is None or len(series) < n + 5:
                    return None
                d = series.diff()
                up = d.clip(lower=0.0)
                down = -d.clip(upper=0.0)
                roll_up = up.rolling(n).mean()
                roll_down = down.rolling(n).mean()
                rs = roll_up / (roll_down + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                try:
                    return float(rsi.iloc[-1])
                except Exception:
                    return None

            rsi = _rsi(close)
            sma50 = float(close.rolling(50).mean().iloc[-1]) if (close is not None and len(close) >= 50) else None
            trend = "Bullish" if (close is not None and sma50 is not None and close.iloc[-1] > sma50) else "Bearish"
            vol20 = (
                float(close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5))
                if (close is not None and len(close) >= 40)
                else None
            )
            sector = info.get("sector"); industry = info.get("industry")
            beta = info.get("beta"); dy = info.get("dividendYield")
            pe = info.get("trailingPE"); mcap = info.get("marketCap")
            debt = info.get("totalDebt"); ebitda = info.get("ebitda")
            debt_ebitda = (debt / ebitda) if isinstance(debt, (int, float)) and isinstance(ebitda, (int, float)) and ebitda else None

            def _mk_pick_line(name, sym, price, score, conf):
                price_part = f"${price:,.2f}" if isinstance(price, (int, float)) else "$—"
                score_part = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
                conf_part = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
                return f"- {name} ({sym}) — {price_part} · score {score_part} · conf {conf_part}"

            out.append(_mk_pick_line(name, sym, price, score, conf))
            bullets = []
            prof = []
            if sector: prof.append(sector + (f" — {industry}" if industry else ""))
            if isinstance(beta, (int, float)): prof.append(f"β≈{beta:.2f}")
            if isinstance(dy, (int, float)) and dy > 0: prof.append(f"DY≈{_fmt_dy(dy)}")
            if prof: bullets.append("Profile: " + " • ".join(prof))
            val = []
            if isinstance(pe, (int, float)): val.append(f"P/E≈{pe:.1f}x")
            if isinstance(mcap, (int, float)): val.append(f"Cap≈${mcap/1e9:.0f}B")
            if val: bullets.append("Valuation: " + " • ".join(val))
            mom = []
            if rsi is not None: mom.append(f"RSI {rsi:.0f}")
            if sma50 is not None and close is not None: mom.append(f"Trend {trend}")
            if vol20 is not None: mom.append(f"σ20d≈{vol20*100:.1f}%")
            if mom: bullets.append("Momentum & risk: " + " • ".join(mom))
            if debt_ebitda is not None: bullets.append(f"Balance sheet: Debt/EBITDA≈{debt_ebitda:.1f}×")
            bullets.append("Why it screens as *defensive*: below-market β and/or dividend cushion, neutral-to-positive trend, and non-stretched multiples (demo data).")
            out.extend([f"    - {b}" for b in bullets]); out.append("")
        if any_match:
            header = f"### Safe stock suggestions ({risk_label})\n"
            return header + "\n".join(out)
    except Exception:
        pass

    cleaned = "\n".join([ln for ln in markdown_text.splitlines() if not ln.strip().startswith("Note:")])
    return cleaned


# ============================ Styles ===========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
/* Hide menu/deploy/footer ONLY — keep header & sidebar visible */
#MainMenu, .stDeployButton, footer { visibility: hidden; }

/* Sidebar: let Streamlit control collapse/expand. Tweak widths only. */
section[data-testid="stSidebar"] {
  width: 20rem;
  min-width: 16rem;
}

/* Make the toggle (hamburger) always clickable/visible */
header [data-testid="collapsedControl"]{
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

/* Mobile: sidebar uses full width; main gets comfortable padding */
@media (max-width: 768px){
  section[data-testid="stSidebar"]{
    width: 100vw !important;
    min-width: 0 !important;
  }
  .main .block-container{
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
  }
  .user-message { margin-left: 0; }
  .assistant-message { margin-right: 0; }
}

/* Readability inside assistant bubbles */
.assistant-message h3, .assistant-message h4 { margin: 1rem 0 .5rem; line-height: 1.25; }
.assistant-message p { margin: .4rem 0 .8rem; }
.assistant-message ul, .assistant-message ol { margin: .25rem 0 .9rem 1.1rem; }
.assistant-message li { margin: .25rem 0; }
.assistant-message a { word-break: break-all; }
.assistant-message code { white-space: pre-wrap; word-break: break-word; }

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


# ---------- Helper: synthesize snippets from processed attachments ----------
def _synthesize_attachment_snippet(sources, char_limit_per_file=1500, total_limit=5000):
    """
    Builds a short text snippet from uploaded images (OCR) and docs (text),
    so the LLM can use them even without a vision model.
    """
    if not sources:
        return ""
    parts, total = [], 0
    for s in sources:
        try:
            path = s.get("path")
            if not path:
                continue
            if s.get("type") == "image":
                txt = extract_text_from_image(path) or ""
            else:
                txt = extract_text_from_file(path) or ""
            if not txt.strip():
                continue
            snippet = txt.strip()[:char_limit_per_file]
            tag = f"[{os.path.basename(path)}]"
            piece = f"{tag}\n{snippet}"
            if total + len(piece) > total_limit:
                break
            parts.append(piece)
            total += len(piece)
        except Exception:
            continue
    if not parts:
        return ""
    return "\n\n".join(parts)


# ======================= Session State =========================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: "user"|"agent", content: str}]

if "query" not in st.session_state:
    st.session_state.query = ""

if "clear_query" not in st.session_state:
    st.session_state.clear_query = False

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_" + hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:12]

if "use_attachments_next" not in st.session_state:
    st.session_state.use_attachments_next = False

# Internal one-shot consumption flag for attachments checkbox
if "use_attachments_consumed" not in st.session_state:
    st.session_state.use_attachments_consumed = False

# NEW: persist last charts/symbols that should render on every rerun
if "last_charts" not in st.session_state:
    st.session_state.last_charts = []

if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False


# =========================== Sidebar ===========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # NOTE: removed the old single-file uploader to prevent duplicates

    # New attachment uploaders (with keys so Streamlit persists them cleanly)
    st.markdown("### 📎 Attachments")
    uploaded_images = st.file_uploader(
        "🖼️ Images (PNG/JPG/JPEG/WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Upload images for OCR text extraction",
        key="attachments_images",
    )
    uploaded_docs = st.file_uploader(
        "📄 Documents (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents for text analysis",
        key="attachments_docs",
    )

    # Quick visual confirmation of what the app has processed
    if st.session_state.get("attachments_sources"):
        with st.expander("Processed attachments", expanded=True):
            for s in st.session_state["attachments_sources"]:
                st.write(f"- {s['type']}: **{os.path.basename(s['path'])}**")

    # One-shot attachment usage toggle (default OFF)
    if st.session_state.get("attachments_sources"):
        st.checkbox(
            "📎 Use attachments for next question only",
            key="use_attachments_next",
            value=False,
            help="When ON, the next message will include any processed files. Then it turns OFF automatically."
        )

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

    # --- Academic mode & user requirements (init + merge) ---
    # Initialize in session_state once
    if "user_requirements" not in st.session_state:
        st.session_state["user_requirements"] = {}

    # Work on a copy for this run
    user_requirements = dict(st.session_state["user_requirements"])

    academic_mode = st.sidebar.toggle("Academic mode", value=False)

    if academic_mode:
        preset = ACADEMIC_PRESET or {
            "risk_tolerance": "Conservative",
            "investment_timeline": "Medium-term",
            "weights": {"valuation":0.30, "risk":0.35, "momentum":0.15, "size":0.15, "sentiment":0.05},
            "style": "academic",
            "require_primary_sources": True,
        }
        user_requirements.update(preset)

    # Persist any changes back into session state
    st.session_state["user_requirements"] = user_requirements

    # LLM Routing Controls
    st.markdown("### 🤖 LLM Settings")
    llm_routing = st.selectbox(
        "LLM Routing:",
        ["Auto", "Single", "Consensus (if available)"],
        index=0,
        help="Auto: Use best available provider. Single: Use primary provider. Consensus: Use multiple providers with judge."
    )
    
    routing_map = {"Auto": "auto", "Single": "single", "Consensus (if available)": "consensus"}
    st.session_state["llm_routing"] = routing_map[llm_routing]
    
    # High fidelity toggle
    high_fidelity = st.checkbox(
        "High Fidelity Mode",
        value=False,
        help="Enable consensus mode for investment decisions and portfolio management"
    )
    st.session_state["high_fidelity"] = high_fidelity
    
    # Provider status
    try:
        provider_status = healthcheck()
        groq_status = "ready" if provider_status.get("groq") == "ready" else "missing"
        ollama_status = "online" if provider_status.get("ollama") == "online" else "offline"
        st.markdown(f"**LLM Status:** Groq [{groq_status}] • Ollama [{ollama_status}]")
    except Exception:
        st.markdown("**LLM Status:** Unable to check provider status")

    # --- Language selection ---
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

    # 🌐 Language
    lang_choice = st.selectbox(
        "🌐 Language",
        list(lang_map.keys()),
        index=0,
        help="Choose the language for the assistant's reply. 'Auto (detect)' replies in your input language."
    )
    st.session_state["ui_language"] = lang_map.get(lang_choice)

    if st.button("🔄 Start New Conversation"):
        st.session_state.clear()
        st.session_state.last_charts = []
        st.rerun()


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

# ========================= Results (Charts) =========================
if st.session_state.get("last_charts"):
    st.markdown("### 📊 Charts")
    for symbol in st.session_state["last_charts"]:
        try:
            charts = create_all_charts(symbol)
            if charts:
                st.markdown(f"#### {symbol} Analysis")
                for chart_type, fig in charts.items():
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render charts for {symbol}: {e}")

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

            # ----- Decide if we use attachments for this message -----
            IMAGE_WORDS = ("image","photo","picture","screenshot","diagram","chart","scan","slide","attachment","file","document","pdf")
            mentions_attachment = any(w in query.lower() for w in IMAGE_WORDS)
            _attach_flag = bool(st.session_state.get("use_attachments_next", False))
            _attach_consumed = bool(st.session_state.get("use_attachments_consumed", False))
            use_attachments_now = (_attach_flag and not _attach_consumed) or mentions_attachment

            # ----- Attachment processing (only if opted-in) -----
            attachments_index = None
            attachments_sources = st.session_state.get("attachments_sources", [])

            if use_attachments_now:
                # Process uploads and create cloud-safe file list
                files = []
                
                # Handle uploaded files
                for uf in (uploaded_images or []):
                    suffix = Path(uf.name).suffix
                    tmp_path = Path(tempfile.gettempdir()) / f"st_{uuid.uuid4().hex}{suffix}"
                    with open(tmp_path, "wb") as f:
                        f.write(uf.getbuffer())
                    files.append((str(tmp_path), uf.type))
                
                for uf in (uploaded_docs or []):
                    suffix = Path(uf.name).suffix
                    tmp_path = Path(tempfile.gettempdir()) / f"st_{uuid.uuid4().hex}{suffix}"
                    with open(tmp_path, "wb") as f:
                        f.write(uf.getbuffer())
                    files.append((str(tmp_path), uf.type))

                # Fallback: if nothing uploaded, use the bundled sample (if it exists)
                if not files and SAMPLE_IMAGE.exists():
                    files = [(str(SAMPLE_IMAGE), "image/png")]

                # Process files
                if files:
                    try:
                        attachments_index = AttachmentIndex()
                        
                        for file_path, file_type in files:
                            if process_attachment_file(file_path, attachments_index):
                                attachments_sources.append({
                                    "type": "image" if file_type.startswith("image/") else "document", 
                                    "name": Path(file_path).name, 
                                    "path": file_path
                                })

                        if getattr(attachments_index, "texts", None):
                            attachments_index.build()
                            st.session_state.attachments_index = attachments_index
                            st.session_state.attachments_sources = attachments_sources
                            st.success(f"✅ Processed {len(attachments_sources)} attachments")

                    except Exception as e:
                        st.error(f"❌ Failed to process attachments: {e}")
                        attachments_index = None

                # Build snippet ONLY if use_attachments_now
                attachment_snippet = _synthesize_attachment_snippet(
                    st.session_state.get("attachments_sources", [])
                )
                
                # Provide clearer user feedback
                if not files:
                    st.info("No files received. Upload an image/PDF (left sidebar) or I'll use the sample chart.")
                elif not attachment_snippet.strip():
                    st.warning("I couldn't read text from your file. If this is an image with text, enable OCR fallback by adding `OCR_SPACE_API_KEY` in your Streamlit secrets, or upload a clearer image.")
            else:
                attachment_snippet = ""
            
            query_for_agent = (
                f"{query}\n\n[Attachment snippet]\n{attachment_snippet}"
                if attachment_snippet else query
            )

            # ----- Chat history pairs -----
            chat_history = []
            msgs = st.session_state.messages
            for i in range(0, len(msgs) - 1):
                if msgs[i]["role"] == "user":
                    agent_reply = msgs[i + 1]["content"] if i + 1 < len(msgs) and msgs[i + 1]["role"] != "user" else ""
                    chat_history.append({"user": msgs[i]["content"], "agent": agent_reply})

            # ----- Profile & weights -----
            user_profile = {}
            # start from sidebar preset (if any)
            user_requirements = dict(st.session_state.get("user_requirements", {}))
            
            if st.session_state.get("is_financial_mode") and is_financial_query(query):
                user_profile = {
                    "risk_tolerance": st.session_state.get("risk_tolerance", "Moderate"),
                    "investment_timeline": st.session_state.get("investment_timeline", "Medium-term"),
                }
                # compute weights from risk and MERGE into any preset weights instead of replacing
                derived = {}
                risk = user_profile["risk_tolerance"]
                if risk == "Aggressive":
                    derived = {"valuation":0.15,"risk":0.20,"momentum":0.35,"size":0.15,"sentiment":0.15}
                elif risk == "Conservative":
                    derived = {"valuation":0.30,"risk":0.40,"momentum":0.10,"size":0.15,"sentiment":0.05}
                else:
                    derived = {"valuation":0.25,"risk":0.30,"momentum":0.20,"size":0.15,"sentiment":0.10}
                base_w = dict(user_requirements.get("weights", {}))
                base_w.update(derived)
                user_requirements["weights"] = base_w

            # ----- Call agent -----
            use_attach = bool(st.session_state.get("use_attachments_next", False))
            attachments_param = st.session_state.get("attachments_index") if use_attach else None
            
            with st.spinner("🤖 Analyzing with multi-RAG and live data…"):
                result = run_customer_support(
                    query=query_for_agent,
                    user_profile=user_profile,
                    user_requirements=user_requirements,
                    force_language=st.session_state.get("ui_language"),
                    chat_history=chat_history,
                    file_path=None,  # not used now that we rely on attachments
                    user_id=st.session_state.user_id,
                    attachments_index=attachments_param,
                    attachments_sources=st.session_state.get("attachments_sources", []),
                    routing_override=st.session_state.get("llm_routing"),
                    high_fidelity=st.session_state.get("high_fidelity", False),
                )

            _enriched = _enrich_recommendations(result.get("response", ""))
            st.session_state.messages.append({"role": "agent", "content": _enriched})
            
            # NEW: persist chart symbols for rendering outside the click branch
            st.session_state.last_charts = result.get("charts", []) or []

            if result.get("explainability"):
                with st.expander("🔍 Analysis Details"):
                    explainability = result["explainability"]
                    if explainability.get("llm_providers"):
                        st.write("**LLM Providers:**", explainability["llm_providers"])
                    if explainability.get("routing_mode"):
                        st.write("**Routing Mode:**", explainability["routing_mode"])
                    if explainability.get("consensus_judge"):
                        st.write("**Consensus Judge:**", explainability["consensus_judge"])
                    if explainability.get("attachment_sources"):
                        st.write(f"**Attachment Sources:** {explainability['attachment_sources']} documents")
                    if explainability.get("filings"):
                        st.write("**SEC Filings:**")
                        for filing in explainability["filings"][:3]:
                            st.write(f"- {filing.get('form', 'N/A')} ({filing.get('filing_date', 'N/A')})")

            # mark the one-shot checkbox as consumed without modifying the widget value directly
            if bool(st.session_state.get("use_attachments_next", False)):
                st.session_state["use_attachments_consumed"] = True
            st.session_state.clear_query = True
            st.rerun()

# ===================== Export last answer ======================
if st.session_state.messages and st.session_state.messages[-1]["role"] == "agent":
    md_text = st.session_state.messages[-1]["content"]
    st.download_button(
        "⬇️ Download last answer (Markdown)",
        data=md_text.encode("utf-8"),
        file_name=f"recommendations_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md",
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
