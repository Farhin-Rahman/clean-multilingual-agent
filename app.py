# app.py — Streamlit UI (revamped) for the Financial Support Agent (free-first)
import os, base64, tempfile, hashlib, uuid, re, json, time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from support_agent import run_customer_support
try:
    from support_agent import portfolio_manager
except Exception:
    portfolio_manager = None

load_dotenv()
st.set_page_config(
    page_title="AI Financial Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖",
)

# ------------- Styles -------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --ink:#0f172a; --muted:#64748b; --card:#ffffff; --chip:#f3f4f6; --accent:#4F46E5; }
html, body, [class^="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, .stDeployButton, footer, header { visibility: hidden; }

.app-header { text-align:center; padding: 1rem 0 1.25rem 0; }
.app-header h1 { font-size: 1.9rem; font-weight: 700; color: var(--ink); margin:0; }
.app-header p { color: var(--muted); margin:.3rem 0 0 0; }
.badges { display:flex; gap:.5rem; justify-content:center; margin-top:.5rem; flex-wrap:wrap; }
.badge { font-size:.8rem; padding:.25rem .6rem; border-radius:999px; background:var(--chip); color:#334155; border:1px solid #e5e7eb; }

.kpi { display:flex; gap:1rem; margin:.5rem 0 1rem 0; flex-wrap:wrap; }
.kpi .item { background: #fff; border:1px solid #e5e7eb; border-radius: .75rem; padding:.75rem 1rem; min-width: 160px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
.kpi .label { color:#64748b; font-size:.8rem; }
.kpi .value { color:#0f172a; font-size:1.1rem; font-weight:600; }

.msg { padding: 1rem 1.25rem; border-radius: 1rem; margin-bottom: .75rem; line-height: 1.6; font-size: .96rem; box-shadow: 0 2px 8px rgba(0,0,0,.05); border:1px solid #e5e7eb; background:#fff; }
.user { background:#F3F4F6; }
.assistant { background:#fff; }

hr.soft { border:0; height:1px; background:linear-gradient(90deg, transparent, #e5e7eb, transparent); margin: 1.25rem 0; }
.small { color:#64748b; font-size:.85rem; }
.prov a { font-size:.9rem; }
.warn { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; padding:.5rem .75rem; border-radius:.5rem; }

.stButton > button { background: var(--accent) !important; color:#fff !important; border:none !important; border-radius:.5rem !important; padding:.6rem 1rem !important; }
.stButton > button:hover { filter: brightness(0.92); }
</style>
""", unsafe_allow_html=True)

# ------------- Session -------------
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
if "messages" not in st.session_state: st.session_state.messages = []
if "query" not in st.session_state: st.session_state.query = ""
if "clear_query" not in st.session_state: st.session_state.clear_query = False
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_" + hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:12]
if st.session_state.clear_query:
    st.session_state.query = ""; st.session_state.clear_query = False

# ------------- Utils -------------
def _extract_links(md: str):
    out=[]
    for m in re.finditer(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', md or ""):
        out.append((m.group(1), m.group(2)))
    return out[:20]

@st.cache_data(ttl=30)
def probe_ollama():
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    tags = url.replace("/api/chat", "/api/tags")
    try:
        r = requests.get(tags, timeout=2.5)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models",[])]
        ok = any(models)
        return {"ok": ok, "models": models}
    except Exception:
        return {"ok": False, "models": []}

def render_health_and_limits(remaining:int|None):
    status = probe_ollama()
    cols = st.columns([1,1,1,5])
    with cols[0]:
        st.markdown(f"""<div class="kpi"><div class="item">
            <div class="label">Ollama</div>
            <div class="value">{'Connected' if status['ok'] else 'Unavailable'}</div>
        </div></div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div class="kpi"><div class="item">
            <div class="label">Models</div>
            <div class="value">{len(status['models'])}</div>
        </div></div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""<div class="kpi"><div class="item">
            <div class="label">Requests left</div>
            <div class="value">{remaining if remaining is not None else '—'}</div>
        </div></div>""", unsafe_allow_html=True)

def safe_run_customer_support(query, language, chat_history, file_path, user_id, answer_style, strict_mode, fast_mode):
    try:
        return run_customer_support(
            query=query,
            force_language=language,
            chat_history=chat_history,
            file_path=file_path,
            user_id=user_id,
            answer_style=answer_style,
            strict_mode=strict_mode,
            fast_mode=fast_mode,
        )
    except TypeError:
        flags = f" [[STYLE:{answer_style}]] [[STRICT:{1 if strict_mode else 0}]] [[FAST:{1 if fast_mode else 0}]]"
        return run_customer_support(
            query=(query + flags),
            force_language=language,
            chat_history=chat_history,
            file_path=file_path,
            user_id=user_id,
        )

# ------------- Header -------------
st.markdown("""
<div class="app-header">
  <h1>AI Financial Assistant</h1>
  <p>Agentic analysis • Multi-RAG (quant/qual/logic) • SQL-safe • Free-first</p>
  <div class="badges">
    <div class="badge">Provenance-gated</div>
    <div class="badge">Neuro-symbolic rules</div>
    <div class="badge">Local LLM (Ollama)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------- Sidebar -------------
with st.sidebar:
    st.subheader("⚙️ Settings")
    display_language = st.selectbox(
        "Response language",
        ["Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"],
        index=0,
    )
    lang_map = {
        "Auto (detect)": None, "English": "en", "Spanish": "es", "French": "fr",
        "German": "de", "Bengali": "bn", "Italian": "it", "Portuguese": "pt",
    }

    answer_style = st.radio("Answer style", ["simple", "balanced", "expert"], index=0, horizontal=True)
    strict_mode = st.checkbox("Strict provenance gate (safer)", value=True)
    fast_mode = st.checkbox("Fast mode (lighter RAG)", value=False)

    uploaded_file = st.file_uploader("📎 Upload file (PDF or TXT)", type=["pdf", "txt"])

    if st.button("🔄 New chat"):
        st.session_state.clear()
        st.rerun()

# ------------- Tabs -------------
tab_chat, tab_portfolio, tab_about = st.tabs(["💬 Chat", "📊 Portfolio", "ℹ️ About"])

with tab_chat:
    render_health_and_limits(remaining=None)

    for msg in st.session_state.messages:
        klass = "user" if msg["role"] == "user" else "assistant"
        who = "You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f'<div class="msg {klass}"><strong>{who}:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Ask about investments, companies, screening…")
    query = st.text_area("Your message", value=st.session_state.query, height=120, label_visibility="collapsed",
                         placeholder="e.g., 'Safe tech stocks under $50' or 'Show my portfolio performance'")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 Send", use_container_width=True):
            if query.strip():
                st.session_state.messages.append({"role": "user", "content": query})

                file_path = None; appended_text = ""
                if uploaded_file:
                    if uploaded_file.type == "application/pdf":
                        safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in ("-", "_", "."))
                        save_path = UPLOAD_DIR / safe_name
                        with save_path.open("wb") as f: f.write(uploaded_file.read())
                        file_path = str(save_path.resolve())
                    else:
                        txt = uploaded_file.read().decode(errors="ignore")
                        appended_text = f"\n\n[Attached note]\n{txt[:2000]}"

                chat_history = []
                last_user = None
                for m in st.session_state.messages[:-1]:
                    if m["role"] == "user": last_user = m["content"]
                    elif last_user is not None:
                        chat_history.append({"user": last_user, "agent": m["content"]}); last_user = None

                with st.spinner("🤖 Thinking (agentic pipeline)…"):
                    result = safe_run_customer_support(
                        query=(query + appended_text),
                        language=lang_map[display_language],
                        chat_history=chat_history,
                        file_path=file_path,
                        user_id=st.session_state.user_id,
                        answer_style=answer_style,
                        strict_mode=strict_mode,
                        fast_mode=fast_mode,
                    )

                resp = result.get("response", "Sorry, no response.")
                st.session_state.messages.append({"role": "agent", "content": resp})
                st.session_state.clear_query = True

                links = _extract_links(resp)
                if links:
                    st.markdown("##### Sources detected")
                    for t, u in links[:12]:
                        st.markdown(f"- [{t}]({u})")
                else:
                    st.caption("No explicit links detected in the last answer.")

                render_health_and_limits(result.get("remaining_requests"))

                st.download_button(
                    "⬇️ Download last answer (Markdown)",
                    data=resp.encode("utf-8"),
                    file_name=f"recommendations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                st.rerun()

with tab_portfolio:
    st.subheader("Your Portfolio")
    if portfolio_manager is None:
        st.info("Portfolio manager not available in this environment.")
    else:
        perf = portfolio_manager.get_portfolio_performance(st.session_state.user_id)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Value", f"${perf.get('total_value',0):,.2f}")
        k2.metric("Cost Basis", f"${perf.get('total_cost',0):,.2f}")
        k3.metric("Total P&L", f"${perf.get('total_pnl',0):,.2f}")
        k4.metric("Return %", f"{perf.get('total_return_pct',0):+.2f}%")
        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        rows = perf.get("positions", [])
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)[["symbol","shares","current_price","purchase_price","current_value","pnl","pnl_pct"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No holdings yet. Try: **buy 10 shares of AAPL** in Chat.")

with tab_about:
    st.markdown("""
**What this app does**

- Agentic pipeline: SQL → live market data → multi-RAG (qual/logic/numeric) → verification → weighted recommendations + neuro-symbolic rules.
- Free-first stack: local LLM (Ollama), yfinance, DDG, FAISS, MiniLM, Streamlit.

**Safety & quality**

- SQL AST validation + allowlist, rate limiting, input sanitization.
- Optional provenance-gated recommendations (numeric + independent sources).
- Numeric sanity (P/E vs EPS check), rank-stability sim.

**Tips**

- Use **Answer style** = *simple* for TL;DR + checklist; *expert* shows fuller traces.
- Turn on **Strict provenance** to block weak evidence.
- Use **Fast mode** for quicker (lighter) runs.
""")
