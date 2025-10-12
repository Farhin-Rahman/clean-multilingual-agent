# AI Financial Assistant (Free-first, Agentic, OSS)

**Agentic pipeline for stock discovery & recommendations**  
**SQL → Live market data → Multi-RAG (qual / logic / numeric) → Verification → Weighted MCDM + neuro-symbolic rules** — all on a **free / local** stack (Ollama, MiniLM, FAISS/Win-skip, Streamlit).

<p align="left">
  <!-- Replace owner/repo with your GitHub handle and repo name -->
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/<owner>/<repo>/ci.yml?label=CI&logo=github" />
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-informational" />
</p>

> **What it shows:** A production-style *agentic* system that **finds information** (A1), **integrates sources** (A2), **turns text to SQL** (A3), and **recommends options under user-weighted criteria** (A4). It blends ML with **neuro-symbolic** rules and hard **guardrails** to reduce hallucinations — a strong fit for Kodamai's JD & assignment.

---

## ✨ Highlights

- **A1: Locate info** – web/DDG for news & filings; qualitative & logical RAG; numeric facts via yfinance.  
- **A2: Integrate** – merge SQLite companies/portfolio + live metrics + RAG hits with **provenance**.  
- **A3: Text→SQL** – LLM-generated queries with **AST validation** (SELECT-only, table allowlist).  
- **A4: Recommend** – weighted **MCDM** + **rule engine** (gating/boosting) + **rank stability**.  
- **Neuro-symbolic** – human-readable rules with flags; numeric sanity checks (PE/EPS).  
- **Free-first** – local LLM (Ollama), MiniLM embeddings, FAISS CPU (auto-skips on Windows), Streamlit UI.  
- **Reliability** – request caching, timeouts, graceful fallbacks; simple rate limiting.  
- **Trust** – explicit sources (news/qual/logic/numeric), confidence, guardrails.

---

## 🧭 Architecture (bird's-eye)

User ↔ Streamlit UI
│
▼
Intent Router ──► Portfolio ops (SQLite)
│
├─► Text→SQL agent ──► AST validate (SELECT + table allowlist) ──► Query DB
│
└─► Data Integration
• Live market (yfinance) + indicators (RSI/SMA/vol)
• Qualitative RAG (MiniLM + FAISS/trafilatura)
• Logical RAG (deal/filing patterns)
• Numeric RAG (financials, cashflow, debt, EPS)
• Provenance stitched across tracks
│
▼
Verification + Rules
• Numeric sanity (PE/EPS)
• Neuro-symbolic flags & gating
• Confidence from evidence mix
│
▼
MCDM Ranker
• Weighted features (valuation/risk/momentum/size/sentiment)
• Rank stability (±10% weights)
│
▼
Final Recommendations (+ citations)
---

## 🚀 Quick start

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
# Optional (local LLM):
# ollama pull llama3.1:8b-instruct
streamlit run app.py
```

### Linux/macOS
```bash
python -m venv venv && source venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
# Optional (local LLM):
# ollama pull llama3.1:8b-instruct
streamlit run app.py
```

## 🚀 Deploy Notes

### Streamlit Cloud
- Streamlit Cloud ignores `.env` files; set secrets in the dashboard:
  - `GROQ_API_KEY` = your Groq API key
  - `SEC_USER_AGENT` = "Your Name your@email"
- OCR will gracefully skip if Tesseract is missing
- DuckDuckGo search will fallback gracefully if unavailable