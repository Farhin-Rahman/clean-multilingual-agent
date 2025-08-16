# AI Financial Assistant (Free-first)

Agentic pipeline for stock screening & recommendations:
SQL → live market data → Multi-RAG (qual/logic/numeric) → verification → weighted MCDM + neuro-symbolic rules — **all on a free/local stack**.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (Optional) pull a local model
# ollama pull llama3.1:8b-instruct
streamlit run app.py
