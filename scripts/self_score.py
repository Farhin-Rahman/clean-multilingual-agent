# scripts/self_score.py — deterministic self-scoring harness
import json, os, re

CATEGORIES = {
    "Architecture": 0.20,
    "Security": 0.15,
    "Reliability": 0.15,
    "Performance": 0.10,
    "UX": 0.10,
    "FreeFirst": 0.15,
    "DocsPackagingTests": 0.15,
}

def exists(path): return os.path.exists(path)

def check_architecture():
    pts=0; maxpts=10
    s = open("support_agent.py", "r", encoding="utf-8").read()
    pts += 3 if exists("support_agent.py") else 0
    pts += 2 if exists("rag_text.py") and exists("rag_logic.py") and exists("rag_numeric.py") else 0
    pts += 2 if "StateGraph" in s else 0
    pts += 1 if "ThreadPoolExecutor" in s else 0
    pts += 2 if "mcdm_score" in s and "apply_rules" in s else 0
    return pts/maxpts*10

def check_security():
    s = open("support_agent.py","r",encoding="utf-8").read()
    pts=0; maxpts=10
    pts += 3 if "validate_sql_ast" in s and "sqlglot" in s else 0
    pts += 2 if "APPROVED_TABLES" in s else 0
    pts += 1 if "RateLimiter" in s else 0
    pts += 2 if "hash_sensitive_data" in s else 0
    # fixed alternation (was matching literally "with|union|attach")
    pts += 2 if re.search(r"\b(with|union|attach)\b", s, re.I) else 0
    return min(10, pts)/maxpts*10

def check_reliability():
    s = open("support_agent.py","r",encoding="utf-8").read()
    pts=0; maxpts=10
    pts += 2 if "numeric_verification" in s else 0
    pts += 2 if "_meets_provenance_gate" in s else 0
    pts += 2 if "requests_cache" in s else 0
    pts += 2 if "retry(" in s else 0
    pts += 2 if "validate_query" in s else 0
    return pts/maxpts*10

def check_performance():
    s = open("support_agent.py","r",encoding="utf-8").read()
    pts=0; maxpts=10
    pts += 4 if "ThreadPoolExecutor" in s else 0
    pts += 2 if "lru_cache" in s else 0
    pts += 2 if "yf.download" in s else 0
    pts += 2 if "AGENT_BUDGET_MS" in s else 0
    return min(10, pts)/maxpts*10

def check_ux():
    pts=0; maxpts=10
    pts += 5 if exists("app.py") else 0
    pts += 3 if "st.file_uploader" in open("app.py","r",encoding="utf-8").read() else 0
    pts += 2 if "download_button" in open("app.py","r",encoding="utf-8").read() else 0
    return pts/maxpts*10

def check_freefirst():
    req = open("requirements.txt","r",encoding="utf-8").read()
    pts=0; maxpts=10
    pts += 5 if "ollama" not in req else 5
    pts += 2 if "yfinance" in req else 0
    pts += 1 if "faiss-cpu" in req else 0
    pts += 1 if "duckduckgo-search" in req else 0
    pts += 1 if "streamlit" in req else 0
    return min(10, pts)/maxpts*10

def check_docs_tests():
    pts=0; maxpts=10
    pts += 2 if exists("README.md") else 0
    pts += 2 if exists("LICENSE") else 0
    pts += 2 if exists("tests") else 0
    pts += 2 if exists(".github/workflows/ci.yml") else 0
    pts += 2 if exists("requirements.txt") else 0
    return pts/maxpts*10

def main():
    scores = {
        "Architecture": round(check_architecture(),2),
        "Security": round(check_security(),2),
        "Reliability": round(check_reliability(),2),
        "Performance": round(check_performance(),2),
        "UX": round(check_ux(),2),
        "FreeFirst": round(check_freefirst(),2),
        "DocsPackagingTests": round(check_docs_tests(),2),
    }
    overall = sum(scores[k]*w for k,w in CATEGORIES.items())
    print(json.dumps({"scores":scores,"overall":round(overall,2)}, indent=2))

if __name__ == "__main__":
    main()
