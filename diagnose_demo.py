# diagnose_demo.py — probes live vs. rate-limited, writes suggest_env.ps1
import os, sys, json

def check_yfinance():
    try:
        import yfinance as yf, pandas as pd
        is_shim = getattr(yf.Ticker, "__name__", "") == "_DemoTicker"
        try:
            t = yf.Ticker("AAPL")
            df = t.history(period="5d", interval="1d")
            ok = isinstance(df, pd.DataFrame) and not df.empty
            if ok and not is_shim:
                return {"status":"LIVE_OK", "shim":False}
            return {"status":"SHIM", "shim":True}
        except Exception as e:
            s = repr(e)
            if "429" in s or "Too Many Requests" in s:
                return {"status":"RATE_LIMIT", "shim":is_shim, "error":s}
            return {"status":"ERROR", "shim":is_shim, "error":s}
    except Exception as e:
        return {"status":"IMPORT_ERROR", "shim":None, "error":repr(e)}

def check_ddg():
    try:
        from duckduckgo_search import DDGS
        try:
            res = list(DDGS().text("x", max_results=1))
            stub = bool(res and "example.com" in str(res[0].get("href","")))
            return {"status": "SHIM" if stub else "LIVE_OK", "shim":stub}
        except Exception as e:
            s = repr(e)
            if "202" in s or "429" in s or "Too Many Requests" in s:
                return {"status":"RATE_LIMIT", "shim":False, "error":s}
            return {"status":"ERROR", "shim":False, "error":s}
    except Exception as e:
        return {"status":"IMPORT_ERROR", "shim":None, "error":repr(e)}

def check_trafilatura():
    try:
        import trafilatura as tr
        try:
            out = tr.fetch_url("https://example.com/")
            # our shim returns None; live usually returns bytes/text
            return {"status":"SHIM_OR_BLOCKED" if out is None else "LIVE_OK"}
        except Exception as e:
            return {"status":"ERROR", "error":repr(e)}
    except Exception as e:
        return {"status":"IMPORT_ERROR", "error":repr(e)}

def check_finbert():
    # works if you've warmed it; otherwise this will fail offline
    try:
        from transformers import pipeline
        nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", device=-1)
        _ = nlp("profits improved this quarter")
        return {"available":True}
    except Exception as e:
        return {"available":False, "error":repr(e)}

def check_ollama():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=0.25)
        return {"available": r.ok}
    except Exception:
        return {"available": False}

def main():
    yfr = check_yfinance()
    ddg = check_ddg()
    tra = check_trafilatura()
    fin = check_finbert()
    oll = check_ollama()

    print("yfinance:", yfr)
    print("DDG     :", ddg)
    print("traf    :", tra)
    print("FinBERT :", fin)
    print("Ollama  :", oll)

    suggest = {
        "FREE_DEMO": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "YF_MAX_SYMBOLS": os.environ.get("YF_MAX_SYMBOLS","3"),
        "PYTHONWARNINGS": "ignore",
    }
    # yfinance: live if possible, else shim
    suggest["SHIM_YF"] = "0" if yfr.get("status") == "LIVE_OK" else "1"
    # ddg: shim if not live-ok
    suggest["SHIM_DDG"] = "1" if ddg.get("status") != "LIVE_OK" else "0"
    # trafilatura: shim unless LIVE_OK
    suggest["SHIM_TRAFI"] = "1" if tra.get("status") != "LIVE_OK" else "0"
    # FinBERT: use finbert if warmed/available
    suggest["SENTIMENT_ENGINE"] = "finbert" if fin.get("available") else "auto"
    # LLM: enable only if Ollama is up
    suggest["AGENT_DISABLE_LLM"] = "0" if oll.get("available") else "1"

    # write PowerShell env file
    with open("suggest_env.ps1","w", encoding="utf-8") as f:
        for k,v in suggest.items():
            f.write(f'$env:{k} = "{v}"\n')
    print("\nWrote suggest_env.ps1 (recommended env). Apply with:")
    print("  . .\\suggest_env.ps1   # note the leading dot (dot-source into current shell)")

if __name__ == "__main__":
    main()
