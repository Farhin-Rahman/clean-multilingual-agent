# warmup_finbert.py — one-time FinBERT download + cache
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

print("[warmup] Downloading ProsusAI/finbert …")
from transformers import pipeline

clf = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert", top_k=None)
print("[warmup] Building pipeline & running a tiny test …")
out = clf("Revenue outlook improved and margins expanded.")
print(out)
print("[warmup] Done. FinBERT cached in your HF hub directory.")
