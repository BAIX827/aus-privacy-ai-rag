import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "docs.jsonl"
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LOCAL_LLM = os.getenv("LOCAL_LLM", "Qwen/Qwen2.5-0.5B-Instruct")
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")  # local | openai | extractive

CHUNK_WORDS, CHUNK_OVERLAP = 180, 40
CANDIDATES = 30  # per retriever before fusion
RRF_K = 60
TOP_K = 4  # chunks passed to the LLM
MIN_RERANK = float(os.getenv("MIN_RERANK", "0.0"))  # cross-encoder logit; 0 = p(relevant) 0.5
