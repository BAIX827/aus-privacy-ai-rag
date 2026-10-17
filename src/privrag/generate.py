"""LLM backends. Each takes (question, chunks) and returns answer text citing [n] source numbers."""

import os
import re
from functools import lru_cache

import httpx

from . import config

NOT_FOUND = "Not found in the indexed OAIC guidance."
SYSTEM = (
    "Answer the question using only the numbered sources. Cite every claim with its source number, "
    "e.g. [1]. Be concise (max 3 sentences)."
)
REFUSE = f" If the sources do not answer it, reply exactly: {NOT_FOUND}"
# Small local models over-refuse with REFUSE, so for them the reranker score gate alone handles "not found".


def build_prompt(question, chunks):
    ctx = "\n\n".join(f"[{i}] ({c['title']} — {c['section']})\n{c['text']}" for i, c in enumerate(chunks, 1))
    return f"Sources:\n{ctx}\n\nQuestion: {question}"


def extractive(question, chunks):
    """No-LLM fallback: the sentence from the top chunks with the most question-word overlap."""
    terms = set(re.findall(r"\w{4,}", question.lower()))
    best, best_score = None, 0
    for i, c in enumerate(chunks[:2], 1):
        for sent in re.split(r"(?<=[.;])\s+", c["text"]):
            score = len(terms & set(re.findall(r"\w{4,}", sent.lower())))
            if score > best_score:
                best, best_score = f"{sent} [{i}]", score
    return best or NOT_FOUND


def openai_chat(question, chunks):
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    resp = httpx.post(
        f"{base}/chat/completions",
        timeout=60,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM + REFUSE},
                {"role": "user", "content": build_prompt(question, chunks)},
            ],
        },
    )
    return resp.raise_for_status().json()["choices"][0]["message"]["content"].strip()


@lru_cache(maxsize=1)
def _local_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.LOCAL_LLM)
    return tok, AutoModelForCausalLM.from_pretrained(config.LOCAL_LLM)


def local_chat(question, chunks):
    tok, model = _local_model()
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**ids, max_new_tokens=160, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


BACKENDS = {"extractive": extractive, "openai": openai_chat, "local": local_chat}


def generate(question, chunks, backend=None):
    return BACKENDS[backend or config.LLM_BACKEND](question, chunks)
