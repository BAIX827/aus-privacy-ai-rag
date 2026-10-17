"""Question -> retrieved chunks -> grounded answer with resolved citations."""

import re

from . import config
from .generate import NOT_FOUND, generate


def answer(question, retriever, backend=None, k=config.TOP_K):
    hits = retriever.search(question, mode="rerank", k=k)
    if not hits or hits[0][1] < config.MIN_RERANK:
        return {"answer": NOT_FOUND, "sources": [], "citation_inferred": False}
    chunks = [c for c, _ in hits]
    text = generate(question, chunks, backend)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
    if NOT_FOUND.lower().rstrip(".") in text.lower():
        return {"answer": NOT_FOUND, "sources": [], "citation_inferred": False}
    inferred = not cited
    cited = cited or [1]  # model gave no marker: attribute to the top-ranked chunk
    sources = [
        {
            "n": n,
            "doc_id": chunks[n - 1]["doc_id"],
            "title": chunks[n - 1]["title"],
            "section": chunks[n - 1]["section"],
            "url": chunks[n - 1]["url"],
            "score": round(hits[n - 1][1], 3),
        }
        for n in cited
    ]
    return {"answer": text, "sources": sources, "citation_inferred": inferred}
