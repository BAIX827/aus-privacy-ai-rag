"""Hybrid retrieval: BM25 + dense (FAISS) fused with reciprocal rank fusion, then cross-encoder rerank."""

import json
import re
from functools import cached_property

import bm25s
import numpy as np

from . import config

APP_REF = re.compile(r"\bAPP\s*(\d{1,2})\b", re.I)


def rrf(rankings, k=config.RRF_K):
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def app_filter(question: str, chunks):
    """Indices of chunks from the APP chapters named in the question (None = no filter)."""
    apps = {f"app-{n}" for n in APP_REF.findall(question) if 1 <= int(n) <= 13}
    if not apps:
        return None
    keep = {i for i, c in enumerate(chunks) if c["doc_id"] in apps}
    return keep or None


class Retriever:
    def __init__(self, index_dir=config.INDEX_DIR, embedder=None, reranker=None):
        self.chunks = json.loads((index_dir / "chunks.json").read_text())
        self.bm25 = bm25s.BM25.load(str(index_dir / "bm25"))
        self.vectors = np.load(index_dir / "vectors.npy")
        self._embedder, self._reranker = embedder, reranker

    @cached_property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    @cached_property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def bm25_rank(self, q, n):
        ids, _ = self.bm25.retrieve(
            bm25s.tokenize([q], stopwords="en", show_progress=False),
            k=min(n, len(self.chunks)),
            show_progress=False,
        )
        return list(ids[0])

    def dense_rank(self, q, n):
        qv = self.embedder.encode([q], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.vectors @ qv))[:n])

    def search(self, q, mode="rerank", k=config.TOP_K, n=config.CANDIDATES):
        """Return [(chunk, score)]; mode in bm25 | dense | hybrid | rerank."""
        keep = app_filter(q, self.chunks)
        pool = n if keep is None else len(self.chunks)  # widen, then filter
        ranks = {"bm25": [self.bm25_rank(q, pool)], "dense": [self.dense_rank(q, pool)]}
        ranks["hybrid"] = ranks["rerank"] = ranks["bm25"] + ranks["dense"]
        order = rrf(ranks[mode])
        if keep is not None:
            order = [i for i in order if i in keep]
        order = order[:n]
        if mode != "rerank":
            return [(self.chunks[i], 1.0 / (r + 1)) for r, i in enumerate(order[:k])]
        scores = self.reranker.predict([(q, self.chunks[i]["text"]) for i in order])
        best = np.argsort(-np.asarray(scores))[:k]
        return [(self.chunks[order[j]], float(scores[j])) for j in best]
