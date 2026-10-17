"""Tiny on-disk index with fake models, so tests run without downloading anything."""

import json

import bm25s
import numpy as np
import pytest
from privrag.retrieve import Retriever

CHUNKS = [
    {
        "id": "app-6#0",
        "doc_id": "app-6",
        "title": "Chapter 6: APP 6",
        "section": "Secondary purpose",
        "url": "https://example.org/app-6",
        "text": "An entity may use personal information for a secondary purpose if the individual would reasonably expect it.",
    },
    {
        "id": "app-11#1",
        "doc_id": "app-11",
        "title": "Chapter 11: APP 11",
        "section": "Security",
        "url": "https://example.org/app-11",
        "text": "An entity must take reasonable steps to protect personal information from misuse and loss.",
    },
    {
        "id": "ai#2",
        "doc_id": "ai-commercial-products",
        "title": "AI products",
        "section": "Inputs",
        "url": "https://example.org/ai",
        "text": "Organisations should not enter personal information into public generative AI tools.",
    },
]
VOCAB = ["secondary", "protect", "generative", "purpose", "security", "ai"]


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, **_):
        v = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeReranker:
    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) - 1.0 for q, t in pairs]


@pytest.fixture
def retriever(tmp_path):
    texts = [c["text"] for c in CHUNKS]
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    bm25.save(str(tmp_path / "bm25"))
    np.save(tmp_path / "vectors.npy", FakeEmbedder().encode(texts))
    (tmp_path / "chunks.json").write_text(json.dumps(CHUNKS))
    return Retriever(tmp_path, embedder=FakeEmbedder(), reranker=FakeReranker())
