"""Chunk data/docs.jsonl and persist the BM25 index, dense vectors and chunk metadata to index/."""

import json

import bm25s
import numpy as np
from privrag import config
from privrag.chunk import load_chunks
from sentence_transformers import SentenceTransformer


def main():
    chunks = load_chunks(config.DOCS)
    config.INDEX_DIR.mkdir(exist_ok=True)
    texts = [f"{c['title']} | {c['section']} | {c['text']}" for c in chunks]
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
    bm25.save(str(config.INDEX_DIR / "bm25"))
    vecs = SentenceTransformer(config.EMBED_MODEL).encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True
    )
    np.save(config.INDEX_DIR / "vectors.npy", vecs.astype(np.float32))
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"{len(chunks)} chunks indexed")


if __name__ == "__main__":
    main()
