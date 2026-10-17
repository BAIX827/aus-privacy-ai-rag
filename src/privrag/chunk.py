"""Split section records into overlapping word windows that keep their source metadata."""

import json
from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_WORDS


def chunk_section(text: str, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def load_chunks(path: Path):
    chunks = []
    for line in path.open():
        doc = json.loads(line)
        for part in chunk_section(doc["text"]):
            chunks.append(
                {
                    **{k: doc[k] for k in ("doc_id", "title", "section", "url")},
                    "id": f"{doc['doc_id']}#{len(chunks)}",
                    "text": part,
                }
            )
    return chunks
