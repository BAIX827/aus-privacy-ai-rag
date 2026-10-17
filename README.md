# Australian Privacy & AI Guidance — RAG Q&A

Question answering over the OAIC's privacy guidance: the Australian Privacy Principles (APP) guidelines plus the two guides on AI. Answers come only from retrieved passages and cite the section and URL they rely on. If the guidance doesn't cover a question, the service says so instead of guessing.

## Architecture

```
scripts/scrape.py   polite crawler (robots.txt, 3 s delay) → data/docs.jsonl (19 pages, 403 sections)
build_index.py      section chunks (180 words, 40 overlap) → BM25 (bm25s) + bge-small-en-v1.5 vectors
retrieve.py         BM25 ∥ dense → reciprocal rank fusion → ms-marco MiniLM cross-encoder rerank
                    "APP n" in the question → restrict to that chapter
rag.py              rerank score < 0 → "not found"; else top-4 chunks → LLM → [n] citations → sources
generate.py         backends: local (Qwen2.5-0.5B-Instruct) | openai (any OpenAI-compatible API) | extractive
app/                FastAPI (/ask, /health) + single-page UI
```

## Results

The eval set has 50 hand-written questions, each with a gold section and an answer phrase. `scripts/make_eval.py` checks every phrase against the corpus text. There are also 5 out-of-scope questions. A retrieval hit means the gold section was retrieved.

| Retriever | Recall@1 | Recall@4 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.42 | 0.78 | 0.92 | 0.574 |
| Dense (bge-small) | 0.58 | 0.80 | 0.96 | 0.689 |
| Hybrid (RRF) | 0.54 | 0.84 | **0.98** | 0.674 |
| Hybrid + rerank | **0.66** | **0.90** | 0.94 | **0.747** |

| Generator | Answer contains gold phrase | Gold section cited | Wrongly "not found" | Out-of-scope refused | s/question (CPU) |
|---|---|---|---|---|---|
| Extractive (no LLM) | 0.66 | 0.60 | 0.00 | 5/5 | 1.0 |
| Qwen2.5-0.5B-Instruct | 0.40 | 0.66 | 0.00 | 5/5 | 8.4 |

The gold-phrase match is strict, so paraphrased answers count as misses. The 0.5B model is a free CPU default and is not tuned for quality; set `LLM_BACKEND=openai` for a stronger model. Raw numbers are in `eval/results.json`.

## Run

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/scrape.py          # optional: data/docs.jsonl is included
python scripts/build_index.py
uvicorn app.main:app              # http://localhost:8000
python scripts/evaluate.py extractive local
pytest -q                         # uses fake models, no downloads
```

Docker: `docker compose up --build`. To use a hosted model, set `LLM_BACKEND=openai OPENAI_API_KEY=… [OPENAI_BASE_URL=… LLM_MODEL=…]`.

## Data & licence

Source: Office of the Australian Information Commissioner website, www.oaic.gov.au, © Commonwealth of Australia, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `data/docs.jsonl` contains extracted text: footnotes are dropped, and email addresses and phone numbers are masked. This is not legal advice.
