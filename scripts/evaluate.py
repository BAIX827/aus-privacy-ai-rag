"""Retrieval (recall@k, MRR) per retriever and answer quality per LLM backend on eval/eval_set.json.

usage: python scripts/evaluate.py [backend ...]   (default: extractive local)
"""

import json
import sys
import time

from privrag.generate import NOT_FOUND
from privrag.rag import answer
from privrag.retrieve import Retriever

KS, DEPTH = (1, 4, 10), 10


def is_gold(chunk, q):
    return chunk["doc_id"] == q["doc_id"] and chunk["section"] == q["section"]


def retrieval(r, qs):
    rows = {}
    for mode in ("bm25", "dense", "hybrid", "rerank"):
        ranks = []
        for q in qs:
            hits = [c for c, _ in r.search(q["question"], mode=mode, k=DEPTH)]
            ranks.append(next((i + 1 for i, c in enumerate(hits) if is_gold(c, q)), None))
        rows[mode] = {
            **{f"recall@{k}": sum(1 for x in ranks if x and x <= k) / len(qs) for k in KS},
            "mrr@10": sum(1 / x for x in ranks if x) / len(qs),
        }
    return rows


def generation(r, qs, oos, backend):
    exact = cite = abstain = 0
    t = time.time()
    for q in qs:
        out = answer(q["question"], r, backend)
        exact += q["answer"].lower() in out["answer"].lower()
        cite += any(s["doc_id"] == q["doc_id"] and s["section"] == q["section"] for s in out["sources"])
        abstain += out["answer"] == NOT_FOUND
    refused = sum(answer(q["question"], r, backend)["answer"] == NOT_FOUND for q in oos)
    n = len(qs)
    return {
        "answer_contains": exact / n,
        "citation_correct": cite / n,
        "false_not_found": abstain / n,
        "out_of_scope_refused": refused / len(oos),
        "sec_per_question": (time.time() - t) / (n + len(oos)),
    }


def main(backends):
    data = json.load(open("eval/eval_set.json"))
    qs = [q for q in data if q["doc_id"] != "NONE"]
    oos = [q for q in data if q["doc_id"] == "NONE"]
    r = Retriever()
    res = {
        "n_questions": len(qs),
        "n_out_of_scope": len(oos),
        "retrieval": retrieval(r, qs),
        "generation": {b: generation(r, qs, oos, b) for b in backends},
    }
    json.dump(res, open("eval/results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["extractive", "local"])
