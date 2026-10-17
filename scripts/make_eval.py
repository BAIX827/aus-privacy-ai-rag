"""Validate hand-written questions against the corpus: the gold section must exist and contain the answer.
Rows with doc_id NONE are out-of-scope questions that should get "not found"."""

import csv
import json

docs = [json.loads(l) for l in open("data/docs.jsonl")]
out, dropped = [], []
for q in csv.DictReader(open("eval/questions_raw.tsv"), delimiter="\t"):
    if q["doc_id"] == "NONE":  # out-of-scope: expect "not found"
        out.append(q)
        continue
    hits = [d for d in docs if d["doc_id"] == q["doc_id"] and q["section"] in d["section"]]
    if hits and any(q["answer"].lower() in d["text"].lower() for d in hits):
        out.append({**q, "section": hits[0]["section"]})
    else:
        dropped.append(q["question"])
json.dump(out, open("eval/eval_set.json", "w"), indent=1, ensure_ascii=False)
print(len(out), "kept;", "dropped:", dropped)
