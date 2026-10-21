"""Build a blind audit sheet of judge decisions for two human annotators.

Samples every judge acceptance and an equal number of random rejections from the test answers,
shuffles them, and writes report/audit/judge_audit.csv (no verdicts) plus a key file. Annotators
fill `annotator_1` / `annotator_2` with 1 (same answer as the reference) or 0 (different), then
run scripts/score_audit.py.
"""
import csv
import json
import random
from pathlib import Path

d = json.loads(Path("data/judgments.json").read_text())
ds = json.loads(Path("data/dataset.json").read_text())
test_q = {q["question"] for q in ds["test"]}
items = [dict(k=k, **v) for k, v in d.items() if v["question"] in test_q]
yes = [x for x in items if x["equivalent"]]
no = [x for x in items if not x["equivalent"]]
random.seed(544)
sample = yes + random.sample(no, min(len(no), max(len(yes), 60)))
random.shuffle(sample)
out = Path("report/audit")
out.mkdir(parents=True, exist_ok=True)
with (out / "judge_audit.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "question", "reference_answer", "predicted_answer", "annotator_1", "annotator_2", "notes"])
    for i, x in enumerate(sample):
        w.writerow([i, x["question"], x["gold"], x["pred"], "", "", ""])
(out / "judge_audit_key.json").write_text(json.dumps({i: int(x["equivalent"]) for i, x in enumerate(sample)}, indent=1))
print(f"{len(sample)} items ({len(yes)} judge acceptances) -> {out / 'judge_audit.csv'}")
