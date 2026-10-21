"""Agreement between the two annotators and between each annotator and the judge (Cohen's kappa)."""
import csv
import json
from pathlib import Path


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


rows = list(csv.DictReader(open("report/audit/judge_audit.csv")))
key = json.loads(Path("report/audit/judge_audit_key.json").read_text())
lab = [r for r in rows if r["annotator_1"].strip() in "01" and r["annotator_2"].strip() in "01"
       and r["annotator_1"].strip() and r["annotator_2"].strip()]
a1 = [int(r["annotator_1"]) for r in lab]
a2 = [int(r["annotator_2"]) for r in lab]
j = [key[r["item"]] for r in lab]
print(f"{len(lab)} items labeled by both annotators")
for name, x, y in (("annotator 1 vs 2", a1, a2), ("annotator 1 vs judge", a1, j), ("annotator 2 vs judge", a2, j)):
    k, po = kappa(x, y)
    print(f"{name}: agreement {100 * po:.1f}%, Cohen's kappa {k:.2f}")
