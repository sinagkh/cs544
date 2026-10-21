"""Faithfulness of cited evidence (deletion test) on the probe issues.

For each probe issue the few-shot model is asked for a label and two exact quotes from the issue.
We check that the quotes occur verbatim, then re-score the issue (same retrieved examples) with the
quotes deleted and, as a control, with random spans of the same total length deleted (two seeds).
Faithful evidence should lower the probability of the predicted label more than random deletion.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from triage.data import load
from triage.llm import Cache, MLXModel
from triage.models import Retriever
from triage.prompts import explain, few_shot, parse_explanation
from triage.data import LABELS
from triage.selective import fit_temperature, label_index, logit_matrix, softmax, split_fit_cert

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
args = ap.parse_args()

ds = load()
E = dict(np.load("data/emb.npz", allow_pickle=False))
retriever = Retriever(ds["history"], np.stack([E[h["id"]] for h in ds["history"]]))
test = {x["id"]: x for x in ds["test"]}
probe = [test[i] for i in ds["probe_ids"]]
model = MLXModel(args.model, Cache("cache/llm_cache.sqlite"))
# Raw LLM label probabilities are saturated near 0/1, so compare deletions on temperature-calibrated
# probabilities (temperature fit on the fit third of the few-shot calibration split, as in analyze.py).
_cal = [r for r in map(json.loads, Path(f"results/llm_{args.model.split('/')[-1]}.jsonl").read_text().splitlines())
        if r["condition"] == "calib" and r["approach"] == "few"]
_fit, _ = split_fit_cert(_cal)
T = fit_temperature(logit_matrix(_fit), label_index(_fit))


def cal_probs(r):
    p = softmax(np.array([[r["logits"][l] for l in LABELS]]), T)[0]
    return dict(zip(LABELS, p))


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(text[last:a])
        last = max(last, b)
    out.append(text[last:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


def random_spans(text: str, lengths: list[int], rng: random.Random) -> list[tuple[int, int]]:
    spans = []
    for L in lengths:
        if len(text) <= L:
            return []
        a = rng.randrange(0, len(text) - L)
        spans.append((a, a + L))
    return spans


rows = []
for it in probe:
    ex = retriever.topk(E[it["id"]], it["repo"], 6)
    gen = model.generate(explain(it, ex), max_tokens=90)["text"]
    lab, quotes = parse_explanation(gen)
    base = cal_probs(model.label_probs(few_shot(it, ex)))
    pred = max(base, key=base.get)
    body = it["body"]
    spans, verbatim = [], []
    for q in quotes:
        i = body.lower().find(q.lower())
        if i < 0:  # tolerate whitespace differences
            nb = norm(body)
            verbatim.append(norm(q) in nb or norm(q) in norm(it["title"]))
            continue
        spans.append((i, i + len(q)))
        verbatim.append(True)
    rec = dict(id=it["id"], repo=it["repo"], label=it["label"], pred=pred, explained_label=lab, quotes=quotes,
               verbatim=verbatim, n_quotes=len(quotes), deletable=len(spans))
    if spans:
        def prob_after(b):
            r = cal_probs(model.label_probs(few_shot(dict(it, body=b), ex)))
            return r[pred], max(r, key=r.get)
        p_ev, pred_ev = prob_after(delete_spans(body, spans))
        lens = [b - a for a, b in spans]
        rand = [prob_after(delete_spans(body, random_spans(body, lens, random.Random(s)))) for s in (1, 2)]
        rec.update(p0=base[pred], p_evidence=p_ev, flip_evidence=pred_ev != pred,
                   p_random=float(np.mean([r[0] for r in rand])), flip_random=float(np.mean([r[1] != pred for r in rand])),
                   deleted_chars=sum(lens), body_chars=len(body))
    rows.append(rec)

Path("results/explanations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
q = [r for r in rows if "p_evidence" in r]
quoted = sum(r["n_quotes"] > 0 for r in rows)
verb = [v for r in rows for v in r["verbatim"]]
d_ev = np.array([r["p0"] - r["p_evidence"] for r in q])
d_rd = np.array([r["p0"] - r["p_random"] for r in q])
md = ["# Evidence faithfulness (deletion test, temperature-calibrated probabilities)", "",
      f"- {len(rows)} probe issues; {quoted} explanations contained quotes; {100 * np.mean(verb):.1f}% of {len(verb)} quotes "
      f"occur verbatim in the issue.",
      f"- Explained label equals the scored label in {100 * np.mean([r['explained_label'] == r['pred'] for r in rows]):.1f}% of issues.",
      f"- Deletion test on {len(q)} issues with deletable quotes (mean {np.mean([r['deleted_chars'] for r in q]):.0f} of "
      f"{np.mean([r['body_chars'] for r in q]):.0f} body characters removed):",
      f"  - mean drop in the predicted label's probability: evidence {d_ev.mean():.3f}, random {d_rd.mean():.3f} "
      f"(Wilcoxon p = {wilcoxon(d_ev, d_rd).pvalue:.3g})",
      f"  - label flips: evidence {100 * np.mean([r['flip_evidence'] for r in q]):.1f}%, random {100 * np.mean([r['flip_random'] for r in q]):.1f}%",
      f"  - evidence deletion lowers confidence more than random deletion in {100 * np.mean(d_ev > d_rd):.1f}% of issues"]
Path("report").mkdir(exist_ok=True)
Path("report/explanations.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
