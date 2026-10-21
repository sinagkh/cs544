"""Is the self-verifier problem a small-model artifact? Compare verifiers on the same initial answers.

Takes a stratified sample of non-abstaining initial answers from one model's test run, rebuilds the
exact passages the model saw, and asks a stronger model (Claude Haiku 4.5 via the logged-in Claude
Code CLI) the same SUPPORTED/UNSUPPORTED question the self-check policy uses. Compares it with the
small model's own verdicts and with MiniCheck on the same items. Overflow cases are skipped because
there the verifier prompt itself would be cut by the serving fault.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ragrepair.data import load_dataset
from ragrepair.faults import FaultyIndex
from ragrepair.judge import key as judge_key
from ragrepair.llm import ClaudeCLIModel, ResponseCache
from ragrepair.prompts import parse_verdict, render_passages, verify_messages
from ragrepair.store import Store
from ragrepair.text import is_abstain, is_correct

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--per-condition", type=int, default=40)
ap.add_argument("--model", default="claude-haiku-4-5")
ap.add_argument("--out", default="report/verifier_study.md")
args = ap.parse_args()

ds = load_dataset("data/dataset.json")
cases = {q["id"]: q for q in ds["test"]}
store = Store(ds)
J = json.loads(Path("data/judgments.json").read_text()) if Path("data/judgments.json").exists() else {}
by = defaultdict(dict)
for line in Path(args.results).read_text().splitlines():
    r = json.loads(line)
    by[(r["qid"], r["condition"])][r["policy"]] = r


def ok(q, pred):
    if is_abstain(pred):
        return False
    if is_correct(pred, q["answer"]):
        return True
    v = J.get(judge_key(q["question"], q["answer"], pred))
    return bool(v and v["equivalent"])


random.seed(544)
items = []
for cond in ["clean", "encoder", "truncation", "stale", "absent", "encoder+truncation"]:
    pool = [(q, p) for (q, c), p in by.items() if c == cond and not is_abstain(p["no_repair"]["initial_answer"])
            and p.get("self_check_nli", {}).get("support_scores")]
    for qid, pols in random.sample(pool, min(args.per_condition, len(pool))):
        case = cases[qid]
        idx = FaultyIndex(store, case, cond)
        rendered = render_passages(idx.retrieve(5), idx.template_max_chars())
        ans = pols["no_repair"]["initial_answer"]
        y = ok(case, ans) and pols["no_repair"]["delivery"]["gold_in_context"]
        items.append(dict(cond=cond, qid=qid, answer=ans, y=y, rendered=rendered,
                          small="verified" in pols["self_check"]["actions"],
                          nli=pols["self_check_nli"]["support_scores"][0]))

strong = ClaudeCLIModel(args.model, ResponseCache("cache/claude_cache.sqlite"))
msgs = [verify_messages(cases[it["qid"]]["question"], it["rendered"], it["answer"]) for it in items]
todo = [i for i, m in enumerate(msgs) if not strong.cached(m)]
print(len(items) - len(todo), "cached;", len(todo), "to ask", flush=True)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

with ThreadPoolExecutor(max_workers=4) as pool:  # CLI start-up dominates; calls are independent
    for n, (i, raw) in enumerate(zip(todo, pool.map(lambda i: strong.call(msgs[i]), todo)), 1):
        strong.store(msgs[i], raw)
        if n % 25 == 0:
            print(n, "of", len(todo), flush=True)
for it, m in zip(items, msgs):
    it["strong"] = parse_verdict(strong.generate(m, max_tokens=6).text)

y = np.array([it["y"] for it in items], bool)
small_name = json.loads(Path(args.results).read_text().splitlines()[0])["model"]
md = [f"# Verifier study ({len(items)} initial answers of {small_name}; {int(y.sum())} correct and grounded)", "",
      "Good kept = correct, grounded answers judged supported. Bad caught = other answers judged unsupported.", "",
      "| Verifier | Good kept % | Bad caught % | Balanced accuracy |", "|---|---:|---:|---:|"]
res = {}
for name, pred in ((f"{small_name} (self-verifier)", np.array([it["small"] for it in items])),
                   (f"{args.model} (same prompt)", np.array([it["strong"] for it in items])),
                   ("MiniCheck-RoBERTa-L, score >= 0.5", np.array([it["nli"] >= 0.5 for it in items]))):
    tpr, tnr = pred[y].mean(), (~pred[~y]).mean()
    md.append(f"| {name} | {100 * tpr:.1f} | {100 * tnr:.1f} | {(tpr + tnr) / 2:.3f} |")
    res[name] = dict(good_kept=float(tpr), bad_caught=float(tnr))
md += ["", "By condition (balanced accuracy):", "", "| Condition | n | self-verifier | strong verifier | MiniCheck |",
       "|---|---:|---:|---:|---:|"]
for cond in dict.fromkeys(it["cond"] for it in items):
    sub = [it for it in items if it["cond"] == cond]
    yy = np.array([it["y"] for it in sub], bool)

    def ba(pred):
        pred = np.array(pred, bool)
        parts = [pred[yy].mean() if yy.any() else np.nan, (~pred[~yy]).mean() if (~yy).any() else np.nan]
        return np.nanmean(parts)
    md.append(f"| {cond} | {len(sub)} | {ba([i['small'] for i in sub]):.2f} | {ba([i['strong'] for i in sub]):.2f} | "
              f"{ba([i['nli'] >= 0.5 for i in sub]):.2f} |")
Path(args.out).write_text("\n".join(md) + "\n")
Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
print("\n".join(md))
