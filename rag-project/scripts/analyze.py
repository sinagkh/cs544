"""Aggregate result JSONL files into tables (Markdown + JSON) and figures."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ragrepair.data import load_dataset
from ragrepair.evaluate import bootstrap_ci, mcnemar_p, paired_diff_ci
from ragrepair.faults import ANSWERABLE, CONDITIONS, FAULTS
from ragrepair.judge import key as judge_key
from ragrepair.text import is_abstain, is_correct

CONDS = list(CONDITIONS)
ORDER = ["no_repair", "retry", "self_check", "self_check_nli", "repair_all", "checks_only",
         "diagnose", "diagnose_llm", "diagnose_nli", "diagnose_norecheck", "diagnose_noabstain", "support_only"]
LABEL = {"no_repair": "No repair", "retry": "Retry", "self_check": "Self-check (LLM)",
         "self_check_nli": "Self-check (MiniCheck)", "repair_all": "Repair-all", "checks_only": "Trace checks only",
         "diagnose": "Diagnose (text match)", "diagnose_llm": "Diagnose (LLM verifier)",
         "diagnose_nli": "Diagnose (MiniCheck)", "diagnose_norecheck": "  − no re-check",
         "diagnose_noabstain": "  − no abstention", "support_only": "  − no trace checks"}
FIG = ["no_repair", "self_check", "self_check_nli", "repair_all", "checks_only", "diagnose"]
# Categorical slots 1-6 of the reference palette (light mode), fixed order.
COLORS = dict(zip(FIG, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COND_LABEL = {"clean": "clean", "encoder": "encoder\nmismatch", "truncation": "chunk\ntruncation",
              "overflow": "context\noverflow", "stale": "stale\nversion", "absent": "missing doc\n(abstain)",
              "encoder+truncation": "encoder +\ntruncation"}

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default="report")
ap.add_argument("--dataset", default="data/dataset.json")
ap.add_argument("--judgments", default="data/judgments.json")
ap.add_argument("--strict", action="store_true", help="string matching only, ignore the judge")
args = ap.parse_args()

rows = [json.loads(l) for f in args.files for l in Path(f).read_text().splitlines()]
ds = load_dataset(args.dataset)
cases = {q["id"]: q for q in ds["dev"] + ds["test"]}
J = {} if args.strict or not Path(args.judgments).exists() else json.loads(Path(args.judgments).read_text())
OVR = Path(args.judgments).with_name("judge_overrides.json")
if J and OVR.exists():  # manual audit corrections of individual judge decisions
    for k, v in json.loads(OVR.read_text())["overrides"].items():
        J[k] = dict(J.get(k, {}), equivalent=v["equivalent"])
judge_accepts = 0


def correct(qid, gold, pred):
    global judge_accepts
    if is_abstain(pred):
        return False
    if is_correct(pred, gold):
        return True
    v = J.get(judge_key(cases[qid]["question"], gold, pred))
    if v and v["equivalent"]:
        judge_accepts += 1
        return True
    return False


for r in rows:
    answerable = r["condition"] in ANSWERABLE
    c0, c1 = correct(r["qid"], r["gold"], r["initial_answer"]), correct(r["qid"], r["gold"], r["final_answer"])
    r["initial"]["correct"], r["correct"] = c0, c1
    r["initial"]["success"] = c0 if answerable else r["initial"]["abstain"]
    r["success"] = c1 if answerable else r["abstain"]

MODEL_ORDER = {"Qwen2.5-7B-Instruct-4bit": 0, "Llama-3.2-3B-Instruct-4bit": 1, "gemma-2-9b-it-4bit": 2}
models = sorted({r["model"] for r in rows}, key=lambda m: (MODEL_ORDER.get(m, 9), m))
idx = defaultdict(dict)
for r in rows:
    idx[(r["model"], r["policy"], r["condition"])][r["qid"]] = r
qids = {m: sorted({r["qid"] for r in rows if r["model"] == m}) for m in models}
pols = {m: [p for p in ORDER if (m, p, "clean") in idx] for m in models}
out = Path(args.out)
(out / "figures").mkdir(parents=True, exist_ok=True)


def vec(m, p, c, key="success"):
    d = idx[(m, p, c)]
    return np.array([float(d[q][key]) for q in qids[m] if q in d])


def pooled(m, p):
    return np.array([np.mean([idx[(m, p, c)][q]["success"] for c in FAULTS]) for q in qids[m]])


def pct(x):
    return "–" if x != x else f"{100 * x:.1f}"


def damage(m, p):
    clean = list(idx[(m, p, "clean")].values())
    ok0 = [r for r in clean if r["initial"]["correct"]]
    return float(np.mean([not r["correct"] for r in ok0])) if ok0 else float("nan")


summary = {}
md = ["# Results", "",
      f"Scoring: {'string matching only' if args.strict else 'string matching plus the audited equivalence judge'}"
      f" (the judge accepted {judge_accepts} answer instances that string matching rejected).", ""]
for m in models:
    n = len(qids[m])
    S = summary.setdefault(m, {})
    md += [f"## {m} (n = {n} questions per condition)", "", "### Success rate by condition (%)", "",
           "Success = correct answer in answerable conditions, `INSUFFICIENT` in `absent`.", "",
           "| Policy | " + " | ".join(CONDS) + " | Faults pooled [95% CI] |", "|---|" + "---:|" * (len(CONDS) + 1)]
    for p in pols[m]:
        pv = pooled(m, p)
        lo, hi = bootstrap_ci(pv)
        md.append(f"| {LABEL[p]} | " + " | ".join(pct(vec(m, p, c).mean()) for c in CONDS)
                  + f" | {pct(pv.mean())} [{pct(lo)}, {pct(hi)}] |")
        S[p] = dict(by_condition={c: float(vec(m, p, c).mean()) for c in CONDS},
                    faults_pooled=float(pv.mean()), faults_pooled_ci=[lo, hi])
    md.append("")

    md += ["### Damage, abstention and cost", "",
           "Clean damage = clean questions answered correctly at first and wrong after the policy. Calls include "
           "the initial call; GPU s is generation time; MiniCheck columns are support checks per case and their time.", "",
           "| Policy | Repair % | Clean damage % | Unneeded abstain % | Stale value used % | Calls | Prompt tokens | GPU s | MiniCheck checks | MiniCheck s |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in pols[m]:
        fr = [r for c in FAULTS for r in idx[(m, p, c)].values()]
        failed = [r for r in fr if not r["initial"]["success"]]
        repair = float(np.mean([r["success"] for r in failed])) if failed else float("nan")
        ans_rows = [r for c in ANSWERABLE for r in idx[(m, p, c)].values()]
        unneeded = float(np.mean([r["abstain"] and not r["initial"]["abstain"] for r in ans_rows]))
        stale_used = float(np.mean([r["stale_value"] for r in idx[(m, p, "stale")].values()]))
        allr = [r for c in CONDS for r in idx[(m, p, c)].values()]
        mean = lambda k: float(np.mean([r.get(k, 0) for r in allr]))
        md.append(f"| {LABEL[p]} | {pct(repair)} | {pct(damage(m, p))} | {pct(unneeded)} | {pct(stale_used)} | "
                  f"{mean('calls'):.2f} | {mean('prompt_tokens'):.0f} | {mean('seconds'):.1f} | "
                  f"{mean('support_checks'):.2f} | {mean('support_seconds'):.2f} |")
        S[p].update(repair=repair, clean_damage=damage(m, p), unneeded_abstain=unneeded, stale_value_used=stale_used,
                    calls=mean("calls"), prompt_tokens=mean("prompt_tokens"), seconds=mean("seconds"),
                    support_checks=mean("support_checks"), support_seconds=mean("support_seconds"))
    md.append("")

    if all(p in pols[m] for p in ("self_check", "self_check_nli", "diagnose_llm", "diagnose_nli", "diagnose")):
        md += ["### Verifier × repair (2 × 2)", "",
               "Rows: what decides that an answer is unsupported. Columns: what happens next. "
               "Cells: success under faults / clean damage (%).", "",
               "| Verifier | Regenerate on the same passages | Repair the context (trace checks + retrieval ladder) |",
               "|---|---:|---:|"]
        for lab, a, b in (("LLM self-verifier", "self_check", "diagnose_llm"), ("MiniCheck", "self_check_nli", "diagnose_nli"),
                          ("Text match", None, "diagnose")):
            if a is None:
                md.append(f"| {lab} | – | {pct(S[b]['faults_pooled'])} / {pct(S[b]['clean_damage'])} |")
                continue
            md.append(f"| {lab} | {pct(S[a]['faults_pooled'])} / {pct(S[a]['clean_damage'])} | "
                      f"{pct(S[b]['faults_pooled'])} / {pct(S[b]['clean_damage'])} |")
        md.append("")

    ref = "diagnose" if "diagnose" in pols[m] else pols[m][-1]
    md += [f"### {LABEL[ref].strip()} vs. each policy (paired by question)", "",
           "Success difference in percentage points (reference minus row). Pooled CI: bootstrap over questions. "
           "Per-condition p: exact McNemar.", "",
           "| Policy | Faults pooled Δ [95% CI] | " + " | ".join(f"{c} Δ (p)" for c in CONDS) + " |",
           "|---|---:|" + "---:|" * len(CONDS)]
    for p in pols[m]:
        if p == ref:
            continue
        d, lo, hi = paired_diff_ci(pooled(m, ref), pooled(m, p))
        cells = []
        for c in CONDS:
            a, b = vec(m, ref, c), vec(m, p, c)
            cells.append(f"{100 * (a.mean() - b.mean()):+.1f} ({mcnemar_p(a, b):.3f})")
        md.append(f"| {LABEL[p]} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | " + " | ".join(cells) + " |")
        S[p]["ref_minus_this_pooled"] = [d, lo, hi]
    md.append("")

    md += ["### Fault delivery and detection (% of cases)", "",
           "Delivery columns use the gold answer (evaluation only); flag columns are the trace checks of "
           f"`{ref}` on the initial answer.", "",
           "| Condition | Answer text not seen by model | Both gold passages retrieved | Stale version in prompt | "
           "Prompt cut by server | Flag: truncation | Flag: version | Flag: overflow | Flag: low retrieval | Flag: unsupported | Abstained |",
           "|---|" + "---:|" * 10]
    for c in CONDS:
        rs = list(idx[(m, ref, c)].values())
        dl = lambda k: pct(np.mean([bool(r["delivery"].get(k)) for r in rs]))
        fl = lambda k: pct(np.mean([k in r["flags"] for r in rs]))
        md.append(f"| {c} | {pct(np.mean([not r['delivery']['gold_in_context'] for r in rs]))} | {dl('both_gold_retrieved')} | "
                  f"{dl('stale_in_context')} | {dl('overflow_cut')} | {fl('truncation')} | {fl('version_conflict')} | "
                  f"{fl('overflow')} | {fl('low_retrieval')} | {fl('unsupported')} | {pct(np.mean([r['abstain'] for r in rs]))} |")
    md.append("")

    # Support checks compared on the initial answers (what each gate would decide)
    if all(p in pols[m] for p in ("self_check", "self_check_nli", "no_repair")):
        ex = []
        for c in CONDS:
            for q in qids[m]:
                r0 = idx[(m, "no_repair", c)].get(q)
                r1 = idx[(m, "self_check", c)].get(q)
                r2 = idx[(m, "self_check_nli", c)].get(q)
                if not r0 or is_abstain(r0["initial_answer"]) or not r2 or not r2["support_scores"]:
                    continue
                y = r0["initial"]["correct"] and r0["delivery"]["gold_in_context"]
                ex.append(dict(y=y, llm="verified" in r1["actions"], nli=r2["support_scores"][0]))
        if ex:
            y = np.array([e["y"] for e in ex], bool)
            s = np.array([e["nli"] for e in ex])
            llm = np.array([e["llm"] for e in ex], bool)
            pos, neg = s[y], s[~y]
            auroc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())) \
                if len(pos) and len(neg) else float("nan")
            md += ["### Support checks on initial answers", "",
                   f"{len(ex)} non-abstaining initial answers; positive = correct and the answer text was in what the "
                   f"model saw ({int(y.sum())} positives). Good kept = share of positives judged supported; "
                   "bad caught = share of negatives judged unsupported.", "",
                   "| Check | Good kept % | Bad caught % | Balanced accuracy |", "|---|---:|---:|---:|"]
            for lab, pred in (("LLM self-verifier", llm), (f"MiniCheck ≥ 0.5 (AUROC {auroc:.2f})", s >= 0.5),
                              ("MiniCheck ≥ 0.05 (abstain gate)", s >= 0.05)):
                tpr, tnr = pred[y].mean(), (~pred[~y]).mean()
                md.append(f"| {lab} | {pct(tpr)} | {pct(tnr)} | {(tpr + tnr) / 2:.3f} |")
            md.append("")
            S["_support_eval"] = dict(n=len(ex), positives=int(y.sum()), auroc=auroc,
                                      llm_tpr=float(llm[y].mean()), llm_tnr=float((~llm[~y]).mean()),
                                      nli_tpr=float((s >= 0.5)[y].mean()), nli_tnr=float((s < 0.5)[~y].mean()))

(out / "results.md").write_text("\n".join(md))
(out / "summary.json").write_text(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
figp = {m: [p for p in FIG if p in pols[m]] for m in models}
fig, axes = plt.subplots(len(models), 1, figsize=(12, 3.4 * len(models)), facecolor=SURF, squeeze=False)
for ax, m in zip(axes[:, 0], models):
    ax.set_facecolor(SURF)
    ps = figp[m]
    w = 0.8 / len(ps)
    x = np.arange(len(CONDS))
    for i, p in enumerate(ps):
        ax.bar(x + (i - (len(ps) - 1) / 2) * w, [100 * vec(m, p, c).mean() for c in CONDS], w - 0.02,
               color=COLORS[p], label=LABEL[p], zorder=3)
    ax.set_xticks(x, [COND_LABEL[c] for c in CONDS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(m, loc="left", color=INK, fontsize=11, fontweight="bold")
axes[0, 0].legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(out / "figures" / "success_by_condition.png", dpi=180, facecolor=SURF)
fig.savefig(out / "figures" / "success_by_condition.pdf", facecolor=SURF)

SCAT = [p for p in ORDER if p not in ("diagnose_norecheck", "diagnose_noabstain", "support_only")]
fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.4), facecolor=SURF, squeeze=False)
for ax, m in zip(axes[0], models):
    ax.set_facecolor(SURF)
    pts = [(p, summary[m][p]["prompt_tokens"], 100 * summary[m][p]["faults_pooled"]) for p in SCAT if p in summary[m]]
    for p, x, y in pts:
        ax.scatter(x, y, s=60, color="#2a78d6", edgecolor=SURF, linewidth=1.5, zorder=3)
    OFF = {"no_repair": (-8, 6, "right"), "retry": (7, 6, "left"), "self_check_nli": (8, 3, "left"),
           "self_check": (7, -13, "left"), "checks_only": (-8, -13, "right"), "diagnose": (7, -13, "left"),
           "diagnose_nli": (7, 7, "left"), "diagnose_llm": (7, -13, "left"), "repair_all": (7, -13, "left")}
    for p, x, y in pts:
        dx, dy, ha = OFF.get(p, (6, 7, "left"))
        ax.annotate(LABEL[p].strip(), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                    color=INK, ha=ha)
    ax.set_xlabel("Prompt tokens per case (mean)")
    ax.set_ylabel("Success under faults (%)")
    ax.set_ylim(0, max(70, max(y for _, _, y in pts) + 10))
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(m, loc="left", color=INK, fontsize=11, fontweight="bold")
    xs = [x for _, x, _ in pts]
    ax.set_xlim(min(xs) * 0.45, max(xs) * 1.35)
fig.tight_layout()
fig.savefig(out / "figures" / "cost_vs_success.png", dpi=180, facecolor=SURF)
fig.savefig(out / "figures" / "cost_vs_success.pdf", facecolor=SURF)
print("wrote", out / "results.md")
