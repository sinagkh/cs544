"""Simulate production monitoring: how fast does each signal detect a bad deploy?

For each fault, a stream of requests is drawn from the test questions: `CLEAN_N` requests under the
clean condition, then `FAULT_N` under the fault (a deploy that introduces it). Each request carries
the signals a service would log for its first answer:

  trace checks   any of the free flags (truncation / version conflict / overflow / low retrieval)
  MiniCheck      the support score of the answer is below 0.5
  LLM verifier   the model's own verifier says UNSUPPORTED
  abstention     the model answered INSUFFICIENT

A monitor alarms when the rate of its signal over the last `WINDOW` requests exceeds the clean
mean plus 3 standard deviations of the windowed rate, both estimated on the clean segment. We
report the detection delay (requests after the deploy until the first alarm) and the false alarm
rate during the clean segment, over `RUNS` random streams.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ragrepair.faults import FAULTS
from ragrepair.text import is_abstain

CLEAN_N, FAULT_N, WINDOW, RUNS = 400, 200, 50, 300
# Share of post-deploy requests the fault affects. Encoder, template and serving faults hit every
# request; an outdated document only affects the questions whose evidence it holds.
PREVALENCE = {"stale": 0.2}
FREE = ("truncation", "version_conflict", "overflow", "low_retrieval")
SIGNALS = ["trace checks", "MiniCheck", "LLM verifier", "abstention"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIG_COLORS = dict(zip(SIGNALS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--out", default="report")
args = ap.parse_args()

by = defaultdict(dict)
for line in Path(args.results).read_text().splitlines():
    r = json.loads(line)
    by[(r["qid"], r["condition"])][r["policy"]] = r
model = next(iter(by.values()))["no_repair"]["model"]


def signals(pols: dict) -> dict:
    nli = pols["self_check_nli"]["support_scores"]
    init_abstain = is_abstain(pols["no_repair"]["initial_answer"])
    return {
        "trace checks": any(f in pols["diagnose"]["flags"] for f in FREE),
        "MiniCheck": init_abstain or (bool(nli) and nli[0] < 0.5),
        "LLM verifier": init_abstain or "verified_unsupported" in pols["self_check"]["actions"],
        "abstention": init_abstain,
    }


table = {c: {q: signals(p) for (q, cc), p in by.items() if cc == c} for c in ["clean"] + FAULTS}
qids = sorted(table["clean"])
rng = np.random.default_rng(0)


def stream(fault: str):
    a = [table["clean"][q] for q in rng.choice(qids, CLEAN_N)]
    prev = PREVALENCE.get(fault, 1.0)
    b = [table[fault if rng.random() < prev else "clean"][q] for q in rng.choice(qids, FAULT_N)]
    return a + b


def rolling(x: np.ndarray) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[WINDOW:] - c[:-WINDOW]) / WINDOW  # rate over requests [i, i+WINDOW)


res = {f: {s: dict(delays=[], false_alarm=[]) for s in SIGNALS} for f in FAULTS}
example = {}
for f in FAULTS:
    for run in range(RUNS):
        st = stream(f)
        for s in SIGNALS:
            x = np.array([r[s] for r in st], float)
            roll = rolling(x)  # roll[i] covers requests i .. i+WINDOW-1; it is known at request i+WINDOW-1
            clean_roll = roll[: CLEAN_N - WINDOW + 1]
            thr = clean_roll.mean() + 3 * clean_roll.std() + 1e-9
            alarms = np.where(roll > thr)[0] + WINDOW - 1  # request index at which the alarm fires
            res[f][s]["false_alarm"].append(float(np.mean(clean_roll > thr)))
            after = alarms[alarms >= CLEAN_N]
            res[f][s]["delays"].append(int(after[0] - CLEAN_N + 1) if len(after) else None)
            if run == 0:
                example.setdefault(f, {})[s] = (roll, thr)

md = [f"# Monitoring simulation ({model})", "",
      f"{RUNS} random streams per fault: {CLEAN_N} clean requests, then {FAULT_N} after the deploy "
      f"(stale affects {int(100 * PREVALENCE['stale'])}% of post-deploy requests, the other faults all of them). "
      f"Rolling window {WINDOW}; alarm when the windowed rate exceeds clean mean + 3 SD. "
      "Delay = requests after the deploy until the first alarm (median [IQR]); "
      "detected = share of runs with an alarm within the fault segment; "
      "false alarms = share of clean windows above the threshold.", "",
      "| Fault | " + " | ".join(SIGNALS) + " |", "|---|" + "---:|" * len(SIGNALS)]
summary = {}
for f in FAULTS:
    cells = []
    for s in SIGNALS:
        d = [x for x in res[f][s]["delays"] if x is not None]
        det = len(d) / RUNS
        fa = float(np.mean(res[f][s]["false_alarm"]))
        if d:
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            cells.append(f"{med:.0f} [{q1:.0f}–{q3:.0f}], {100 * det:.0f}% det., {100 * fa:.1f}% FA")
        else:
            cells.append(f"not detected, {100 * fa:.1f}% FA")
        summary.setdefault(f, {})[s] = dict(detected=det, median_delay=float(np.median(d)) if d else None,
                                            false_alarm=fa)
    md.append(f"| {f} | " + " | ".join(cells) + " |")
md.append("")
rate = {s: {c: float(np.mean([v[s] for v in table[c].values()])) for c in ["clean"] + FAULTS} for s in SIGNALS}
md += ["Signal rates per condition (% of requests):", "", "| Signal | " + " | ".join(["clean"] + FAULTS) + " |",
       "|---|" + "---:|" * (len(FAULTS) + 1)]
for s in SIGNALS:
    md.append(f"| {s} | " + " | ".join(f"{100 * rate[s][c]:.1f}" for c in ["clean"] + FAULTS) + " |")
out = Path(args.out)
(out / "figures").mkdir(parents=True, exist_ok=True)
short = model.replace("-Instruct-4bit", "").replace("-it-4bit", "")
(out / f"monitoring_{short}.md").write_text("\n".join(md) + "\n")
(out / f"monitoring_{short}.json").write_text(json.dumps(dict(summary=summary, rates=rate), indent=1))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
show = ["truncation", "overflow", "stale", "encoder"]
fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 2.9), facecolor=SURF, sharey=True)
for ax, f in zip(axes, show):
    ax.set_facecolor(SURF)
    for s in SIGNALS:
        roll, _ = example[f][s]
        xs = np.arange(len(roll)) + WINDOW - 1
        ax.plot(xs, 100 * roll, color=SIG_COLORS[s], lw=1.6, label=s)
    ax.axvline(CLEAN_N, color=INK2, lw=1, ls="--")
    ax.text(CLEAN_N + 6, 96, "deploy", color=INK2, fontsize=8.5, va="top")
    ax.set_title(f, loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Request")
    ax.grid(True, color=GRID, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel(f"Signal rate, last {WINDOW} requests (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=len(SIGNALS), labelcolor=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out / "figures" / f"monitoring_{short}.png", dpi=180, facecolor=SURF)
fig.savefig(out / "figures" / f"monitoring_{short}.pdf", facecolor=SURF)
print("\n".join(md))
