"""Monitoring simulation: which logged signal notices that the issue stream has changed?

Streams: CLEAN_N requests drawn from the NLBSE'24 test set, then SHIFT_N from a shifted source:
fresh 2025-26 issues, a repository the service has no history for, or non-native rewrites. Signals
per request (few-shot LLM unless noted): handed to a human (confidence below the calibrated
threshold), 1 - calibrated confidence, disagreement with the embedding classifier, and novelty
(1 - cosine similarity to the closest labeled history issue it may retrieve from). A window-level
signal compares the predicted-label distribution in the last WINDOW requests with the calibration
distribution (total variation). Alarm: windowed value > clean mean + 3 SD, where mean and SD come
from a separate clean calibration stream. The clean test issues are split in two disjoint halves:
one feeds the calibration streams, the other the evaluation streams, on which false alarms are
measured before the shift.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triage.data import LABELS, load
from triage.selective import fit_temperature, label_index, logit_matrix, risk_threshold, softmax, split_fit_cert

CLEAN_N, SHIFT_N, WINDOW, RUNS = 400, 200, 50, 300
rows = [json.loads(l) for f in ("results/llm_Qwen2.5-7B-Instruct-4bit.jsonl", "results/classical.jsonl")
        for l in Path(f).read_text().splitlines()]
R = defaultdict(dict)
for r in rows:
    R[(r["approach"], r["condition"])][r["id"]] = r
ds = load()
E = dict(np.load("data/emb.npz", allow_pickle=False))
H = ds["history"]
HE = np.stack([E[h["id"]] for h in H])
hrepo = np.array([h["repo"] for h in H])

fit, cert = split_fit_cert(list(R[("few", "calib")].values()))
T = fit_temperature(logit_matrix(fit), label_index(fit))
pc = softmax(logit_matrix(cert), T)
tau = risk_threshold(pc.max(1), pc.argmax(1) == label_index(cert))
cal_dist = np.bincount(pc.argmax(1), minlength=3) / len(cert)


def signals(r, cond):
    p = softmax(logit_matrix([r]), T)[0]
    e = R[("emb", cond)].get(r["id"]) or R[("emb", "test")].get(r.get("base_id", r["id"]))
    same = cond != "newrepo"
    mask = (hrepo == r["repo"]) if same else (hrepo != r["repo"])
    nov = 1 - float((HE[mask] @ E[r["id"]]).max())
    return dict(deferred=float(p.max() < tau), low_conf=float(1 - p.max()),
                disagree=float(e is not None and e["pred"] != r["pred"]), novelty=nov,
                pred=int(p.argmax()), error=float(r["pred"] != r["label"]))


import hashlib  # noqa: E402

_test = sorted(R[("few", "test")].values(), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
pools = {"clean_cal": [signals(r, "test") for r in _test[::2]],
         "clean": [signals(r, "test") for r in _test[1::2]],
         "fresh": [signals(r, "fresh") for r in R[("few", "fresh")].values()],
         "newrepo": [signals(r, "newrepo") for r in R[("few", "newrepo")].values()],
         "non_native": [signals(r, "variants") for r in R[("few", "variants")].values()
                        if r["kind"] == "non_native" and r["applicable"]]}
SIG = ["deferred", "low_conf", "disagree", "novelty", "label_shift"]
SNAME = {"deferred": "handed to human", "low_conf": "low confidence", "disagree": "LLM-classifier disagreement",
         "novelty": "novelty vs. history", "label_shift": "predicted-label shift"}
rng = np.random.default_rng(0)


def rolling(x):
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[WINDOW:] - c[:-WINDOW]) / WINDOW


def label_shift(preds):
    out = []
    for i in range(len(preds) - WINDOW + 1):
        d = np.bincount(preds[i:i + WINDOW], minlength=3) / WINDOW
        out.append(0.5 * np.abs(d - cal_dist).sum())
    return np.array(out)


res = {s: {k: dict(delays=[], fa=[]) for k in SIG} for s in ("fresh", "newrepo", "non_native")}
example = {}
for scen in res:
    for run in range(RUNS):
        calib_stream = [pools["clean_cal"][i] for i in rng.integers(0, len(pools["clean_cal"]), CLEAN_N)]
        a = [pools["clean"][i] for i in rng.integers(0, len(pools["clean"]), CLEAN_N)]
        b = [pools[scen][i] for i in rng.integers(0, len(pools[scen]), SHIFT_N)]
        st = a + b
        for s in SIG:
            series = (lambda xs: label_shift(np.array([x["pred"] for x in xs]))) if s == "label_shift" else \
                     (lambda xs: rolling(np.array([x[s] for x in xs])))
            base = series(calib_stream)
            thr = base.mean() + 3 * base.std() + 1e-9
            roll = series(st)
            clean = roll[: CLEAN_N - WINDOW + 1]
            alarms = np.where(roll > thr)[0] + WINDOW - 1
            after = alarms[alarms >= CLEAN_N]
            res[scen][s]["delays"].append(int(after[0] - CLEAN_N + 1) if len(after) else None)
            res[scen][s]["fa"].append(float((clean > thr).mean()))
            if run == 0:
                example.setdefault(scen, {})[s] = roll
        if run == 0:
            example[scen]["error"] = rolling(np.array([x["error"] for x in st]))

md = ["# Monitoring simulation (Qwen2.5-7B few-shot)", "",
      f"{RUNS} streams per shift: {CLEAN_N} NLBSE'24 test requests, then {SHIFT_N} shifted requests. Window {WINDOW}; "
      "alarm when the windowed value exceeds mean + 3 SD of a separate clean calibration stream (disjoint half of the "
      "test issues). Cells: median delay in requests [IQR], share of runs detected, false-alarm rate on the evaluation "
      "stream's clean windows.", "",
      "| Shift | " + " | ".join(SNAME[s] for s in SIG) + " |", "|---|" + "---:|" * len(SIG)]
summ = {}
for scen in res:
    cells = []
    for s in SIG:
        d = [x for x in res[scen][s]["delays"] if x is not None]
        det, fa = len(d) / RUNS, float(np.mean(res[scen][s]["fa"]))
        summ.setdefault(scen, {})[s] = dict(detected=det, median_delay=float(np.median(d)) if d else None, false_alarm=fa)
        cells.append(f"{np.median(d):.0f} [{np.percentile(d, 25):.0f}–{np.percentile(d, 75):.0f}], {100 * det:.0f}%, {100 * fa:.1f}% FA"
                     if d else f"not detected, {100 * fa:.1f}% FA")
    md.append(f"| {scen} | " + " | ".join(cells) + " |")
md += ["", "Mean signal per request in each pool:", "", "| Pool | error | " + " | ".join(SNAME[s] for s in SIG[:-1]) + " |",
       "|---|" + "---:|" * len(SIG)]
for k, pool in pools.items():
    md.append(f"| {k} (n={len(pool)}) | {100 * np.mean([x['error'] for x in pool]):.1f} | " +
              " | ".join(f"{np.mean([x[s] for x in pool]):.3f}" for s in SIG[:-1]) + " |")
Path("report/monitoring.md").write_text("\n".join(md) + "\n")
Path("report/monitoring.json").write_text(json.dumps(summ, indent=1))

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = dict(zip(SIG, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), facecolor=SURF, sharey=True)
for ax, scen in zip(axes, res):
    ax.set_facecolor(SURF)
    for s in SIG:
        roll = example[scen][s]
        base = roll[: CLEAN_N - WINDOW + 1]
        z = (roll - base.mean()) / (base.std() + 1e-9)  # standardized so signals share one axis (display only)
        ax.plot(np.arange(len(roll)) + WINDOW - 1, z, color=COL[s], lw=1.5, label=SNAME[s])
    ax.axhline(3, color=INK2, lw=0.8, ls=":")
    ax.axvline(CLEAN_N, color=INK2, lw=1, ls="--")
    ax.text(CLEAN_N + 6, ax.get_ylim()[1] if False else 14, "shift", color=INK2, fontsize=8.5)
    ax.set_title({"fresh": "2025-26 issues", "newrepo": "new repository", "non_native": "non-native English"}[scen],
                 loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("Request")
    ax.set_ylim(-4, 16)
    ax.grid(True, color=GRID, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Windowed signal (SD from clean)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=5, labelcolor=INK)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("report/figures/monitoring.png", dpi=180, facecolor=SURF)
fig.savefig("report/figures/monitoring.pdf", facecolor=SURF)
print("\n".join(md))
