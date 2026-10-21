"""Tables and figures for the triage study."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triage.data import LABELS, REPOS
from triage.selective import (TARGET_RISK, conformal_qhat, conformal_sets, ece, fit_temperature, label_index,
                              logit_matrix, ltt_threshold, macro_f1, risk_threshold, selective_metrics, softmax)

ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="results/llm_Qwen2.5-7B-Instruct-4bit.jsonl")
ap.add_argument("--classical", default="results/classical.jsonl")
ap.add_argument("--out", default="report")
args = ap.parse_args()

rows = [json.loads(l) for f in (args.llm, args.classical) if Path(f).exists() for l in Path(f).read_text().splitlines()]
R = defaultdict(dict)
for r in rows:
    R[(r["approach"], r["condition"])][r["id"]] = r
# zero-shot prompts do not depend on history, so the new-repository condition reuses the test scores
for i, r in list(R[("zero", "test")].items()):
    R[("zero", "newrepo")][i] = r
newrepo_ids = set(R[("few", "newrepo")]) or set(R[("emb", "newrepo")])
R[("zero", "newrepo")] = {i: r for i, r in R[("zero", "newrepo")].items() if i in newrepo_ids}

APPS = [a for a in ("tfidf", "emb", "zero", "few") if (a, "test") in R]
NAME = {"tfidf": "TF-IDF + LR", "emb": "Embedding + LR", "zero": "LLM zero-shot", "few": "LLM few-shot (retrieval)"}
CONDS = [c for c in ("test", "fresh", "newrepo") if any((a, c) in R for a in APPS)]
CNAME = {"test": "NLBSE'24 test", "fresh": "Fresh 2025-26", "newrepo": "New repository"}
out = Path(args.out)
(out / "figures").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def arrays(app, cond, T):
    rs = list(R[(app, cond)].values())
    z, y = logit_matrix(rs), label_index(rs)
    p = softmax(z, T)
    return rs, p, y


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


calib = {}
for a in APPS:
    rs = list(R[(a, "calib")].values())
    z, y = logit_matrix(rs), label_index(rs)
    T = fit_temperature(z, y)
    p = softmax(z, T)
    conf, correct = p.max(1), p.argmax(1) == y
    calib[a] = dict(T=T, tau=risk_threshold(conf, correct), tau_ltt=ltt_threshold(conf, correct), qhat=conformal_qhat(p, y),
                    n=len(rs), acc=float(correct.mean()), ece_raw=ece(softmax(z), y), ece_cal=ece(p, y))

summary = dict(calibration=calib)
md = ["# Triage results", "", "## Calibration (fit on the calibration split)", "",
      "| Approach | n | Accuracy | Temperature | ECE before → after | Naive threshold | Risk-controlled threshold | Conformal q̂ (α=0.1) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
for a in APPS:
    c = calib[a]
    md.append(f"| {NAME[a]} | {c['n']} | {100 * c['acc']:.1f} | {c['T']:.2f} | {c['ece_raw']:.3f} → {c['ece_cal']:.3f} | "
              f"{c['tau']:.3f} | {c['tau_ltt']:.3f} | {c['qhat']:.3f} |")
md.append("")

# Accuracy table
md += ["## Accuracy", "", "Macro-F1 and cross-repo F1 (mean of per-repository macro-F1, the NLBSE'24 ranking metric).", "",
       "| Approach | Condition | n | Accuracy [95% CI] | Macro-F1 | Cross-repo F1 | ECE (calibrated) | Prompt tokens | s / issue |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
acc = defaultdict(dict)
for a in APPS:
    for c in CONDS:
        if (a, c) not in R or not R[(a, c)]:
            continue
        rs, p, y = arrays(a, c, calib[a]["T"])
        pred = p.argmax(1)
        ok = pred == y
        lo, hi = boot_ci(ok)
        repo = np.array([r["repo"] for r in rs])
        xf1 = np.mean([macro_f1(y[repo == rp], pred[repo == rp]) for rp in REPOS if (repo == rp).any()])
        tok = np.mean([r.get("prompt_tokens", 0) for r in rs])
        sec = np.mean([r.get("seconds", 0) for r in rs])
        acc[a][c] = dict(n=len(rs), acc=float(ok.mean()), ci=[lo, hi], macro_f1=macro_f1(y, pred), xrepo_f1=float(xf1),
                         ece=ece(p, y), prompt_tokens=float(tok), seconds=float(sec))
        md.append(f"| {NAME[a]} | {CNAME[c]} | {len(rs)} | {100 * ok.mean():.1f} [{100 * lo:.1f}, {100 * hi:.1f}] | "
                  f"{macro_f1(y, pred):.3f} | {xf1:.3f} | {ece(p, y):.3f} | {tok:.0f} | {sec:.2f} |")
md.append("")
summary["accuracy"] = acc

# Selective automation
md += ["## Selective automation (target: at most 10% errors among auto-labeled issues)", "",
       "Naive threshold = lowest threshold with ≤10% calibration error; risk-controlled = Learn-then-Test with a 95% "
       "Clopper-Pearson bound. Thresholds and conformal sets are calibrated on the calibration split and applied unchanged. "
       "Agreement = auto-label only when the LLM few-shot and embedding classifiers agree. "
       "Cells: coverage % / risk % (errors among auto-labeled).", "",
       "| Approach | Policy | " + " | ".join(CNAME[c] for c in CONDS) + " |", "|---|---|" + "---:|" * len(CONDS)]
sel = defaultdict(dict)
for a in APPS:
    for pol in ("auto-all", "threshold", "risk-controlled", "conformal"):
        cells = []
        for c in CONDS:
            if not R.get((a, c)):
                cells.append("–")
                continue
            rs, p, y = arrays(a, c, calib[a]["T"])
            correct = p.argmax(1) == y
            if pol == "auto-all":
                auto = np.ones(len(y), bool)
            elif pol == "threshold":
                auto = p.max(1) >= calib[a]["tau"]
            elif pol == "risk-controlled":
                auto = p.max(1) >= calib[a]["tau_ltt"]
            else:
                S = conformal_sets(p, calib[a]["qhat"])
                auto = S.sum(1) == 1
                sel[a].setdefault(c, {})["conformal_set_coverage"] = float(S[np.arange(len(y)), y].mean())
            m = selective_metrics(auto, correct)
            sel[a].setdefault(c, {})[pol] = m
            cells.append(f"{100 * m['coverage']:.1f} / {100 * m['risk']:.1f}")
        md.append(f"| {NAME[a]} | {pol} | " + " | ".join(cells) + " |")
if "few" in APPS and "emb" in APPS:
    cells = []
    for c in CONDS:
        f, e = R[("few", c)], R[("emb", c)]
        ids = [i for i in f if i in e]
        auto = np.array([f[i]["pred"] == e[i]["pred"] for i in ids])
        correct = np.array([f[i]["pred"] == f[i]["label"] for i in ids])
        m = selective_metrics(auto, correct)
        sel["few"].setdefault(c, {})["agreement"] = m
        cells.append(f"{100 * m['coverage']:.1f} / {100 * m['risk']:.1f}")
    md.append("| LLM few-shot + Embedding | agreement | " + " | ".join(cells) + " |")
md += ["", "Conformal prediction-set coverage (target ≥ 90%): " + "; ".join(
    f"{NAME[a]}: " + ", ".join(f"{CNAME[c]} {100 * sel[a][c]['conformal_set_coverage']:.1f}%" for c in CONDS if c in sel[a])
    for a in APPS), ""]
summary["selective"] = sel

# Robustness to rewording
var = defaultdict(lambda: defaultdict(list))
for a in APPS:
    for i, r in R.get((a, "variants"), {}).items():
        var[a][r["kind"]].append(r)
KINDS = ["no_code", "no_template", "newcomer", "non_native", "non_native+no_code"]
KNAME = {"no_code": "no code / stack trace", "no_template": "no template", "newcomer": "newcomer cue",
         "non_native": "non-native English", "non_native+no_code": "non-native + no code"}
rob = defaultdict(dict)
if var:
    md += ["## Robustness to meaning-preserving rewording", "",
           "Flip = the label differs from the label on the original issue. Auto flips = both versions auto-labeled "
           "(calibrated threshold) with different labels. Accuracy: original → variant.", "",
           "| Approach | Rewording | n | Flip % | Auto flips % | Accuracy % | → question % |", "|---|---|---:|---:|---:|---:|---:|"]
    for a in APPS:
        base = R[(a, "test")]
        for k in KINDS:
            vs = [v for v in var[a][k] if v["applicable"] and v["base_id"] in base]
            if not vs:
                continue
            b = [base[v["base_id"]] for v in vs]
            zb, zv = softmax(logit_matrix(b), calib[a]["T"]), softmax(logit_matrix(vs), calib[a]["T"])
            pb, pv = zb.argmax(1), zv.argmax(1)
            flip = pb != pv
            auto_flip = flip & (zb.max(1) >= calib[a]["tau"]) & (zv.max(1) >= calib[a]["tau"])
            y = label_index(vs)
            q = LABELS.index("question")
            to_q = float(((pv == q) & (pb != q)).mean())
            rob[a][k] = dict(n=len(vs), flip=float(flip.mean()), flip_ci=boot_ci(flip), auto_flip=float(auto_flip.mean()),
                             acc_orig=float((pb == y).mean()), acc_var=float((pv == y).mean()), to_question=to_q)
            md.append(f"| {NAME[a]} | {KNAME[k]} | {len(vs)} | {100 * flip.mean():.1f} | {100 * auto_flip.mean():.1f} | "
                      f"{100 * (pb == y).mean():.1f} → {100 * (pv == y).mean():.1f} | {100 * to_q:.1f} |")
    md.append("")
summary["robustness"] = rob

(out / "results.md").write_text("\n".join(md) + "\n")
(out / "summary.json").write_text(json.dumps(summary, indent=1))

# ---------------- figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
CCOL = dict(zip(CONDS, ["#2a78d6", "#eb6834", "#1baf7a"]))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor=SURF, sharey=True)
for ax, a in zip(axes, [x for x in ("few", "emb") if x in APPS]):
    ax.set_facecolor(SURF)
    for c in CONDS:
        if not R.get((a, c)):
            continue
        rs, p, y = arrays(a, c, calib[a]["T"])
        conf, ok = p.max(1), p.argmax(1) == y
        order = np.argsort(-conf)
        n = np.arange(1, len(order) + 1)
        risk = np.cumsum(~ok[order]) / n
        ax.plot(n / len(n) * 100, risk * 100, color=CCOL[c], lw=2, label=CNAME[c])
        m = sel[a][c]["threshold"]
        ax.scatter([m["coverage"] * 100], [m["risk"] * 100], color=CCOL[c], s=55, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.axhline(TARGET_RISK * 100, color=INK2, lw=1, ls="--")
    ax.text(98, TARGET_RISK * 100 - 3.2, "target risk", color=INK2, fontsize=8.5, ha="right")
    ax.set_title(NAME[a], loc="left", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Issues auto-labeled (%)")
    ax.set_xlim(0, 100)
    ax.grid(True, color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Errors among auto-labeled (%)")
axes[0].set_ylim(0, 45)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=len(CONDS), fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out / "figures" / "risk_coverage.png", dpi=180, facecolor=SURF)
fig.savefig(out / "figures" / "risk_coverage.pdf", facecolor=SURF)

if rob:
    ACOL = dict(zip(APPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    fig, ax = plt.subplots(figsize=(10.5, 3.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    ks = [k for k in KINDS if any(k in rob[a] for a in APPS)]
    w = 0.8 / len(APPS)
    for i, a in enumerate(APPS):
        ax.bar(np.arange(len(ks)) + (i - (len(APPS) - 1) / 2) * w, [100 * rob[a].get(k, {}).get("flip", 0) for k in ks],
               w - 0.02, color=ACOL[a], label=NAME[a], zorder=3)
    ax.set_xticks(np.arange(len(ks)), [KNAME[k] for k in ks])
    ax.set_ylabel("Label flips (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "figures" / "flips.png", dpi=180, facecolor=SURF)
    fig.savefig(out / "figures" / "flips.pdf", facecolor=SURF)
print("wrote", out / "results.md")
