"""Tables and figures for the triage study.

Calibration protocol: each classifier's calibration rows are split (balanced, deterministic) into a
fit part (one third: temperature and candidate threshold grid) and a certification part (two
thirds: naive threshold, risk-controlled threshold, conformal quantile). For the new-repository
condition two protocols are reported: reusing the in-distribution calibration (what a service
does when it onboards a repository without recalibrating) and leave-one-repository-out (LORO)
calibration on the calibration issues of the other four repositories, scored by the same
cross-repository pipeline.
"""
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
                              logit_matrix, ltt_fixed_sequence, macro_f1, risk_threshold, selective_metrics, softmax,
                              split_fit_cert, threshold_grid)

ap = argparse.ArgumentParser()
ap.add_argument("--llm", default="results/llm_Qwen2.5-7B-Instruct-4bit.jsonl")
ap.add_argument("--classical", default="results/classical.jsonl")
ap.add_argument("--out", default="report")
args = ap.parse_args()

rows = [json.loads(l) for f in (args.llm, args.classical) if Path(f).exists() for l in Path(f).read_text().splitlines()]
R = defaultdict(dict)
for r in rows:
    R[(r["approach"], r["condition"])][r["id"]] = r
# Zero-shot prompts do not depend on the history, so its cross-repository rows equal its normal rows.
R[("zero", "calib_xrepo")] = dict(R[("zero", "calib")])
newrepo_ids = set(R[("few", "newrepo")])
R[("zero", "newrepo")] = {i: r for i, r in R[("zero", "test")].items() if i in newrepo_ids}

APPS = [a for a in ("tfidf", "emb", "zero", "few") if (a, "test") in R]
NAME = {"tfidf": "TF-IDF + LR", "emb": "Embedding + LR", "zero": "LLM zero-shot", "few": "LLM few-shot (retrieval)"}
CONDS = ["test", "fresh", "newrepo"]
CNAME = {"test": "NLBSE'24 test", "fresh": "Fresh 2025-26", "newrepo": "New repository",
         "newrepo_loro": "New repo (LORO calib.)"}
out = Path(args.out)
(out / "figures").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_sel(m):
    risk = "–" if not m["n_auto"] else f"{100 * m['risk']:.1f}"
    return f"{100 * m['coverage']:.1f} / {risk}"


def cal_params(cal_rows):
    fit, cert = split_fit_cert(cal_rows)
    zf, yf = logit_matrix(fit), label_index(fit)
    T = fit_temperature(zf, yf)
    grid = threshold_grid(softmax(zf, T).max(1))
    pc, yc = softmax(logit_matrix(cert), T), label_index(cert)
    conf, correct = pc.max(1), pc.argmax(1) == yc
    return dict(T=T, grid=grid, n_fit=len(fit), n_cert=len(cert), acc=float(correct.mean()),
                tau_naive=risk_threshold(conf, correct),
                tau_ltt={str(t): ltt_fixed_sequence(conf, correct, grid, t) for t in (0.10, 0.15, 0.20, 0.25)},
                qhat=conformal_qhat(pc, yc), ece_raw=ece(softmax(logit_matrix(cert)), yc), ece_cal=ece(pc, yc))


calib = {a: cal_params(list(R[(a, "calib")].values())) for a in APPS}
for a in APPS:
    if len(R[(a, "calib_xrepo")]) < len(R[(a, "calib")]):
        print(f"WARNING: {a}: cross-repository calibration rows incomplete; using in-distribution rows")
        R[(a, "calib_xrepo")] = dict(R[(a, "calib")])
loro = {a: {rp: cal_params([x for x in R[(a, "calib_xrepo")].values() if x["repo"] != rp]) for rp in REPOS}
        for a in APPS}


def scored(a, cond, protocol="reuse"):
    rs = list(R[(a, cond)].values())
    params = [loro[a][r["repo"]] if protocol == "loro" else calib[a] for r in rs]
    p = np.vstack([softmax(logit_matrix([r]), q["T"]) for r, q in zip(rs, params)])
    return rs, p, label_index(rs), params


def policy_auto(pol, p, params, target="0.1"):
    conf = p.max(1)
    if pol == "auto-all":
        return np.ones(len(p), bool), None
    if pol == "naive threshold":
        return conf >= np.array([q["tau_naive"] for q in params]), None
    if pol == "risk-controlled":
        return conf >= np.array([q["tau_ltt"][target] for q in params]), None
    S = np.vstack([conformal_sets(p[i:i + 1], q["qhat"]) for i, q in enumerate(params)])
    return S.sum(1) == 1, S


summary = dict(calibration={a: {k: v for k, v in calib[a].items() if k != "grid"} for a in APPS})
md = ["# Triage results", "", "## Calibration", "",
      "Calibration rows are split into a fit part (temperature, threshold grid) and a certification part "
      "(thresholds, conformal quantile). Risk-controlled thresholds: Learn-then-Test with fixed-sequence testing over "
      "the pre-specified grid and exact binomial tests at δ = 0.05.", "",
      "| Approach | n fit / cert | Accuracy (cert) | Temperature | ECE raw → calibrated (cert) | Naive threshold | "
      "Risk-controlled (10%) | Conformal q̂ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for a in APPS:
    c = calib[a]
    md.append(f"| {NAME[a]} | {c['n_fit']} / {c['n_cert']} | {100 * c['acc']:.1f} | {c['T']:.2f} | "
              f"{c['ece_raw']:.3f} → {c['ece_cal']:.3f} | {c['tau_naive']:.3f} | {c['tau_ltt']['0.1']:.3f} | {c['qhat']:.3f} |")
md.append("")

# ---------------- accuracy
md += ["## Accuracy", "", "Cross-repo F1 = mean of per-repository macro-F1 (the NLBSE'24 ranking metric).", "",
       "| Approach | Condition | n | Accuracy [95% CI] | Macro-F1 | Cross-repo F1 | ECE | Prompt tokens | s / issue |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
acc = defaultdict(dict)
for a in APPS:
    for c in CONDS:
        rs, p, y, _ = scored(a, c)
        pred = p.argmax(1)
        ok = pred == y
        lo, hi = boot_ci(ok)
        repo = np.array([r["repo"] for r in rs])
        xf1 = float(np.mean([macro_f1(y[repo == rp], pred[repo == rp]) for rp in REPOS if (repo == rp).any()]))
        tok = float(np.mean([r.get("prompt_tokens", 0) for r in rs]))
        sec = float(np.mean([r.get("seconds", 0) for r in rs]))
        acc[a][c] = dict(n=len(rs), acc=float(ok.mean()), ci=[lo, hi], macro_f1=macro_f1(y, pred), xrepo_f1=xf1,
                         ece=ece(p, y), prompt_tokens=tok, seconds=sec)
        md.append(f"| {NAME[a]} | {CNAME[c]} | {len(rs)} | {100 * ok.mean():.1f} [{100 * lo:.1f}, {100 * hi:.1f}] | "
                  f"{macro_f1(y, pred):.3f} | {xf1:.3f} | {ece(p, y):.3f} | {tok:.0f} | {sec:.2f} |")
fresh_labels = np.bincount(label_index(list(R[("few", "fresh")].values())), minlength=3)
md += ["", "Fresh 2025-26 sample label mix: " + ", ".join(f"{l} {n}" for l, n in zip(LABELS, fresh_labels))
       + " (the benchmark is balanced).", ""]
summary["accuracy"] = acc
summary["fresh_label_mix"] = dict(zip(LABELS, map(int, fresh_labels)))

# ---------------- selective automation
SCOLS = ["test", "fresh", "newrepo", "newrepo_loro"]
md += ["## Selective automation (target: at most 10% errors among auto-labeled issues)", "",
       "Cells: coverage % / risk % (n auto-labeled, errors). 'New repository' reuses the in-distribution calibration; "
       "'LORO calib.' recalibrates on the other repositories' calibration issues scored cross-repository.", "",
       "| Approach | Policy | " + " | ".join(CNAME[c] for c in SCOLS) + " |", "|---|---|" + "---:|" * len(SCOLS)]
sel = defaultdict(dict)
for a in APPS:
    for pol in ("auto-all", "naive threshold", "risk-controlled", "conformal"):
        cells = []
        for col in SCOLS:
            cond, prot = ("newrepo", "loro") if col == "newrepo_loro" else (col, "reuse")
            rs, p, y, params = scored(a, cond, prot)
            correct = p.argmax(1) == y
            auto, S = policy_auto(pol, p, params)
            if S is not None:
                sel[a].setdefault(col, {})["conformal_set_coverage"] = float(S[np.arange(len(y)), y].mean())
            m = selective_metrics(auto, correct)
            m.update(n_auto=int(auto.sum()), errors=int((auto & ~correct).sum()))
            sel[a].setdefault(col, {})[pol] = m
            cells.append(f"{fmt_sel(m)} ({m['n_auto']}, {m['errors']})")
        md.append(f"| {NAME[a]} | {pol} | " + " | ".join(cells) + " |")
cells = []
for col in CONDS:
    f, e = R[("few", col)], R[("emb", col)]
    ids = [i for i in f if i in e]
    auto = np.array([f[i]["pred"] == e[i]["pred"] for i in ids])
    correct = np.array([f[i]["pred"] == f[i]["label"] for i in ids])
    m = selective_metrics(auto, correct)
    m.update(n_auto=int(auto.sum()), errors=int((auto & ~correct).sum()))
    sel["few"].setdefault(col, {})["agreement"] = m
    cells.append(fmt_sel(m))
md.append("| LLM few-shot + Embedding | agreement | " + " | ".join(cells) + " | – |")
md += ["", "Conformal prediction-set coverage (target ≥ 90%): " + "; ".join(
    f"{NAME[a]}: " + ", ".join(f"{CNAME[c]} {100 * sel[a][c]['conformal_set_coverage']:.1f}%" for c in SCOLS)
    for a in APPS), ""]

md += ["### Risk-control frontier", "",
       "Coverage % / actual risk % when the risk-controlled threshold is certified for each target (δ = 0.05).", "",
       "| Approach | Target | " + " | ".join(CNAME[c] for c in SCOLS) + " |", "|---|---:|" + "---:|" * len(SCOLS)]
frontier = defaultdict(dict)
for a in APPS:
    for t in ("0.1", "0.15", "0.2", "0.25"):
        cells = []
        for col in SCOLS:
            cond, prot = ("newrepo", "loro") if col == "newrepo_loro" else (col, "reuse")
            rs, p, y, params = scored(a, cond, prot)
            auto, _ = policy_auto("risk-controlled", p, params, t)
            correct = p.argmax(1) == y
            m = selective_metrics(auto, correct)
            m.update(n_auto=int(auto.sum()), errors=int((auto & ~correct).sum()))
            frontier[a].setdefault(t, {})[col] = m
            cells.append(fmt_sel(m))
        md.append(f"| {NAME[a]} | {int(round(float(t) * 100))}% | " + " | ".join(cells) + " |")
md.append("")
summary["selective"] = sel
summary["frontier"] = frontier

# ---------------- rewording
var = defaultdict(lambda: defaultdict(list))
for a in APPS:
    for r in R.get((a, "variants"), {}).values():
        var[a][r["kind"]].append(r)
KINDS = ["no_code", "no_template", "newcomer", "non_native", "non_native+no_code"]
CUEK = ["newcomer", "cue_newcomer_only", "cue_apology_only", "cue_question_word", "cue_neutral"]
KNAME = {"no_code": "no code / stack trace", "no_template": "no template", "newcomer": "newcomer sentence (full)",
         "non_native": "non-native English", "non_native+no_code": "non-native + no code",
         "cue_newcomer_only": "newcomer status only", "cue_apology_only": "apology only",
         "cue_question_word": "'silly question' only", "cue_neutral": "neutral sentence"}


def rob_stats(a, k):
    base = R[(a, "test")]
    vs = [v for v in var[a][k] if v["applicable"] and v["base_id"] in base]
    if not vs:
        return None
    b = [base[v["base_id"]] for v in vs]
    T, tau = calib[a]["T"], calib[a]["tau_naive"]
    zb, zv = softmax(logit_matrix(b), T), softmax(logit_matrix(vs), T)
    pb, pv = zb.argmax(1), zv.argmax(1)
    y = label_index(vs)
    q = LABELS.index("question")
    flip = pb != pv
    ab, av = zb.max(1) >= tau, zv.max(1) >= tau
    to_q = (pv == q) & (pb != q)
    return dict(n=len(vs), flip=float(flip.mean()), flip_ci=boot_ci(flip), auto_flip=float((flip & ab & av).mean()),
                acc_orig=float((pb == y).mean()), acc_var=float((pv == y).mean()), to_question=float(to_q.mean()),
                auto_orig=float(ab.mean()), auto_var=float(av.mean()), auto_drop_ci=boot_ci(ab.astype(float) - av),
                _flip=flip, _to_q=to_q, _ids=[v["base_id"] for v in vs])


rob = defaultdict(dict)
md += ["## Robustness to rewording", "",
       "Flip = label differs from the original's. Auto flips = both versions auto-labeled (naive threshold) with "
       "different labels. Auto-labeled = share auto-labeled under the naive threshold, original → variant (the rest "
       "goes to a maintainer). 'No code' removes information and is reported as reduced input, not as an invariance test.",
       "", "| Approach | Rewording | n | Flip % [95% CI] | Auto flips % | Accuracy % | → question % | Auto-labeled % |",
       "|---|---|---:|---:|---:|---:|---:|---:|"]
for a in APPS:
    for k in KINDS + CUEK[1:]:
        st = rob_stats(a, k)
        if not st:
            continue
        rob[a][k] = {kk: v for kk, v in st.items() if not kk.startswith("_")}
        md.append(f"| {NAME[a]} | {KNAME[k]} | {st['n']} | {100 * st['flip']:.1f} [{100 * st['flip_ci'][0]:.1f}, "
                  f"{100 * st['flip_ci'][1]:.1f}] | {100 * st['auto_flip']:.1f} | {100 * st['acc_orig']:.1f} → "
                  f"{100 * st['acc_var']:.1f} | {100 * st['to_question']:.1f} | {100 * st['auto_orig']:.1f} → "
                  f"{100 * st['auto_var']:.1f} |")
md.append("")

md += ["### Which part of the newcomer sentence matters?", "",
       "Paired by issue: difference from the neutral sentence, in percentage points [95% bootstrap CI].", "",
       "| Approach | Sentence | Flip % | Δ flip vs neutral | → question % | Δ → question vs neutral | Auto-labeled % |",
       "|---|---|---:|---:|---:|---:|---:|"]
cue = defaultdict(dict)
for a in APPS:
    neu = rob_stats(a, "cue_neutral")
    if not neu:
        continue
    idx = {b: i for i, b in enumerate(neu["_ids"])}
    for k in CUEK:
        st = rob_stats(a, k)
        if not st:
            continue
        j = [idx[b] for b in st["_ids"]]
        df = st["_flip"].astype(float) - neu["_flip"][j].astype(float)
        dq = st["_to_q"].astype(float) - neu["_to_q"][j].astype(float)
        cf, cq = boot_ci(df), boot_ci(dq)
        cue[a][k] = dict(flip=st["flip"], d_flip=float(df.mean()), d_flip_ci=cf, to_question=st["to_question"],
                         d_q=float(dq.mean()), d_q_ci=cq, auto=st["auto_var"], auto_orig=st["auto_orig"])
        md.append(f"| {NAME[a]} | {KNAME[k]} | {100 * st['flip']:.1f} | {100 * df.mean():+.1f} [{100 * cf[0]:+.1f}, "
                  f"{100 * cf[1]:+.1f}] | {100 * st['to_question']:.1f} | {100 * dq.mean():+.1f} [{100 * cq[0]:+.1f}, "
                  f"{100 * cq[1]:+.1f}] | {100 * st['auto_var']:.1f} |")
md.append("")
summary["robustness"] = rob
summary["cue_controls"] = cue

(out / "results.md").write_text("\n".join(md) + "\n")
(out / "summary.json").write_text(json.dumps(summary, indent=1))

# ---------------- figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
CCOL = {"test": "#2a78d6", "fresh": "#eb6834", "newrepo": "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor=SURF, sharey=True)
for ax, a in zip(axes, ["few", "tfidf"]):
    ax.set_facecolor(SURF)
    for c in CONDS:
        rs, p, y, _ = scored(a, c)
        conf, ok = p.max(1), p.argmax(1) == y
        order = np.argsort(-conf)
        n = np.arange(1, len(order) + 1)
        ax.plot(n / len(n) * 100, np.cumsum(~ok[order]) / n * 100, color=CCOL[c], lw=2, label=CNAME[c])
        m = sel[a][c]["naive threshold"]
        ax.scatter([m["coverage"] * 100], [m["risk"] * 100], color=CCOL[c], s=55, edgecolor=SURF, linewidth=1.5, zorder=4)
        m = sel[a][c]["risk-controlled"]
        if m["n_auto"]:
            ax.scatter([m["coverage"] * 100], [m["risk"] * 100], color=SURF, edgecolor=CCOL[c], s=55, linewidth=2, zorder=4)
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
fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=9, labelcolor=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out / "figures" / "risk_coverage.png", dpi=180, facecolor=SURF)
fig.savefig(out / "figures" / "risk_coverage.pdf", facecolor=SURF)

ACOL = dict(zip(APPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
ks = [k for k in KINDS + CUEK[1:] if any(k in rob[a] for a in APPS)]
TICK = {"no_code": "no code /\nstack trace", "no_template": "no\ntemplate", "newcomer": "newcomer\nsentence",
        "non_native": "non-native\nEnglish", "non_native+no_code": "non-native\n+ no code",
        "cue_newcomer_only": "newcomer\nstatus only", "cue_apology_only": "apology\nonly",
        "cue_question_word": "'silly question'\nonly", "cue_neutral": "neutral\nsentence"}
fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURF)
ax.set_facecolor(SURF)
w = 0.8 / len(APPS)
for i, a in enumerate(APPS):
    ax.bar(np.arange(len(ks)) + (i - (len(APPS) - 1) / 2) * w, [100 * rob[a].get(k, {}).get("flip", 0) for k in ks],
           w - 0.02, color=ACOL[a], label=NAME[a], zorder=3)
ax.set_xticks(np.arange(len(ks)), [TICK[k] for k in ks], fontsize=8.5)
ax.axvline(len(KINDS) - 0.5, color=GRID, lw=1)
ax.set_ylabel("Label flips (%)")
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left", labelcolor=INK)
fig.tight_layout()
fig.savefig(out / "figures" / "flips.png", dpi=180, facecolor=SURF)
fig.savefig(out / "figures" / "flips.pdf", facecolor=SURF)
print("wrote", out / "results.md")
