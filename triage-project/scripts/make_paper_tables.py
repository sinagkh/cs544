"""LaTeX tables for the triage paper, generated from report/*.json."""
import json
from pathlib import Path

S = json.loads(Path("report/summary.json").read_text())
M = json.loads(Path("report/monitoring.json").read_text())
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)
APPS = ["tfidf", "emb", "zero", "few"]
NAME = {"tfidf": "TF-IDF + LR", "emb": "Embedding + LR", "zero": "LLM zero-shot", "few": "LLM few-shot"}
CONDS = ["test", "fresh", "newrepo"]


def p(x):
    return "--" if x is None or x != x else f"{100 * x:.1f}"


# Accuracy
L = ["\\begin{tabular}{lrrrr}", "\\toprule", "Approach & NLBSE'24 & 2025--26 & New repo & Tokens \\\\", "\\midrule"]
for a in APPS:
    A = S["accuracy"][a]
    cells = [f"{p(A[c]['acc'])}" for c in CONDS]
    L.append(f"{NAME[a]} & " + " & ".join(cells) + f" & {A['test']['prompt_tokens'] / 1000:.1f}k \\\\" if a in ("zero", "few")
             else f"{NAME[a]} & " + " & ".join(cells) + " & -- \\\\")
L += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Cross-repository F1 (NLBSE ranking metric)}} \\\\"]
for a in APPS:
    A = S["accuracy"][a]
    L.append(f"{NAME[a]} & " + " & ".join(f"{A[c]['xrepo_f1']:.3f}" for c in CONDS) + " & \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "accuracy.tex").write_text("\n".join(L) + "\n")

# Selective automation
POL = [("auto-all", "auto-all"), ("threshold", "naive threshold"), ("risk-controlled", "risk-controlled"), ("conformal", "conformal")]
L = ["\\begin{tabular}{llrrr}", "\\toprule", "Approach & Policy & NLBSE'24 & 2025--26 & New repo \\\\", "\\midrule"]
for a in APPS:
    for i, (k, lab) in enumerate(POL):
        cells = []
        for c in CONDS:
            m = S["selective"][a][c][k]
            cells.append(f"{p(m['coverage'])} / {p(m['risk'])}" if m["coverage"] > 0 else "0.0 / --")
        L.append(f"{NAME[a] if i == 0 else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
ag = S["selective"]["few"]
L.append("LLM few-shot + Emb. & agreement & " + " & ".join(f"{p(ag[c]['agreement']['coverage'])} / {p(ag[c]['agreement']['risk'])}" for c in CONDS) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "selective.tex").write_text("\n".join(L) + "\n")

# Robustness
KINDS = ["no_code", "no_template", "newcomer", "non_native", "non_native+no_code"]
KN = {"no_code": "No code", "no_template": "No templ.", "newcomer": "Newcomer", "non_native": "Non-native", "non_native+no_code": "NN + no code"}
L = ["\\begin{tabular}{l" + "r" * len(KINDS) + "}", "\\toprule", "Approach & " + " & ".join(KN[k] for k in KINDS) + " \\\\",
     "\\midrule", f"\\emph{{n}} & " + " & ".join(str(S["robustness"]["few"][k]["n"]) for k in KINDS) + " \\\\"]
for a in APPS:
    L.append(f"{NAME[a]} & " + " & ".join(p(S["robustness"][a][k]["flip"]) for k in KINDS) + " \\\\")
L += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Accuracy change, original $\\to$ variant (points)}} \\\\"]
for a in APPS:
    R = S["robustness"][a]
    L.append(f"{NAME[a]} & " + " & ".join(f"{100 * (R[k]['acc_var'] - R[k]['acc_orig']):+.1f}" for k in KINDS) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "robustness.tex").write_text("\n".join(L) + "\n")

# Monitoring
SIG = ["deferred", "low_conf", "disagree", "novelty", "label_shift"]
SN = {"deferred": "Handoff", "low_conf": "Low conf.", "disagree": "Disagree", "novelty": "Novelty", "label_shift": "Label shift"}
SH = {"fresh": "2025--26 issues", "newrepo": "New repository", "non_native": "Non-native"}
L = ["\\begin{tabular}{l" + "r" * len(SIG) + "}", "\\toprule", "Shift & " + " & ".join(SN[s] for s in SIG) + " \\\\", "\\midrule"]
for sh, d in M.items():
    L.append(f"{SH[sh]} & " + " & ".join("--" if d[s]["detected"] < 0.5 else f"{d[s]['median_delay']:.0f} ({100 * d[s]['detected']:.0f}\\%)" for s in SIG) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "monitoring.tex").write_text("\n".join(L) + "\n")
print(sorted(x.name for x in OUT.iterdir()))
