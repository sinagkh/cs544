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
SCOLS = ["test", "fresh", "newrepo", "newrepo_loro"]


def p(x):
    return "--" if x is None or x != x else f"{100 * x:.1f}"


def cell(m, counts=False):
    if not m["n_auto"]:
        return "0.0 / --"
    s = f"{p(m['coverage'])} / {p(m['risk'])}"
    return s + (f" \\scriptsize({m['errors']}/{m['n_auto']})" if counts else "")


# Accuracy
L = ["\\begin{tabular}{lrrrr}", "\\toprule", "Approach & NLBSE'24 & 2025--26 & New repo & Tokens \\\\", "\\midrule"]
for a in APPS:
    A = S["accuracy"][a]
    tok = f"{A['test']['prompt_tokens'] / 1000:.1f}k" if a in ("zero", "few") else "--"
    L.append(f"{NAME[a]} & " + " & ".join(p(A[c]["acc"]) for c in CONDS) + f" & {tok} \\\\")
L += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Cross-repository F1 (NLBSE ranking metric)}} \\\\"]
for a in APPS:
    A = S["accuracy"][a]
    L.append(f"{NAME[a]} & " + " & ".join(f"{A[c]['xrepo_f1']:.3f}" for c in CONDS) + " & \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "accuracy.tex").write_text("\n".join(L) + "\n")

# Selective automation
POL = [("auto-all", "auto-all"), ("naive threshold", "naive threshold"), ("risk-controlled", "risk-controlled"),
       ("conformal", "conformal")]
L = ["\\begin{tabular}{llrrrr}", "\\toprule",
     "Approach & Policy & NLBSE'24 & 2025--26 & New repo (reused) & New repo (LORO) \\\\", "\\midrule"]
for a in APPS:
    for i, (k, lab) in enumerate(POL):
        L.append(f"{NAME[a] if i == 0 else ''} & {lab} & " +
                 " & ".join(cell(S["selective"][a][c][k], counts=k == "risk-controlled") for c in SCOLS) + " \\\\")
    L.append("\\midrule")
ag = S["selective"]["few"]
L.append("LLM few-shot + Emb. & agreement & " + " & ".join(f"{p(ag[c]['agreement']['coverage'])} / {p(ag[c]['agreement']['risk'])}"
                                                           for c in CONDS) + " & -- \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "selective.tex").write_text("\n".join(L) + "\n")

# Risk-control frontier
L = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Approach & Target & NLBSE'24 & 2025--26 & New (reused) & New (LORO) \\\\", "\\midrule"]
for a in APPS:
    for i, t in enumerate(("0.1", "0.15", "0.2", "0.25")):
        L.append(f"{NAME[a] if i == 0 else ''} & {int(round(float(t) * 100))}\\% & " +
                 " & ".join(cell(S["frontier"][a][t][c]) for c in SCOLS) + " \\\\")
    if a != APPS[-1]:
        L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "frontier.tex").write_text("\n".join(L) + "\n")

# Robustness (main rewordings) with workload
KINDS = ["no_code", "no_template", "newcomer", "non_native", "non_native+no_code"]
KN = {"no_code": "No code$^\\dagger$", "no_template": "No templ.", "newcomer": "Newcomer", "non_native": "Non-native",
      "non_native+no_code": "NN + no code$^\\dagger$"}
R = S["robustness"]
L = ["\\begin{tabular}{l" + "r" * len(KINDS) + "}", "\\toprule", " & " + " & ".join(KN[k] for k in KINDS) + " \\\\",
     "\\midrule", "\\emph{n} & " + " & ".join(str(R["few"][k]["n"]) for k in KINDS) + " \\\\",
     "\\multicolumn{6}{l}{\\emph{Label flips (\\%)}} \\\\"]
for a in APPS:
    L.append(f"{NAME[a]} & " + " & ".join(p(R[a][k]["flip"]) for k in KINDS) + " \\\\")
L.append("\\multicolumn{6}{l}{\\emph{Accuracy change, original $\\to$ variant (points)}} \\\\")
for a in APPS:
    L.append(f"{NAME[a]} & " + " & ".join(f"{100 * (R[a][k]['acc_var'] - R[a][k]['acc_orig']):+.1f}" for k in KINDS) + " \\\\")
L.append("\\multicolumn{6}{l}{\\emph{Auto-labeled (\\%), original $\\to$ variant}} \\\\")
for a in APPS:
    L.append(f"{NAME[a]} & " + " & ".join(f"{100 * R[a][k]['auto_orig']:.0f}$\\to${100 * R[a][k]['auto_var']:.0f}" for k in KINDS) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "robustness.tex").write_text("\n".join(L) + "\n")

# Cue controls
C = S.get("cue_controls", {})
CK = ["newcomer", "cue_newcomer_only", "cue_apology_only", "cue_question_word", "cue_neutral"]
CN = {"newcomer": "Full newcomer sentence", "cue_newcomer_only": "Newcomer status only", "cue_apology_only": "Apology only",
      "cue_question_word": "``Silly question'' only", "cue_neutral": "Neutral sentence"}
if C:
    L = ["\\begin{tabular}{llrrrr}", "\\toprule", "Approach & Sentence & Flips & $\\to$question & $\\Delta$ vs.\\ neutral & Auto-lab. \\\\", "\\midrule"]
    for a in ("zero", "few", "tfidf"):
        if a not in C:
            continue
        for i, k in enumerate(CK):
            c = C[a][k]
            d = "--" if k == "cue_neutral" else f"{100 * c['d_q']:+.1f} [{100 * c['d_q_ci'][0]:+.1f}, {100 * c['d_q_ci'][1]:+.1f}]"
            L.append(f"{NAME[a] if i == 0 else ''} & {CN[k]} & {p(c['flip'])} & {p(c['to_question'])} & {d} & {p(c['auto'])} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    (OUT / "cues.tex").write_text("\n".join(L) + "\n")

# Monitoring
SIG = ["deferred", "low_conf", "disagree", "novelty", "label_shift"]
SN = {"deferred": "Handoff", "low_conf": "Low conf.", "disagree": "Disagree", "novelty": "Novelty", "label_shift": "Label shift"}
SH = {"fresh": "2025--26 issues", "newrepo": "New repository", "non_native": "Non-native"}
L = ["\\begin{tabular}{l" + "r" * len(SIG) + "}", "\\toprule", "Shift & " + " & ".join(SN[s] for s in SIG) + " \\\\", "\\midrule"]
for sh, d in M.items():
    L.append(f"{SH[sh]} & " + " & ".join("--" if d[s]["detected"] < 0.5 else f"{d[s]['median_delay']:.0f} ({100 * d[s]['detected']:.0f}\\%)"
                                         for s in SIG) + " \\\\")
L.append("\\midrule")
L.append("\\emph{False alarms} & " + " & ".join(f"{100 * max(d[s]['false_alarm'] for d in M.values()):.1f}\\%" for s in SIG) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(OUT / "monitoring.tex").write_text("\n".join(L) + "\n")
print(sorted(x.name for x in OUT.iterdir()))
