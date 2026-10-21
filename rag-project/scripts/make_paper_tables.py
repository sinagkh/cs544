"""Generate the paper's LaTeX result tables from report/summary.json and the side studies."""
import json
from pathlib import Path

import sys
S = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "report/summary.json").read_text())
OUT = Path("paper/tables")
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"Qwen2.5-7B-Instruct-4bit": "Qwen2.5-7B", "Llama-3.2-3B-Instruct-4bit": "Llama-3.2-3B",
         "gemma-2-9b-it-4bit": "Gemma-2-9B"}
LABEL = {"no_repair": "No repair", "retry": "Retry", "self_check": "Self-check (LLM)",
         "self_check_nli": "Self-check (MiniCheck)", "repair_all": "Repair-all", "checks_only": "Trace checks only",
         "diagnose": "\\textbf{Diagnose (text match)}", "diagnose_llm": "Diagnose (LLM verifier)",
         "diagnose_nli": "Diagnose (MiniCheck)", "diagnose_norecheck": "\\quad $-$ re-check",
         "diagnose_noabstain": "\\quad $-$ abstention", "support_only": "\\quad $-$ trace checks"}
ORDER = list(LABEL)
models = [m for m in SHORT if m in S]


def p(x):
    return "--" if x is None or x != x else f"{100 * x:.1f}"


# Main table: primary models with all policies
prim = [m for m in models if "diagnose_llm" in S[m]]
cols = "l" + "rrrr" * len(prim)
lines = ["\\begin{tabular}{" + cols + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{SHORT[m]}}}" for m in prim) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(prim))),
         "Policy & " + " & ".join(["Faults", "Dmg.", "Abst.", "Tok."] * len(prim)) + " \\\\", "\\midrule"]
for pol in ORDER:
    if not all(pol in S[m] for m in prim):
        continue
    if pol == "diagnose_norecheck":
        lines.append("\\midrule")
    cells = []
    for m in prim:
        v = S[m][pol]
        cells += [p(v["faults_pooled"]), p(v["clean_damage"]), p(v["by_condition"]["absent"]), f"{v['prompt_tokens'] / 1000:.1f}k"]
    lines.append(f"{LABEL[pol]} & " + " & ".join(cells) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "main.tex").write_text("\n".join(lines) + "\n")

# Per-condition success for the proposed policy vs. key baselines, every model
conds = ["clean", "encoder", "truncation", "overflow", "stale", "encoder+truncation", "absent"]
cl = {"clean": "Clean", "encoder": "Enc.", "truncation": "Trunc.", "overflow": "Overfl.", "stale": "Stale",
      "encoder+truncation": "Enc.+Tr.", "absent": "Absent"}
lines = ["\\begin{tabular}{ll" + "r" * len(conds) + "}", "\\toprule",
         "Model & Policy & " + " & ".join(cl[c] for c in conds) + " \\\\", "\\midrule"]
for m in models:
    first = True
    for pol in ("no_repair", "self_check", "repair_all", "checks_only", "diagnose", "diagnose_nli"):
        if pol not in S[m]:
            continue
        v = S[m][pol]["by_condition"]
        lines.append(f"{SHORT[m] if first else ''} & {LABEL[pol]} & " + " & ".join(p(v[c]) for c in conds) + " \\\\")
        first = False
    lines.append("\\midrule" if m != models[-1] else "\\bottomrule")
lines.append("\\end{tabular}")
(OUT / "by_condition.tex").write_text("\n".join(lines) + "\n")

# 2x2 verifier x repair
lines = ["\\begin{tabular}{llrr}", "\\toprule", "Model & Verifier & Regenerate & Repair context \\\\", "\\midrule"]
for m in prim:
    for i, (lab, a, b) in enumerate((("LLM self-verifier", "self_check", "diagnose_llm"),
                                     ("MiniCheck", "self_check_nli", "diagnose_nli"),
                                     ("Text match", None, "diagnose"))):
        left = "--" if a is None else f"{p(S[m][a]['faults_pooled'])} / {p(S[m][a]['clean_damage'])}"
        lines.append(f"{SHORT[m] if i == 0 else ''} & {lab} & {left} & "
                     f"{p(S[m][b]['faults_pooled'])} / {p(S[m][b]['clean_damage'])} \\\\")
    if m != prim[-1]:
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "factorial.tex").write_text("\n".join(lines) + "\n")

# Support checks
lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Check & Good kept & Bad caught & Bal.\\ acc. \\\\", "\\midrule"]
for m in prim:
    e = S[m].get("_support_eval")
    if not e:
        continue
    lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{SHORT[m]}: {e['n']} initial answers, {e['positives']} correct and grounded}}}} \\\\")
    for lab, t, n in (("LLM self-verifier", e["llm_tpr"], e["llm_tnr"]),
                      (f"MiniCheck $\\geq 0.5$ (AUROC {e['auroc']:.2f})", e["nli_tpr"], e["nli_tnr"])):
        lines.append(f"{lab} & {p(t)} & {p(n)} & {(t + n) / 2:.2f} \\\\")
vs = Path("report/verifier_study.json")
if vs.exists():
    v = json.loads(vs.read_text())
    nice = ["Llama-3.2-3B self-verifier", "Claude Haiku 4.5, same prompt", "MiniCheck $\\geq 0.5$"]
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\emph{Same 240 answers (40 per condition) for all three}} \\\\")
    for lab, d in zip(nice, v.values()):
        lines.append(f"{lab} & {p(d['good_kept'])} & {p(d['bad_caught'])} & {(d['good_kept'] + d['bad_caught']) / 2:.2f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "support.tex").write_text("\n".join(lines) + "\n")

# Monitoring
for f in Path("report").glob("monitoring_*.json"):
    mon = json.loads(f.read_text())["summary"]
    sigs = ["trace checks", "MiniCheck", "LLM verifier", "abstention"]
    lines = ["\\begin{tabular}{l" + "r" * len(sigs) + "}", "\\toprule", "Fault & " + " & ".join({"trace checks": "Trace", "LLM verifier": "LLM ver.", "abstention": "Abstain"}.get(s, s) for s in sigs) + " \\\\", "\\midrule"]
    for fault, d in mon.items():
        cells = []
        for s in sigs:
            x = d[s]
            cells.append("--" if x["median_delay"] is None or x["detected"] < 0.5 else f"{x['median_delay']:.0f} ({100 * x['detected']:.0f}\\%)")
        lines.append(f"{fault.replace('encoder+truncation', 'enc.+trunc.')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{f.stem}.tex").write_text("\n".join(lines) + "\n")
print("tables:", sorted(x.name for x in OUT.iterdir()))
