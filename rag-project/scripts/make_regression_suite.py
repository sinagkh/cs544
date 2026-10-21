"""Turn observed failures into a regression suite for the proposed policy.

Selects, from one model's test run, clean questions the pipeline answers correctly (they must stay
correct) and, for each fault condition, failures of the deployed pipeline that `diagnose_nli`
repaired. Uses the record-time (string-match) scores, which the replay test also uses.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--policy", default="diagnose")
ap.add_argument("--repo", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
ap.add_argument("--out", default="tests/regression_cases.json")
ap.add_argument("--per-condition", type=int, default=5)
args = ap.parse_args()

by = defaultdict(dict)
for line in Path(args.results).read_text().splitlines():
    r = json.loads(line)
    by[(r["qid"], r["condition"])][r["policy"]] = r

keep, fixed = [], defaultdict(list)
for (qid, cond), pols in sorted(by.items()):
    d, base = pols.get(args.policy), pols.get("no_repair")
    if not d or not base:
        continue
    if cond == "clean" and base["success"] and d["success"]:
        keep.append(dict(qid=qid, condition=cond, kind="must_not_break", gold=d["gold"], answer=d["final_answer"]))
    elif cond != "clean" and not base["success"] and d["success"]:
        fixed[cond].append(dict(qid=qid, condition=cond, kind="fixed_failure", gold=d["gold"],
                                broken_answer=base["final_answer"], repaired_answer=d["final_answer"],
                                actions=d["actions"]))
cases = keep[: args.per_condition] + [c for cond in fixed for c in fixed[cond][: args.per_condition]]
Path(args.out).write_text(json.dumps(dict(model=args.repo, policy=args.policy, cases=cases), indent=1))
print(f"{len(cases)} regression cases ({', '.join(f'{c}: {len(v[: args.per_condition])}' for c, v in fixed.items())}) -> {args.out}")
