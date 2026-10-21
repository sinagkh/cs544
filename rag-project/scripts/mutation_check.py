"""Does the regression suite catch regressions? Disable one diagnostic at a time and replay it live.

Each mutation removes one trace check (or the re-check after repair) from the proposed policy. The
suite is replayed on the local model (new prompts are generated and cached); a case fails if the
policy no longer succeeds. A useful suite fails under every mutation that matters.
"""
import json
from collections import Counter
from pathlib import Path

import ragrepair.policies as P
from ragrepair.data import load_dataset
from ragrepair.evaluate import score_answer
from ragrepair.llm import MLXModel, ResponseCache
from ragrepair.store import Store
from ragrepair.support import SupportChecker
from ragrepair.system import CaseOps, initial_run

suite = json.loads(Path("tests/regression_cases.json").read_text())
ds = load_dataset("data/dataset.json")
cases = {q["id"]: q for q in ds["dev"] + ds["test"]}
store = Store(ds)
model = MLXModel(suite["model"], ResponseCache("cache/llm_cache.sqlite"))
checker = SupportChecker()
orig_flags, orig_diag = P.free_flags, P.diagnose


def without(flag):
    return lambda ops, init: [f for f in orig_flags(ops, init) if f != flag]


MUTATIONS = {
    "none (original)": {},
    "no truncation check": {"free_flags": without("truncation")},
    "no version check": {"free_flags": without("version_conflict")},
    "no overflow check": {"free_flags": without("overflow")},
    "no low-retrieval check": {"free_flags": without("low_retrieval")},
    "no re-check after repair": {"recheck": False},
}
rows = []
for name, mut in MUTATIONS.items():
    P.free_flags = mut.get("free_flags", orig_flags)
    fails = Counter()
    for c in suite["cases"]:
        case = cases[c["qid"]]
        ops = CaseOps(case, c["condition"], store, model, checker)
        res = orig_diag(ops, initial_run(ops), support="lexical", recheck=mut.get("recheck", True))
        if not score_answer(res["answer"], res["rendered"], case, c["condition"])["success"]:
            fails[c["condition"]] += 1
    rows.append((name, sum(fails.values()), dict(fails)))
    print(name, sum(fails.values()), dict(fails), flush=True)
P.free_flags = orig_flags
total = len(suite["cases"])
md = [f"# Mutation check of the regression suite ({total} cases, {suite['model'].split('/')[-1]})", "",
      "| Mutation | Failing cases | By condition |", "|---|---:|---|"]
md += [f"| {n} | {k} / {total} | {', '.join(f'{c}: {v}' for c, v in f.items()) or '–'} |" for n, k, f in rows]
Path("report/mutation_check.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
