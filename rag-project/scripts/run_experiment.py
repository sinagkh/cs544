"""Run every (question, condition, policy) case for one model and append records to a JSONL file."""
import argparse
import json
import time
from pathlib import Path

from ragrepair.data import load_dataset
from ragrepair.evaluate import record
from ragrepair.faults import CONDITIONS
from ragrepair.llm import ClaudeCLIModel, MLXModel, ResponseCache
from ragrepair.policies import POLICIES
from ragrepair.store import Store
from ragrepair.support import SupportChecker
from ragrepair.system import CaseOps, initial_run

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="data/dataset.json")
ap.add_argument("--split", default="dev", choices=["dev", "test"])
ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
ap.add_argument("--backend", default="mlx", choices=["mlx", "claude"])
ap.add_argument("--conditions", default=",".join(CONDITIONS))
ap.add_argument("--policies", default=",".join(POLICIES))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="")
ap.add_argument("--cache", default="cache/llm_cache.sqlite")
args = ap.parse_args()

ds = load_dataset(args.dataset)
cases = ds[args.split][: args.limit or None]
store = Store(ds)
cache = ResponseCache(args.cache)
model = MLXModel(args.model, cache) if args.backend == "mlx" else ClaudeCLIModel(args.model, cache)
checker = SupportChecker()
short = args.model.split("/")[-1]
out = Path(args.out or f"results/{short}_{args.split}.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        r = json.loads(line)
        done.add((r["qid"], r["condition"], r["policy"]))

conditions, policies = args.conditions.split(","), args.policies.split(",")
total = len(cases) * len(conditions) * len(policies)
t0, n = time.time(), 0
with out.open("a") as f:
    for case in cases:
        for cond in conditions:
            for pol in policies:
                n += 1
                if (case["id"], cond, pol) in done:
                    continue
                ops = CaseOps(case, cond, store, model, checker)
                init = initial_run(ops)
                res = POLICIES[pol](ops, init)
                f.write(json.dumps(record(short, case, cond, pol, init, res, ops.usage)) + "\n")
                f.flush()
        el = time.time() - t0
        print(f"[{n}/{total}] {el/60:.1f} min elapsed, ~{el/max(n,1)*(total-n)/60:.1f} min left", flush=True)
print("done", out)
