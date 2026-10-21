"""Judge every (question, gold, prediction) triple that string matching rejects; write data/judgments.json."""
import argparse
import json
from pathlib import Path

from ragrepair.data import load_dataset
from ragrepair.judge import judge, key, needs_judge
from ragrepair.llm import MLXModel, ResponseCache

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--dataset", default="data/dataset.json")
ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
ap.add_argument("--out", default="data/judgments.json")
args = ap.parse_args()

ds = load_dataset(args.dataset)
qtext = {q["id"]: q["question"] for q in ds["dev"] + ds["test"]}
out = Path(args.out)
done = json.loads(out.read_text()) if out.exists() else {}
todo = {}
for f in args.files:
    for line in Path(f).read_text().splitlines():
        r = json.loads(line)
        for pred in (r["initial_answer"], r["final_answer"]):
            if needs_judge(pred, r["gold"]):
                k = key(qtext[r["qid"]], r["gold"], pred)
                if k not in done:
                    todo[k] = (qtext[r["qid"]], r["gold"], pred)
print(f"{len(done)} cached judgments, {len(todo)} to judge", flush=True)
m = MLXModel(args.model, ResponseCache("cache/llm_cache.sqlite"))
for i, (k, (q, g, p)) in enumerate(todo.items(), 1):
    done[k] = dict(question=q, gold=g, pred=p, equivalent=judge(m, q, g, p))
    if i % 100 == 0:
        out.write_text(json.dumps(done, indent=1))
        print(i, flush=True)
out.write_text(json.dumps(done, indent=1))
yes = sum(v["equivalent"] for v in done.values())
print(f"judged {len(done)}: {yes} equivalent")
