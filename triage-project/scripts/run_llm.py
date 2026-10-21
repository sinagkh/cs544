"""Score issues with the LLM, zero-shot and retrieval-augmented few-shot; append to results/llm_<model>.jsonl.

Conditions:
  calib     NLBSE'24 training-half calibration split (same-repo examples)
  calib_xrepo  the calibration split scored as if each issue's repository were new (examples from the
               other repositories), used to calibrate the new-repository pipeline
  test      NLBSE'24 test set, 1,500 issues (same-repo examples)
  fresh     issues created 2025-01 .. 2026-10 in the same repositories (history and calibration are older)
  newrepo   750 test issues scored as if the repository were new: examples come only from other repos
  variants  meaning-preserving rewrites of 210 probe issues (needs data/variants.json)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from triage.data import load
from triage.llm import Cache, MLXModel
from triage.models import Retriever, embed
from triage.prompts import few_shot, zero_shot

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
ap.add_argument("--conditions", default="calib,test,fresh,newrepo,variants")
ap.add_argument("--k", type=int, default=6)
args = ap.parse_args()

ds = load()
variants = json.loads(Path("data/variants.json").read_text())["variants"] if Path("data/variants.json").exists() else []
test_by_id = {x["id"]: x for x in ds["test"]}
sets = {
    "calib": ds["calib"], "calib_xrepo": ds["calib"], "test": ds["test"], "fresh": ds["fresh"],
    "newrepo": [test_by_id[i] for i in ds["newrepo_ids"]], "variants": variants,
}

# Embeddings for retrieval (cached on disk by id)
emb_path = Path("data/emb.npz")
E = dict(np.load(emb_path, allow_pickle=False)) if emb_path.exists() else {}
need = [x for x in ds["history"] + [y for s in sets.values() for y in s] if x["id"] not in E]
uniq = list({x["id"]: x for x in need}.values())
if uniq:
    V = embed(uniq)
    E.update({x["id"]: v for x, v in zip(uniq, V)})
    np.savez(emb_path, **E)
retriever = Retriever(ds["history"], np.stack([E[h["id"]] for h in ds["history"]]))

model = MLXModel(args.model, Cache("cache/llm_cache.sqlite"))
short = args.model.split("/")[-1]
out = Path(f"results/llm_{short}.jsonl")
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        r = json.loads(line)
        done.add((r["id"], r["condition"], r["approach"]))

t0, n = time.time(), 0
with out.open("a") as f:
    for cond in args.conditions.split(","):
        items = sets[cond]
        xrepo = cond in ("newrepo", "calib_xrepo")
        approaches = ["few"] if xrepo else ["zero", "few"]
        for i, it in enumerate(items):
            for appr in approaches:
                if (it["id"], cond, appr) in done:
                    continue
                if appr == "zero":
                    msgs, ex = zero_shot(it), []
                else:
                    ex = retriever.topk(E[it["id"]], it["repo"], args.k, same_repo=not xrepo)
                    msgs = few_shot(it, ex, same_repo=not xrepo)
                r = model.label_probs(msgs)
                pred = max(r["probs"], key=r["probs"].get)
                f.write(json.dumps(dict(id=it["id"], base_id=it.get("base_id", it["id"]), kind=it.get("kind", "original"),
                                        applicable=it.get("applicable", True), condition=cond, approach=appr,
                                        repo=it["repo"], label=it["label"], pred=pred, probs=r["probs"], logits=r["logits"],
                                        label_mass=r["label_mass"], prompt_tokens=r["prompt_tokens"], seconds=r["seconds"],
                                        retrieved=[e["id"] for e in ex], retrieved_labels=[e["label"] for e in ex])) + "\n")
                f.flush()
                n += 1
            if (i + 1) % 100 == 0:
                print(f"{cond}: {i + 1}/{len(items)}  {(time.time() - t0) / 60:.1f} min", flush=True)
print("done", out)
