"""Does the regression suite catch plausible bugs in the triage service? Inject one at a time, replay live."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import triage.prompts as P
from triage.data import LABELS, load
from triage.llm import Cache, MLXModel
from triage.models import Retriever
from triage.selective import softmax

suite = json.loads(Path("tests/regression_cases.json").read_text())
ds = load()
E = dict(np.load("data/emb.npz", allow_pickle=False))
var = json.loads(Path("data/variants.json").read_text())["variants"]
items = {x["id"]: x for x in ds["test"] + var}
model = MLXModel(suite["model"], Cache("cache/llm_cache.sqlite"))
ret = Retriever(ds["history"], np.stack([E[h["id"]] for h in ds["history"]]))
orig_defs = P.DEFINITIONS


def run(item, *, retrieval=True, same_repo=True, T=None, body_chars=None):
    it = dict(item, body=item["body"][:body_chars]) if body_chars else item
    msgs = P.few_shot(it, ret.topk(E[item["id"]], item["repo"], suite["k"], same_repo=same_repo), same_repo) \
        if retrieval else P.zero_shot(it)
    r = model.label_probs(msgs)
    p = softmax(np.array([[r["logits"][l] for l in LABELS]]), T or suite["temperature"])[0]
    return LABELS[int(p.argmax())], float(p.max()) >= suite["threshold"]


MUTATIONS = {
    "none (original)": {},
    "retrieval dropped (zero-shot prompt)": dict(retrieval=False),
    "examples from other repositories": dict(same_repo=False),
    "temperature scaling missing": dict(T=1.0),
    "label definitions removed": dict(defs=""),
    "issue body cut to 300 characters": dict(body_chars=300),
}
md = [f"# Mutation check of the triage regression suite ({len(suite['cases'])} cases)", "",
      "| Injected bug | Failing cases | By case kind |", "|---|---:|---|"]
for name, m in MUTATIONS.items():
    P.DEFINITIONS = m.get("defs", orig_defs)
    fails = Counter()
    for c in suite["cases"]:
        label, auto = run(items[c["id"]], **{k: v for k, v in m.items() if k != "defs"})
        ok = (auto and label == c["label"]) if c["kind"] == "auto_correct" else \
             (not auto or label == c["label"]) if c["kind"] == "must_defer" else label == c["label"]
        if not ok:
            fails[c["kind"]] += 1
    md.append(f"| {name} | {sum(fails.values())} / {len(suite['cases'])} | {', '.join(f'{k}: {v}' for k, v in fails.items()) or '–'} |")
    print(md[-1], flush=True)
P.DEFINITIONS = orig_defs
Path("report/mutation_check.md").write_text("\n".join(md) + "\n")
