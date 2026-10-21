"""Classical baselines: TF-IDF + logistic regression and bge-small embeddings + logistic regression.

Trained on the labeled history (all repositories) for calib/test/fresh/variants, and on the history
without the target repository for the new-repository condition. Writes results/classical.jsonl.
"""
import json
import time
from pathlib import Path

import numpy as np

from triage.data import REPOS, load
from triage.models import emb_lr, proba_dict, text_of, tfidf_lr

ds = load()
E = dict(np.load("data/emb.npz", allow_pickle=False))
variants = json.loads(Path("data/variants.json").read_text())["variants"] if Path("data/variants.json").exists() else []
test_by_id = {x["id"]: x for x in ds["test"]}
sets = {"calib": ds["calib"], "test": ds["test"], "fresh": ds["fresh"],
        "newrepo": [test_by_id[i] for i in ds["newrepo_ids"]], "variants": variants}
H = ds["history"]
y = [h["label"] for h in H]


def fit(train):
    a = tfidf_lr().fit([text_of(h) for h in train], [h["label"] for h in train])
    b = emb_lr().fit(np.stack([E[h["id"]] for h in train]), [h["label"] for h in train])
    return a, b


models = {"all": fit(H)}
for repo in REPOS:
    models[repo] = fit([h for h in H if h["repo"] != repo])

rows = []
for cond, items in sets.items():
    for it in items:
        if it["id"] not in E:
            continue
        a, b = models[it["repo"]] if cond == "newrepo" else models["all"]
        for name, clf, x in (("tfidf", a, [text_of(it)]), ("emb", b, E[it["id"]][None])):
            t0 = time.time()
            p = clf.predict_proba(x)[0]
            probs = proba_dict(p, clf.classes_)
            rows.append(dict(id=it["id"], base_id=it.get("base_id", it["id"]), kind=it.get("kind", "original"),
                             applicable=it.get("applicable", True), condition=cond, approach=name, repo=it["repo"],
                             label=it["label"], pred=max(probs, key=probs.get), probs=probs,
                             logits={k: float(np.log(max(v, 1e-12))) for k, v in probs.items()},
                             seconds=time.time() - t0))
Path("results/classical.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "classical predictions")
