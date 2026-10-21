"""Classical baselines: TF-IDF + logistic regression and bge-small embeddings + logistic regression.

Both see the same input as the LLM (title and the cleaned body, up to 1,500 characters); the
embedding classifier uses its own embeddings of that text (data/emb_clf.npz), separate from the
retrieval embeddings. Models are trained on the labeled history (all repositories) for
calib/test/fresh/variants, and without the target repository for the new-repository conditions
(`newrepo`, and `calib_xrepo`, which scores each calibration issue with the model trained without
its own repository). Writes results/classical.jsonl.
"""
import json
import time
from pathlib import Path

import numpy as np

from triage.data import BODY_CHARS, REPOS, load
from triage.models import emb_lr, embed, proba_dict, text_of, tfidf_lr

ds = load()
variants = json.loads(Path("data/variants.json").read_text())["variants"] if Path("data/variants.json").exists() else []
test_by_id = {x["id"]: x for x in ds["test"]}
sets = {"calib": ds["calib"], "calib_xrepo": ds["calib"], "test": ds["test"], "fresh": ds["fresh"],
        "newrepo": [test_by_id[i] for i in ds["newrepo_ids"]], "variants": variants}
H = ds["history"]

path = Path("data/emb_clf.npz")
E = dict(np.load(path, allow_pickle=False)) if path.exists() else {}
need = list({x["id"]: x for x in H + [y for s in sets.values() for y in s] if x["id"] not in E}.values())
if need:
    E.update(zip([x["id"] for x in need], embed(need, chars=BODY_CHARS)))
    np.savez(path, **E)


def fit(train):
    a = tfidf_lr().fit([text_of(h, BODY_CHARS) for h in train], [h["label"] for h in train])
    b = emb_lr().fit(np.stack([E[h["id"]] for h in train]), [h["label"] for h in train])
    return a, b


models = {"all": fit(H)}
for repo in REPOS:
    models[repo] = fit([h for h in H if h["repo"] != repo])

rows = []
for cond, items in sets.items():
    for it in items:
        a, b = models[it["repo"]] if cond in ("newrepo", "calib_xrepo") else models["all"]
        for name, clf, x in (("tfidf", a, [text_of(it, BODY_CHARS)]), ("emb", b, E[it["id"]][None])):
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
