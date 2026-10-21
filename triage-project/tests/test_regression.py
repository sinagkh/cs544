"""Replay the triage regression suite through the full pipeline (retrieval, prompt, model, calibration).

    pytest -m replay                      # from recorded model responses, no GPU
    TRIAGE_LIVE=1 pytest -m replay        # re-score on the local model
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "regression_cases.json"
pytestmark = pytest.mark.replay


@pytest.fixture(scope="module")
def env():
    if not SUITE.exists() or not (ROOT / "data" / "emb.npz").exists():
        pytest.skip("needs tests/regression_cases.json and the built data/ directory")
    from triage.data import load
    from triage.llm import Cache, MLXModel
    from triage.models import Retriever

    suite = json.loads(SUITE.read_text())
    ds = load(ROOT / "data" / "dataset.json")
    E = dict(np.load(ROOT / "data" / "emb.npz", allow_pickle=False))
    var = json.loads((ROOT / "data" / "variants.json").read_text())["variants"]
    items = {x["id"]: x for x in ds["test"] + var}
    model = MLXModel(suite["model"], Cache(ROOT / "cache" / "llm_cache.sqlite"),
                     replay_only=os.environ.get("TRIAGE_LIVE") != "1")
    ret = Retriever(ds["history"], np.stack([E[h["id"]] for h in ds["history"]]))
    return suite, items, E, model, ret


def triage(env, item):
    from triage.data import LABELS
    from triage.prompts import few_shot
    from triage.selective import softmax

    suite, _, E, model, ret = env
    r = model.label_probs(few_shot(item, ret.topk(E[item["id"]], item["repo"], suite["k"])))
    p = softmax(np.array([[r["logits"][l] for l in LABELS]]), suite["temperature"])[0]
    return LABELS[int(p.argmax())], float(p.max()) >= suite["threshold"]


def _ids():
    if not SUITE.exists():
        return []
    return [f"{c['kind']}:{c['id'][-14:]}" for c in json.loads(SUITE.read_text())["cases"]]


@pytest.mark.parametrize("i", range(len(_ids())), ids=_ids())
def test_case(env, i):
    suite, items, *_ = env
    c = suite["cases"][i]
    label, auto = triage(env, items[c["id"]])
    if c["kind"] == "auto_correct":
        assert auto and label == c["label"]
    elif c["kind"] == "must_defer":
        assert not auto or label == c["label"], "a wrong label is auto-applied"
    else:
        assert label == c["label"]
