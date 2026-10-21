"""Replay saved failure cases through the full pipeline with the proposed repair policy.

    pytest -m replay tests/test_regression.py          # replay from recorded responses, no GPU
    RAGREPAIR_LIVE=1 pytest -m replay tests/...         # regenerate on the local model instead

In replay mode every model response and support score must come from the recorded caches. If a
code change alters a prompt, the retrieval, or a repair decision, the needed response is missing
and the test fails with CacheMiss: rerun live, inspect, and re-record.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "regression_cases.json"

pytestmark = pytest.mark.replay


@pytest.fixture(scope="module")
def env():
    if not SUITE.exists() or not (ROOT / "data" / "embeddings.npz").exists():
        pytest.skip("needs tests/regression_cases.json and the built data/ directory")
    from ragrepair.data import load_dataset
    from ragrepair.llm import MLXModel, ResponseCache
    from ragrepair.store import Store
    from ragrepair.support import SupportChecker

    live = os.environ.get("RAGREPAIR_LIVE") == "1"
    suite = json.loads(SUITE.read_text())
    ds = load_dataset(ROOT / "data" / "dataset.json")
    cases = {q["id"]: q for q in ds["dev"] + ds["test"]}
    model = MLXModel(suite["model"], ResponseCache(ROOT / "cache" / "llm_cache.sqlite"), replay_only=not live)
    checker = SupportChecker(cache_path=ROOT / "cache" / "support_cache.sqlite", replay_only=not live)
    return suite, cases, Store(ds, ROOT / "data"), model, checker


def _ids():
    if not SUITE.exists():
        return []
    return [f"{c['kind']}:{c['condition']}:{c['qid'][:8]}" for c in json.loads(SUITE.read_text())["cases"]]


@pytest.mark.parametrize("i", range(len(_ids())), ids=_ids())
def test_case(env, i):
    from ragrepair.evaluate import score_answer
    from ragrepair.policies import POLICIES
    from ragrepair.system import CaseOps, initial_run

    suite, cases, store, model, checker = env
    c = suite["cases"][i]
    case = cases[c["qid"]]
    ops = CaseOps(case, c["condition"], store, model, checker)
    res = POLICIES[suite["policy"]](ops, initial_run(ops))
    assert score_answer(res["answer"], res["rendered"], case, c["condition"])["success"], (res["answer"], res["actions"])
