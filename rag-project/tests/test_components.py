"""Unit tests for scoring, parsing, fault injection and the observable checks (no LLM needed)."""

import json
from pathlib import Path

import pytest

from ragrepair.faults import TRUNC_CHARS, WINDOW, FaultyIndex
from ragrepair.policies import keep_newest, union
from ragrepair.prompts import parse_answer, parse_verdict, render_passages
from ragrepair.text import f1_score, is_abstain, is_correct, lexically_supported, normalize_answer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.json"


def test_normalize_and_scores():
    assert normalize_answer("The  Beatles!") == "beatles"
    assert is_correct("Tim Burton", "Tim Burton")
    assert is_correct("director Tim Burton", "Tim Burton")
    assert is_correct("7", "seven") and is_correct("1800", "1,800")
    assert not is_correct("INSUFFICIENT", "Tim Burton")
    assert not is_correct("Johnny Depp", "Tim Burton")
    assert f1_score("Tim Burton", "Burton") == pytest.approx(2 / 3)
    assert is_abstain("insufficient") and is_abstain("") and not is_abstain("1994")


def test_lexical_support():
    ctx = "Ed Wood is a 1994 American film directed by Tim Burton."
    assert lexically_supported("Tim Burton", ctx)
    assert lexically_supported("the 1994", ctx)
    assert not lexically_supported("Johnny Depp", ctx)
    assert not lexically_supported("INSUFFICIENT", ctx)


def test_parse():
    a, e = parse_answer("Answer: Tim Burton\nEvidence: P1, P3")
    assert a == "Tim Burton" and e == ["P1", "P3"]
    a, _ = parse_answer("Answer: 1994 (P2).")
    assert a == "1994"
    assert parse_verdict("SUPPORTED") and not parse_verdict("UNSUPPORTED")


def test_render_truncation_flags():
    docs = [dict(doc_id="d1", title="A", date="2024-01-01", text="x" * 300)]
    r = render_passages(docs, TRUNC_CHARS)
    assert r[0]["truncated"] and len(r[0]["rendered"]) == TRUNC_CHARS
    assert not render_passages(docs, None)[0]["truncated"]


def test_keep_newest_and_union():
    docs = [dict(title="A", date="2019-01-01", doc_id="old"), dict(title="B", date="2024-01-01", doc_id="b"),
            dict(title="A", date="2024-05-01", doc_id="new")]
    assert [d["doc_id"] for d in keep_newest(docs)] == ["b", "new"]
    assert [d["doc_id"] for d in union(docs[:2], docs[1:])] == ["old", "b", "new"]


@pytest.fixture(scope="module")
def env():
    if not (ROOT / "data" / "embeddings.npz").exists():
        pytest.skip("run build_dataset.py, make_stale_values.py and embed_store.py first")
    from ragrepair.store import Store

    d = json.loads(DATA.read_text())
    return d, Store(d, ROOT / "data")


def test_clean_retrieval_has_both_gold(env):
    d, store = env
    for case in d["test"][:40]:
        top = FaultyIndex(store, case, "clean").retrieve(5)
        assert set(case["gold_titles"]) <= {x["title"] for x in top}
        assert all(not x.get("stale") for x in top)


def test_faults(env):
    d, store = env
    thr = store.thresholds["retrieval_top1_p05"]
    low = 0
    for case in d["test"][:40]:
        ans = case["answer_doc_id"]
        absent = FaultyIndex(store, case, "absent")
        assert ans not in [x["doc_id"] for x in absent.retrieve(30)]
        assert ans not in [x["doc_id"] for x in absent.retrieve(30, "bm25")]
        stale_ids = [x["doc_id"] for x in FaultyIndex(store, case, "stale").retrieve(10)]
        assert case["stale_doc"]["doc_id"] in stale_ids  # both versions are retrievable
        enc = FaultyIndex(store, case, "encoder").retrieve(5)
        low += enc[0]["score"] < thr
        assert FaultyIndex(store, case, "truncation").template_max_chars() == TRUNC_CHARS
        assert FaultyIndex(store, case, "overflow").serving_window() == WINDOW
        assert FaultyIndex(store, case, "clean").serving_window() is None
    assert low >= 30  # the mismatched encoder produces abnormally low similarity scores
