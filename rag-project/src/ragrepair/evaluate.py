"""Per-case scoring and aggregate statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from .faults import ANSWERABLE
from .policies import ctx_text
from .text import contains_phrase, exact_match, f1_score, is_abstain, is_correct, lexically_supported


def fault_delivery(case: dict, init: dict) -> dict:
    """What the injected fault actually did to the initial model input (uses gold; evaluation only)."""
    seen = init.get("_processed_text") or ctx_text(init["rendered"])
    gold_titles = set(case["gold_titles"])
    return dict(
        gold_in_context=contains_phrase(seen, case["answer"]),
        stale_in_context=any(p.get("stale") for p in init["rendered"]),
        truncated_any=any(p["truncated"] for p in init["rendered"]),
        overflow_cut=init.get("processed_tokens", 0) < init.get("sent_tokens", 0),
        answer_doc_retrieved=any(p["doc_id"] == case["answer_doc_id"] for p in init["rendered"]),
        both_gold_retrieved=gold_titles <= {p["title"] for p in init["rendered"] if not p.get("stale")},
        top_score=init["docs"][0].get("score") if init["docs"] else None,
    )


def score_answer(answer: str, rendered: list[dict], case: dict, condition: str) -> dict:
    abst = is_abstain(answer)
    correct = is_correct(answer, case["answer"])
    supported = (not abst) and lexically_supported(answer, ctx_text(rendered))
    success = correct if condition in ANSWERABLE else abst
    return dict(
        abstain=abst,
        em=0.0 if abst else exact_match(answer, case["answer"]),
        f1=0.0 if abst else f1_score(answer, case["answer"]),
        correct=correct,
        supported=supported,
        supported_correct=correct and supported,
        stale_value=(not abst) and is_correct(answer, case["stale_answer"]),
        success=success,
    )


def record(model: str, case: dict, condition: str, policy: str, init: dict, res: dict, usage) -> dict:
    init_scores = score_answer(init["answer"], init["rendered"], case, condition)
    final_scores = score_answer(res["answer"], res["rendered"], case, condition)
    return dict(
        model=model, qid=case["id"], qtype=case["type"], condition=condition, policy=policy,
        gold=case["answer"], stale_answer=case["stale_answer"],
        initial_answer=init["answer"], final_answer=res["answer"],
        actions=res["actions"], flags=res["flags"],
        delivery=fault_delivery(case, init),
        initial={k: init_scores[k] for k in ("correct", "success", "abstain")},
        **final_scores,
        calls=usage.calls, extra_calls=usage.calls - 1,
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        seconds=usage.seconds, calls_by_kind=usage.by_kind,
        support_checks=usage.support_checks, support_seconds=usage.support_seconds,
        support_scores=res.get("support_scores", []),
        final_context_ids=[p["doc_id"] for p in res["rendered"]],
    )


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean; one value per question, so resampling is by question."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    d = a.astype(float) - b.astype(float)
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return float(d.mean()), lo, hi


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar test on paired binary outcomes."""
    a, b = a.astype(bool), b.astype(bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)
