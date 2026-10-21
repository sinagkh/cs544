"""Repair policies compared in the study.

All policies start from the same initial answer (shared through the response cache) and may make
at most two additional model calls (answer regenerations and LLM verifier calls both count). The
MiniCheck support check is not an LLM call; its cost is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import parse_verdict, selfcheck_hint, verify_messages
from .system import CaseOps
from .text import ABSTAIN, is_abstain, lexically_supported

BUDGET = 2
TAU_REPAIR = 0.5  # MiniCheck score below this triggers a repair (MiniCheck's default threshold;
#                   balanced accuracy on dev was flat at 0.59-0.60 for thresholds 0.02-0.5)
TAU_ABSTAIN = 0.01  # ...and below this, after repairs, the answer becomes INSUFFICIENT
# TAU_ABSTAIN is the 2nd percentile of MiniCheck scores of correct, grounded answers on the dev split
# (rule fixed before the test run: abstention may cost at most ~2% of good answers).


def keep_newest(docs: list[dict]) -> list[dict]:
    """Drop older versions of any document whose title appears more than once (keep rank order)."""
    newest = {}
    for d in docs:
        if d["title"] not in newest or d["date"] > newest[d["title"]]["date"]:
            newest[d["title"]] = d
    return [d for d in docs if newest[d["title"]] is d]


def union(*lists: list[dict]) -> list[dict]:
    seen, out = set(), []
    for lst in lists:
        for d in lst:
            if d["doc_id"] not in seen:
                seen.add(d["doc_id"])
                out.append(d)
    return out


def ctx_text(rendered: list[dict]) -> str:
    return " ".join(p["rendered"] for p in rendered)


def fit_to_window(ops: CaseOps, rendered: list[dict], window: int) -> list[dict]:
    """Keep passages in rank order while the whole prompt fits in `window` tokens."""
    kept = []
    for p in rendered:
        if ops.count_tokens(kept + [p]) <= window - 8:
            kept.append(p)
        else:
            break
    if not kept and rendered:  # not even one passage fits: shorten the first one
        p = dict(rendered[0])
        lo, hi = 0, len(p["rendered"])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ops.count_tokens([dict(p, rendered=p["source"][:mid])]) <= window - 8:
                lo = mid
            else:
                hi = mid - 1
        kept = [dict(p, rendered=p["source"][:lo], truncated=lo < len(p["source"]))]
    return kept


def _final(out: dict, rendered: list[dict], actions: list[str], flags: list[str] | None = None,
           support_scores: list[float] | None = None) -> dict:
    return dict(answer=out["answer"], evidence=out.get("evidence", []), raw=out.get("raw", ""),
                rendered=rendered, actions=actions, flags=flags or [], support_scores=support_scores or [])


# ---------------------------------------------------------------- generic baselines

def no_repair(ops: CaseOps, init: dict) -> dict:
    return _final(init, init["rendered"], [])


def retry(ops: CaseOps, init: dict) -> dict:
    """Generic retry: if the answer is missing or INSUFFICIENT, resample once with the same prompt."""
    if not is_abstain(init["answer"]):
        return _final(init, init["rendered"], [])
    out = ops.answer(init["rendered"], "retry", temperature=0.7, seed=1)
    return _final(out, init["rendered"], ["resample"])


def _llm_verify(ops: CaseOps, rendered: list[dict], answer: str) -> bool:
    v = ops.generate(verify_messages(ops.question, rendered, answer), "verify", max_tokens=6)
    return parse_verdict(v.text)


def self_check(ops: CaseOps, init: dict, verifier: str = "llm") -> dict:
    """Generic self-check: verify the answer; if unsupported, regenerate on the same passages with a hint."""
    if is_abstain(init["answer"]):
        return _final(init, init["rendered"], [])
    scores = []
    if verifier == "llm":
        ok = _llm_verify(ops, init["rendered"], init["answer"])
    else:
        ok, s = ops.support(init["rendered"], init["answer"])
        scores.append(s)
    if ok:
        return _final(init, init["rendered"], ["verified"], support_scores=scores)
    out = ops.answer(init["rendered"], "regen", hint=selfcheck_hint(init["answer"]))
    return _final(out, init["rendered"], ["verified_unsupported", "regenerate"], support_scores=scores)


def repair_all(ops: CaseOps, init: dict) -> dict:
    """Apply every available repair unconditionally: dense top-10 plus BM25 top-5, full rendering,
    newest versions only, one regeneration. It has no way to know the serving window."""
    docs = keep_newest(union(ops.retrieve(CaseOps.WIDE_K), ops.retrieve(CaseOps.K, "bm25")))
    rendered = ops.render(docs, full=True)
    out = ops.answer(rendered, "regen")
    return _final(out, rendered, ["widen", "add_bm25", "full_render", "keep_newest", "regenerate"])


# ---------------------------------------------------------------- trace-based diagnosis

def free_flags(ops: CaseOps, init: dict) -> list[str]:
    """Checks on the trace that need no model call."""
    flags = []
    if any(p["truncated"] for p in init["rendered"]):
        flags.append("truncation")  # the template dropped text that retrieval returned
    titles = [d["title"] for d in init["docs"]]
    if len(set(titles)) < len(titles):
        flags.append("version_conflict")  # two versions of one document in the context
    if init["processed_tokens"] < init["sent_tokens"]:
        flags.append("overflow")  # the server processed fewer tokens than were sent
    top = init["docs"][0].get("score") if init["docs"] else None
    if top is not None and top < ops.thresholds["retrieval_top1_p05"]:
        flags.append("low_retrieval")  # top similarity below the 5th percentile of normal traffic
    return flags


@dataclass
class _State:
    docs: list
    full: bool = False
    window: int | None = None
    method: str = "dense"
    widened: bool = False
    lexical: bool = False


def _render(ops: CaseOps, st: _State) -> list[dict]:
    rendered = ops.render(st.docs, full=st.full)
    return fit_to_window(ops, rendered, st.window) if st.window else rendered


def _next_step(ops: CaseOps, st: _State) -> str | None:
    if not st.widened:
        st.widened = True
        st.docs = keep_newest(ops.retrieve(CaseOps.WIDE_K, st.method))
        return f"widen_{st.method}"
    if not st.lexical:
        st.lexical = True
        st.docs = keep_newest(union(ops.retrieve(CaseOps.K, "dense"), ops.retrieve(CaseOps.K, "bm25")))
        return "add_bm25"
    return None


def diagnose(ops: CaseOps, init: dict, support: str = "nli", recheck: bool = True, abstain: bool = True,
             use_free: bool = True) -> dict:
    """Targeted repair chosen from the trace.

    1. Free checks (no model call) -> matching repair:
         truncation -> render in full; version conflict -> keep the newest version;
         overflow -> fit the prompt to the window the server reported;
         low retrieval score -> fall back to BM25.
    2. Support check on the (repaired) answer: "nli" (MiniCheck), "llm" (the self-check verifier,
       which costs a call), or "lexical" (answer text appears in the passages); "none" skips it.
       While unsupported and budget remains: widen retrieval, then add BM25 results, regenerating
       after each step. `recheck=False` checks support only when no free check fired.
    3. If the final answer is still clearly unsupported, return INSUFFICIENT.
    """
    flags = free_flags(ops, init) if use_free else []
    st = _State(docs=list(init["docs"]))
    actions, scores = [], []
    budget = BUDGET
    out, rendered = init, init["rendered"]

    memo: dict = {}

    def nli_score(ans: str, rend: list[dict]) -> float:
        k = (ans, tuple(p["rendered"] for p in rend))
        if k not in memo:
            memo[k] = ops.support(rend, ans)[1]
            scores.append(memo[k])
        return memo[k]

    def check(ans: str, rend: list[dict]) -> bool:
        nonlocal budget
        if is_abstain(ans):
            return False
        if support == "nli":
            return nli_score(ans, rend) >= TAU_REPAIR
        if support == "lexical":
            return lexically_supported(ans, ctx_text(rend))
        if support == "llm":
            if not lexically_supported(ans, ctx_text(rend)):
                return False
            if budget <= 0:
                return True
            budget -= 1
            return _llm_verify(ops, rend, ans)
        return True

    if flags:
        if "version_conflict" in flags:
            st.docs = keep_newest(st.docs)
            actions.append("keep_newest")
        if "low_retrieval" in flags:
            st.method = "bm25"
            st.docs = keep_newest(ops.retrieve(CaseOps.K, "bm25"))
            actions.append("bm25_fallback")
        if "truncation" in flags:
            st.full = True
            actions.append("full_render")
        if "overflow" in flags:
            st.window = init["processed_tokens"]
            actions.append("fit_window")
        rendered = _render(ops, st)
        out = ops.answer(rendered, "regen")
        budget -= 1
        actions.append("regenerate")

    if support != "none" and (recheck or not flags):
        while budget > 0:
            if check(out["answer"], rendered):
                break
            if budget <= 0:
                break
            if not flags:
                flags.append("unsupported")
            step = _next_step(ops, st)
            if step is None:
                break
            actions.append(step)
            rendered = _render(ops, st)
            out = ops.answer(rendered, "regen")
            budget -= 1
            actions.append("regenerate")

    if abstain and support != "none" and not is_abstain(out["answer"]):
        if support == "nli":
            give_up = nli_score(out["answer"], rendered) < TAU_ABSTAIN
        else:
            give_up = not lexically_supported(out["answer"], ctx_text(rendered))
        if give_up:
            out = dict(out, answer=ABSTAIN)
            actions.append("abstain")
    return _final(out, rendered, actions, flags, scores)


POLICIES = {
    # Generic baselines
    "no_repair": no_repair,
    "retry": retry,
    "self_check": self_check,
    "self_check_nli": lambda ops, init: self_check(ops, init, verifier="nli"),
    "repair_all": repair_all,
    "checks_only": lambda ops, init: diagnose(ops, init, support="none", abstain=False),
    # Proposed policy (support check chosen on the dev split): trace checks + text-match support
    "diagnose": lambda ops, init: diagnose(ops, init, support="lexical"),
    # Same policy with a different support check
    "diagnose_llm": lambda ops, init: diagnose(ops, init, support="llm"),
    "diagnose_nli": lambda ops, init: diagnose(ops, init, support="nli"),
    # Ablations of the proposed policy
    "diagnose_norecheck": lambda ops, init: diagnose(ops, init, support="lexical", recheck=False),
    "diagnose_noabstain": lambda ops, init: diagnose(ops, init, support="lexical", abstain=False),
    "support_only": lambda ops, init: diagnose(ops, init, support="lexical", use_free=False),
}
