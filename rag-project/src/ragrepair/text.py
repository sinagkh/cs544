"""Answer normalization, scoring, and lexical support checks."""

from __future__ import annotations

import re
import string
from collections import Counter

ABSTAIN = "INSUFFICIENT"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """HotpotQA / SQuAD normalization: lowercase, drop punctuation and articles, squash spaces."""
    s = s.lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def is_abstain(pred: str | None) -> bool:
    if pred is None:
        return True
    p = pred.strip().lower()
    return p == "" or "insufficient" in p


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-token phrase containment after normalization."""
    h, n = normalize_answer(haystack), normalize_answer(needle)
    if not n:
        return False
    return f" {n} " in f" {h} "


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def _numnorm(s: str) -> str:
    return " ".join(_NUM_WORDS.get(t, t) for t in normalize_answer(s.replace(",", "")).split())


def is_correct(pred: str | None, gold: str) -> bool:
    """Deterministic correctness (the string-match layer; an LLM judge can add aliases later).

    Correct if exact match (after mapping number words to digits), F1 >= 0.8, or the gold answer
    appears as a whole phrase in a short prediction (at most 4 tokens longer than the gold answer).
    """
    if is_abstain(pred):
        return False
    if exact_match(pred, gold) or _numnorm(pred) == _numnorm(gold) or f1_score(pred, gold) >= 0.8:
        return True
    return contains_phrase(pred, gold) and len(tokens(pred)) <= len(tokens(gold)) + 4


def lexically_supported(answer: str | None, context: str, min_coverage: float = 0.8) -> bool:
    """Observable support check: does the answer text appear in the rendered context?

    Uses whole-phrase containment, falling back to token coverage so that small formatting
    differences ("the 1990s" vs "1990s") do not count as unsupported.
    """
    if is_abstain(answer):
        return False
    if contains_phrase(context, answer):
        return True
    ans = tokens(answer)
    if not ans:
        return False
    ctx = set(tokens(context))
    return sum(t in ctx for t in ans) / len(ans) >= min_coverage
