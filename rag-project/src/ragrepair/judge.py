"""Answer-equivalence judge for predictions the deterministic matcher rejects.

The judge sees only the question, the gold answer, and the prediction (never the passages), and
decides whether they name the same thing: aliases, abbreviations, extra or missing qualifiers,
and formatting differences count as equivalent.
"""

from __future__ import annotations

import hashlib

from .text import is_abstain, is_correct

PROMPT = (
    "Question: {q}\nReference answer: {gold}\nPredicted answer: {pred}\n\n"
    "Does the predicted answer give the same answer as the reference? Count aliases, abbreviations, "
    "different formatting, and answers that are slightly more or less specific but clearly refer to "
    "the same thing as the same. Count a different entity, value, or a non-answer as different.\n"
    "Reply with exactly one word: YES or NO."
)


def key(question: str, gold: str, pred: str) -> str:
    return hashlib.sha256(f"{question}\\x1f{gold}\\x1f{pred}".encode()).hexdigest()[:20]


def needs_judge(pred: str | None, gold: str) -> bool:
    return not is_abstain(pred) and not is_correct(pred, gold)


def judge(model, question: str, gold: str, pred: str) -> bool:
    msgs = [{"role": "system", "content": "You grade short answers to quiz questions."},
            {"role": "user", "content": PROMPT.format(q=question, gold=gold, pred=pred)}]
    r = model.generate(msgs, max_tokens=4, temperature=0.0)
    return r.text.strip().upper().startswith("YES")
