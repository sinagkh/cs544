"""Meaning-preserving rewrites of an issue: the issue type must not change.

Rule-based: remove code blocks and stack traces, strip template scaffolding, add a newcomer cue.
Model-based: rewrite the prose as a non-native English speaker would (code blocks are replaced by
placeholders before rewriting and restored after, so technical content is kept verbatim).
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```.*?```", re.S)
_TRACE_LINE = re.compile(
    r"^\s*(at\s+\S+\(|at\s+\S+:\d+|File \".*\", line \d+|Traceback \(most recent call last\)|\S+Error:|\S+Exception:|"
    r"#\d+\s+0x[0-9a-f]+|\s*\^+\s*$)", re.I)
_HEADING = re.compile(r"^\s*#{1,6}\s.*$|^\s*\*\*[^*]{2,60}\*\*\s*:?\s*$", re.M)
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s?.*$", re.M)

NEWCOMER = ("Hi everyone, I'm new to this project and not very experienced, so sorry if this is a silly "
            "question or the wrong place to ask.\n\n")


def no_code(body: str) -> str:
    b = _FENCE.sub("", body)
    b = "\n".join(l for l in b.splitlines() if not _TRACE_LINE.search(l))
    return re.sub(r"\n{3,}", "\n\n", b).strip()


def no_template(body: str) -> str:
    b = _HEADING.sub("", body)
    b = _CHECKBOX.sub("", b)
    return re.sub(r"\n{3,}", "\n\n", b).strip()


def newcomer(body: str) -> str:
    return NEWCOMER + body


REWRITE_PROMPT = (
    "Rewrite the GitHub issue text below as if it were written by a non-native English speaker with "
    "limited English: use simpler words, a few grammar mistakes and missing articles. Keep every "
    "technical detail exactly (names, versions, commands, error messages, steps, and every [CODEk] "
    "placeholder). Do not add or remove information, and do not change whether it reports a problem, "
    "asks for a feature, or asks a question. Output only the rewritten text.\n\nIssue text:\n{body}"
)


def mask_code(body: str) -> tuple[str, list[str]]:
    blocks = _FENCE.findall(body)
    masked = body
    for i, b in enumerate(blocks):
        masked = masked.replace(b, f"[CODE{i}]", 1)
    return masked, blocks


def unmask_code(text: str, blocks: list[str]) -> str | None:
    for i, b in enumerate(blocks):
        if f"[CODE{i}]" not in text:
            return None
        text = text.replace(f"[CODE{i}]", b, 1)
    return text


_TECH = re.compile(r"[A-Za-z_][\w.\-/]*\d[\w.\-/]*|[A-Za-z]+[_.][\w.]+|\w+\(\)")


def tech_tokens(s: str) -> set[str]:
    return set(_TECH.findall(_FENCE.sub("", s)))


def valid_rewrite(orig: str, new: str | None) -> bool:
    if not new:
        return False
    if not (0.5 <= len(new) / max(len(orig), 1) <= 1.8):
        return False
    t = tech_tokens(orig)
    return not t or len(t & tech_tokens(new)) / len(t) >= 0.8
