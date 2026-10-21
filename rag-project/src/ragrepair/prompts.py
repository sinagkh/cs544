"""Prompt rendering and output parsing."""

from __future__ import annotations

import re

from .text import ABSTAIN

SYSTEM = "You answer questions using only the provided passages."

INSTRUCTIONS = (
    "Answer using only the passages above. Give the shortest possible answer (a name, number, date, "
    "or short phrase), not a full sentence. If the passages do not contain the information needed, "
    f"answer {ABSTAIN}.\n"
    "Respond in exactly this format:\n"
    "Answer: <answer>\n"
    "Evidence: <passage ids, e.g. P1, P3>"
)


def render_passages(docs: list[dict], max_chars: int | None) -> list[dict]:
    """Render retrieved docs for the prompt. `max_chars` is the template's per-passage limit."""
    out = []
    for i, d in enumerate(docs, start=1):
        src = d["text"]
        txt = src if max_chars is None or len(src) <= max_chars else src[:max_chars]
        out.append(dict(pid=f"P{i}", doc_id=d["doc_id"], title=d["title"], date=d["date"],
                        stale=d.get("stale", False), source=src, rendered=txt,
                        truncated=len(txt) < len(src), score=d.get("score"), method=d.get("method")))
    return out


def context_block(rendered: list[dict]) -> str:
    return "\n\n".join(f"[{p['pid']}] {p['title']} (last updated {p['date']})\n{p['rendered']}" for p in rendered)


def answer_messages(question: str, rendered: list[dict], hint: str | None = None) -> list[dict]:
    user = f"Passages:\n{context_block(rendered)}\n\nQuestion: {question}\n\n{INSTRUCTIONS}"
    if hint:
        user += f"\n\n{hint}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def verify_messages(question: str, rendered: list[dict], answer: str) -> list[dict]:
    user = (
        f"Passages:\n{context_block(rendered)}\n\nQuestion: {question}\nProposed answer: {answer}\n\n"
        "Is the proposed answer correct and directly supported by the passages? "
        "Reply with exactly one word: SUPPORTED or UNSUPPORTED."
    )
    return [{"role": "system", "content": "You check whether answers are supported by passages."},
            {"role": "user", "content": user}]


def selfcheck_hint(prev_answer: str) -> str:
    return (
        f'A previous attempt answered "{prev_answer}", and a checker judged that answer unsupported '
        "by the passages. Re-read the passages carefully and answer again. If the passages do not "
        f"support an answer, answer {ABSTAIN}."
    )


_ANS = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_EVI = re.compile(r"\bP(\d+)\b")


def parse_answer(text: str) -> tuple[str, list[str]]:
    """Extract (answer, evidence ids). Falls back to the first non-empty line."""
    ans = None
    for line in text.splitlines():
        m = _ANS.search(line)
        if m:
            ans = m.group(1).strip()
            break
    if ans is None:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ans = lines[0] if lines else ""
    ans = ans.strip().strip('"').strip("*").strip().rstrip(".").strip()
    ans = re.sub(r"\s*[\[(]?\bP\d+(\s*,\s*P\d+)*[\])]?\s*$", "", ans).strip().rstrip(".").strip()
    evi_part = text.split("Evidence", 1)[1] if "Evidence" in text else ""
    evidence = [f"P{n}" for n in dict.fromkeys(_EVI.findall(evi_part))]
    return ans, evidence


def parse_verdict(text: str) -> bool:
    """True if the verifier says SUPPORTED."""
    t = text.strip().upper()
    if "UNSUPPORTED" in t or "NOT SUPPORTED" in t:
        return False
    return "SUPPORTED" in t
