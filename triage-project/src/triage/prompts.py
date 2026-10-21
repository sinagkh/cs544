"""Prompts for zero-shot, retrieval-augmented few-shot, and explained triage."""

from __future__ import annotations

import re

DEFINITIONS = (
    "- bug: something is broken or behaves incorrectly (a crash, an error, a regression, wrong output).\n"
    "- feature: a request for new functionality or an improvement to existing behavior.\n"
    "- question: a request for help, clarification or usage support, not a defect report or a feature request."
)
EXAMPLE_CHARS = 300


def _system(repo: str) -> dict:
    return {"role": "system", "content": f"You are a maintainer triaging GitHub issues for the {repo} repository."}


def _issue(item: dict) -> str:
    return f"Issue title: {item['title']}\nIssue body:\n{item['body'] or '(empty)'}"


def _examples(examples: list[dict], same_repo: bool) -> str:
    where = "this repository" if same_repo else "other repositories"
    parts = [f"Similar past issues from {where}, with the labels maintainers gave them:"]
    for i, e in enumerate(examples, 1):
        body = re.sub(r"\s+", " ", e["body"])[:EXAMPLE_CHARS]
        parts.append(f"Example {i} (label: {e['label']})\nTitle: {e['title']}\nBody: {body}")
    return "\n\n".join(parts)


def zero_shot(item: dict) -> list[dict]:
    user = (f"Classify the issue into exactly one category:\n{DEFINITIONS}\n\n{_issue(item)}\n\n"
            "Reply with exactly one word: bug, feature, or question.")
    return [_system(item["repo"]), {"role": "user", "content": user}]


def few_shot(item: dict, examples: list[dict], same_repo: bool = True) -> list[dict]:
    user = (f"Classify the issue into exactly one category:\n{DEFINITIONS}\n\n{_examples(examples, same_repo)}\n\n"
            f"Now the new issue.\n{_issue(item)}\n\nReply with exactly one word: bug, feature, or question.")
    return [_system(item["repo"]), {"role": "user", "content": user}]


def explain(item: dict, examples: list[dict], same_repo: bool = True) -> list[dict]:
    user = (f"Classify the issue into exactly one category:\n{DEFINITIONS}\n\n{_examples(examples, same_repo)}\n\n"
            f"Now the new issue.\n{_issue(item)}\n\n"
            "Reply in exactly this format, quoting the issue word for word:\n"
            "Label: <bug, feature, or question>\n"
            'Evidence: "<short exact quote from the issue>"; "<another short exact quote>"')
    return [_system(item["repo"]), {"role": "user", "content": user}]


_LABEL = re.compile(r"label\s*:\s*(bug|feature|question)", re.I)
_QUOTE = re.compile(r'"([^"]{3,200})"')


def parse_explanation(text: str) -> tuple[str | None, list[str]]:
    m = _LABEL.search(text)
    ev = text.split("Evidence", 1)[1] if "Evidence" in text else ""
    return (m.group(1).lower() if m else None), [q.strip() for q in _QUOTE.findall(ev)][:2]
