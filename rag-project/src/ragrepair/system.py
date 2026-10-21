"""The RAG service as seen by a repair policy: retrieval, rendering, generation, and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass, field

from .faults import FaultyIndex
from .prompts import answer_messages, parse_answer, render_passages


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    support_checks: int = 0
    support_seconds: float = 0.0
    by_kind: dict = field(default_factory=dict)

    def add(self, kind: str, r):
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.seconds += r.seconds
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


class CaseOps:
    """Operations available to a policy for one (question, condition) case.

    A policy sees the question, the retrieved documents (text, metadata, retrieval scores), the
    rendered passages, the token counts it sent and the server reports, and model outputs. It
    never sees the condition label or the gold answer.
    """

    K = 5
    WIDE_K = 10

    def __init__(self, case: dict, condition: str, store, model, checker=None):
        self.case, self.condition, self.model, self.checker = case, condition, model, checker
        self.index = FaultyIndex(store, case, condition)
        self.thresholds = store.thresholds
        self.deployed_max_chars = self.index.template_max_chars()
        self.window = self.index.serving_window()
        self.usage = Usage()

    @property
    def question(self) -> str:
        return self.case["question"]

    def retrieve(self, n: int | None = None, method: str = "dense") -> list[dict]:
        return self.index.retrieve(n or self.K, method)

    def render(self, docs: list[dict], full: bool = False) -> list[dict]:
        """Render with the deployed template, or with no truncation when `full` (a repair)."""
        return render_passages(docs, None if full else self.deployed_max_chars)

    def count_tokens(self, rendered: list[dict], hint: str | None = None) -> int:
        return self.model.count_tokens(answer_messages(self.question, rendered, hint))

    def generate(self, messages, kind: str, max_tokens: int = 48, temperature: float = 0.0, seed: int = 0):
        r = self.model.generate(messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
                                max_prompt_tokens=self.window)
        self.usage.add(kind, r)
        return r

    def answer(self, rendered: list[dict], kind: str, hint: str | None = None, temperature: float = 0.0,
               seed: int = 0) -> dict:
        r = self.generate(answer_messages(self.question, rendered, hint), kind, temperature=temperature, seed=seed)
        ans, evidence = parse_answer(r.text)
        return dict(answer=ans, evidence=evidence, raw=r.text, sent_tokens=r.sent_tokens,
                    processed_tokens=r.prompt_tokens, _processed_text=r.processed_text)

    def support(self, rendered: list[dict], answer: str) -> tuple[bool, float]:
        """Grounding check of `answer` against the passages the model saw (MiniCheck)."""
        passages = [f"{p['title']}: {p['rendered']}" for p in rendered]
        before = self.checker.seconds
        ok, s = self.checker.supported(passages, self.question, answer)
        self.usage.support_checks += 1
        self.usage.support_seconds += self.checker.seconds - before
        return ok, s


def initial_run(ops: CaseOps) -> dict:
    """The deployed pipeline with no repair: dense top-k, deployed template, one generation."""
    docs = ops.retrieve()
    rendered = ops.render(docs)
    out = ops.answer(rendered, "initial")
    return dict(docs=docs, rendered=rendered, **out)
