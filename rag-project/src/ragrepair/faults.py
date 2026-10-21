"""Component faults injected into retrieval, prompt assembly, serving, and index content.

Each condition models a failure that has been reported for deployed RAG services:

- clean:     no fault.
- encoder:   the query service embeds questions with a different encoder (all-MiniLM-L6-v2) than
             the one the index was built with (bge-small-en-v1.5), e.g. after a library default
             changed. Retrieval degrades for every question, not only for the answer passage.
- truncation: a template bug cuts every passage to its first `TRUNC_CHARS` characters
             (`text[:max_chars]` with the wrong setting).
- overflow:  the serving runtime's context window (`WINDOW` tokens) is smaller than the
             assembled prompt and silently keeps only the last `WINDOW` tokens, dropping the
             system prompt and the top-ranked passages (as with an unset `num_ctx` default).
- stale:     an outdated version of the answer passage (older date, old value) is still in the
             index alongside the current one; retrieval ranks both naturally.
- absent:    the answer passage is missing from the index. Nothing can recover it, so the right
             behavior is to abstain.
- encoder+truncation: two independent faults at once.
"""

from __future__ import annotations

import numpy as np

from .embed import INDEX_ENCODER, MISMATCH_ENCODER

TRUNC_CHARS = 100
WINDOW = 512

CONDITIONS: dict[str, dict] = {
    "clean": {},
    "encoder": {"encoder": MISMATCH_ENCODER},
    "truncation": {"truncate": True},
    "overflow": {"window": WINDOW},
    "stale": {"stale": True},
    "absent": {"remove": True},
    "encoder+truncation": {"encoder": MISMATCH_ENCODER, "truncate": True},
}
ANSWERABLE = [c for c in CONDITIONS if c != "absent"]
FAULTS = [c for c in ANSWERABLE if c != "clean"]


class FaultyIndex:
    """Applies one case's fault consistently to every retrieval call the policies make."""

    def __init__(self, store, case: dict, condition: str):
        self.store, self.case, self.condition = store, case, condition
        self.fault = CONDITIONS[condition]
        mask = store.base_mask.copy()
        if self.fault.get("stale"):
            mask[store.pos[case["stale_doc"]["doc_id"]]] = True
        if self.fault.get("remove"):
            mask[store.pos[case["answer_doc_id"]]] = False
        self.mask = mask
        self.encoder = self.fault.get("encoder", INDEX_ENCODER)

    def retrieve(self, n: int, method: str = "dense") -> list[dict]:
        if method == "dense":
            hits = self.store.dense.rank(self.case["question"], n, self.encoder, self.mask)
        else:
            hits = self.store.bm25.rank(self.case["question"], n, self.mask)
        return [dict(self.store.docs[i], score=s, method=method) for i, s in hits]

    def template_max_chars(self) -> int | None:
        return TRUNC_CHARS if self.fault.get("truncate") else None

    def serving_window(self) -> int | None:
        return self.fault.get("window")
