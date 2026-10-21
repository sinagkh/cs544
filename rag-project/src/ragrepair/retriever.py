"""Retrievers over a shared document store: dense (the deployed retriever) and BM25 (lexical fallback).

The store holds every corpus passage plus the outdated versions used by the `stale` fault. Each
retrieval call passes a visibility mask, so a fault decides which documents exist for a case.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_STOP = set(
    "a an the of in on at to for from by with and or is are was were be been being as that this "
    "which who whom whose what when where why how did do does it its their his her they he she "
    "than then also not no".split()
)


def bm25_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _top(scores: np.ndarray, n: int) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:n] if np.isfinite(scores[i])]


class BM25Retriever:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        # Title counted twice: titles are strong signals for Wikipedia-style passages.
        corpus = [bm25_tokens(d["title"] + " " + d["title"] + " " + d["text"]) for d in docs]
        self.bm25 = BM25Okapi(corpus)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(bm25_tokens(query)), dtype=np.float64)

    def rank(self, query: str, n: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return [(i, float(s[i])) for i in _top(s, n)]


class DenseRetriever:
    """Cosine similarity between precomputed query and document embeddings."""

    def __init__(self, doc_emb: np.ndarray, query_emb: dict[str, dict[str, np.ndarray]]):
        self.E = doc_emb  # rows aligned with the store
        self.Q = query_emb  # encoder name -> question -> vector

    def scores(self, question: str, encoder: str) -> np.ndarray:
        return self.E @ self.Q[encoder][question]

    def rank(self, question: str, n: int, encoder: str, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(question, encoder).astype(np.float64)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return [(i, float(s[i])) for i in _top(s, n)]
