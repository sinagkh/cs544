"""Embeddings, retrieval of similar past issues, and classical classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import LABELS

ENCODER = "BAAI/bge-small-en-v1.5"
_enc = None


def text_of(item: dict, chars: int = 1000) -> str:
    return f"{item['title']}. {item['body'][:chars]}"


def embed(items: list[dict], chars: int = 1000) -> np.ndarray:
    global _enc
    if _enc is None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)
        _enc = SentenceTransformer(ENCODER, device="mps" if torch.backends.mps.is_available() else "cpu")
    return _enc.encode([text_of(x, chars) for x in items], batch_size=32, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


class Retriever:
    """Top-k most similar labeled history issues, from the same repository or from the others."""

    def __init__(self, history: list[dict], H: np.ndarray):
        self.history, self.H = history, H
        self.repo = np.array([h["repo"] for h in history])

    def topk(self, q: np.ndarray, repo: str, k: int = 6, same_repo: bool = True) -> list[dict]:
        s = self.H @ q
        mask = (self.repo == repo) if same_repo else (self.repo != repo)
        s = np.where(mask, s, -np.inf)
        return [self.history[i] for i in np.argsort(-s)[:k]]


def tfidf_lr():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000),
                         LogisticRegression(max_iter=3000, C=4.0))


def emb_lr():
    return LogisticRegression(max_iter=3000, C=2.0)


def proba_dict(p: np.ndarray, classes) -> dict:
    m = dict(zip(classes, p))
    return {lab: float(m.get(lab, 0.0)) for lab in LABELS}
