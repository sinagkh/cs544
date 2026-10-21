"""Sentence-embedding helpers (run once, offline; results are stored in data/embeddings.npz)."""

from __future__ import annotations

import numpy as np

INDEX_ENCODER = "BAAI/bge-small-en-v1.5"
# The fault: the query service silently falls back to a different library-default encoder.
MISMATCH_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_models: dict = {}


def _model(name: str):
    if name not in _models:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)  # keep CPU use low; encoding runs on the Apple GPU (MPS)
        _models[name] = SentenceTransformer(name, device="mps" if torch.backends.mps.is_available() else "cpu")
    return _models[name]


def doc_text(d: dict) -> str:
    return f"{d['title']}. {d['text']}"


def embed_docs(docs: list[dict], name: str = INDEX_ENCODER) -> np.ndarray:
    return _model(name).encode([doc_text(d) for d in docs], batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def embed_queries(questions: list[str], name: str) -> np.ndarray:
    prefix = BGE_QUERY_PREFIX if "bge" in name else ""
    return _model(name).encode([prefix + q for q in questions], batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
