"""The document store and its two indexes, loaded once per experiment run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embed import INDEX_ENCODER, MISMATCH_ENCODER
from .retriever import BM25Retriever, DenseRetriever


class Store:
    """Corpus passages plus the outdated versions used by the `stale` fault.

    `data/embeddings.npz` (scripts/embed_store.py) holds index embeddings for every stored doc and
    query embeddings for every dev/test question under both the index encoder and the mismatched
    encoder. `data/thresholds.json` holds the retrieval-confidence threshold calibrated on
    held-out questions.
    """

    def __init__(self, ds: dict, data_dir: str | Path = "data"):
        data_dir = Path(data_dir)
        cases = ds["dev"] + ds["test"]
        self.docs = list(ds["corpus"]) + [q["stale_doc"] for q in cases]
        self.pos = {d["doc_id"]: i for i, d in enumerate(self.docs)}
        z = np.load(data_dir / "embeddings.npz", allow_pickle=False)
        assert list(z["doc_ids"]) == [d["doc_id"] for d in self.docs], "embeddings out of date; rerun embed_store.py"
        qtexts = list(z["questions"])
        self.dense = DenseRetriever(z["E"], {
            INDEX_ENCODER: dict(zip(qtexts, z["Q_index"])),
            MISMATCH_ENCODER: dict(zip(qtexts, z["Q_mismatch"])),
        })
        self.bm25 = BM25Retriever(self.docs)
        self.base_mask = np.array([not d.get("stale", False) for d in self.docs])
        self.thresholds = json.loads((data_dir / "thresholds.json").read_text())
