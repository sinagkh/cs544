"""Embed the document store and all dev/test questions; calibrate the retrieval-confidence threshold.

Run after build_dataset.py and make_stale_values.py. Writes data/embeddings.npz and
data/thresholds.json. The threshold is the 5th percentile of the clean top-1 similarity on
held-out questions that are in neither dev nor test, so no fault data is used to set it.
"""
import json
from pathlib import Path

import numpy as np

from ragrepair.data import load_dataset
from ragrepair.embed import INDEX_ENCODER, MISMATCH_ENCODER, embed_docs, embed_queries

ds = load_dataset("data/dataset.json")
cases = ds["dev"] + ds["test"]
E_corpus = np.load("data/corpus_emb.npy")
assert len(E_corpus) == len(ds["corpus"])
E_stale = embed_docs([q["stale_doc"] for q in cases], INDEX_ENCODER)
E = np.vstack([E_corpus, E_stale]).astype(np.float32)
doc_ids = [d["doc_id"] for d in ds["corpus"]] + [q["stale_doc"]["doc_id"] for q in cases]
questions = [q["question"] for q in cases]
Q_index = embed_queries(questions, INDEX_ENCODER)
Q_mismatch = embed_queries(questions, MISMATCH_ENCODER)
np.savez("data/embeddings.npz", doc_ids=np.array(doc_ids), E=E, questions=np.array(questions),
         Q_index=Q_index, Q_mismatch=Q_mismatch)

used = {q["id"] for q in cases}
held = [q for q in ds["selected"] if q["id"] not in used]
Qh = embed_queries([q["question"] for q in held], INDEX_ENCODER)
top1 = (Qh @ E_corpus.T).max(axis=1)
thr = dict(retrieval_top1_p05=float(np.percentile(top1, 5)), calibration_questions=len(held),
           clean_top1_median=float(np.median(top1)))
Path("data/thresholds.json").write_text(json.dumps(thr, indent=1))
print(len(doc_ids), "docs;", len(questions), "questions;", thr)
