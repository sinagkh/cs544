"""Grounding check with MiniCheck (Tang et al., EMNLP 2024): is a claim supported by the passages?

The rendered passages are packed in rank order into chunks of at most `CHUNK_TOKENS` tokens, each
chunk is scored against the claim, and the support score is the maximum over chunks. Packing keeps
neighboring passages together, which matters for two-hop questions whose support spans two
passages. Scores are cached in SQLite, keyed by chunk text and claim.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

DEFAULT_MODEL = "lytang/MiniCheck-RoBERTa-Large"
CHUNK_TOKENS = 400


def make_claim(question: str, answer: str) -> str:
    # Chosen on prototype dev traces among three formats (AUROC 0.77 vs. 0.68-0.74).
    return f"{question.strip()} {answer.strip()}"


class SupportChecker:
    def __init__(self, repo: str = DEFAULT_MODEL, cache_path: str | Path = "cache/support_cache.sqlite",
                 threshold: float = 0.5, replay_only: bool = False):
        self.repo, self.threshold, self.replay_only = repo, threshold, replay_only
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path))
        self.db.execute("CREATE TABLE IF NOT EXISTS s (key TEXT PRIMARY KEY, score REAL)")
        self.db.commit()
        self._model = None
        self.seconds = 0.0
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo)

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification

            torch.set_num_threads(2)
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = AutoModelForSequenceClassification.from_pretrained(self.repo).to(self.device).eval()

    def chunks(self, passages: list[str]) -> list[str]:
        out, cur, n = [], [], 0
        for p in passages:
            t = len(self.tok.encode(p, add_special_tokens=False))
            if cur and n + t > CHUNK_TOKENS:
                out.append("\n".join(cur))
                cur, n = [], 0
            cur.append(p)
            n += t
        if cur:
            out.append("\n".join(cur))
        return out

    def _key(self, chunk: str, claim: str) -> str:
        return hashlib.sha256(f"{self.repo}\x1f{chunk}\x1f{claim}".encode()).hexdigest()

    def score(self, passages: list[str], claim: str) -> float:
        chunks = self.chunks([p for p in passages if p.strip()]) or [""]
        scores, todo = {}, []
        for c in chunks:
            hit = self.db.execute("SELECT score FROM s WHERE key=?", (self._key(c, claim),)).fetchone()
            if hit:
                scores[c] = hit[0]
            else:
                todo.append(c)
        if todo:
            if self.replay_only:
                raise RuntimeError("support score not cached")
            import torch

            self._load()
            t0 = time.time()
            enc = self.tok(todo, [claim] * len(todo), truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1)[:, 1].cpu().tolist()
            self.seconds += time.time() - t0
            for c, sc in zip(todo, probs):
                scores[c] = sc
                self.db.execute("INSERT OR REPLACE INTO s VALUES (?,?)", (self._key(c, claim), sc))
            self.db.commit()
        return max(scores.values())

    def supported(self, passages: list[str], question: str, answer: str) -> tuple[bool, float]:
        s = self.score(passages, make_claim(question, answer))
        return s >= self.threshold, s
