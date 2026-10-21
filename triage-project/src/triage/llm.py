"""Local LLM backend (MLX) with a persistent cache.

`label_probs` reads the model's next-token distribution right after the prompt and returns the
probability mass on each label word, renormalized over the three labels. This is one prefill pass
per issue and gives a confidence score without asking the model to verbalize one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from .data import LABELS


class Cache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS c (key TEXT PRIMARY KEY, val TEXT)")
        self.db.commit()

    def get(self, key):
        r = self.db.execute("SELECT val FROM c WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key, val):
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, json.dumps(val)))
        self.db.commit()


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# Llama 3.x chat templates insert today's date into the prompt. Pin it so prompts (and cache keys)
# do not change from day to day; all recorded runs used this date.
PINNED_DATE = "20 Oct 2026"


class CacheMiss(RuntimeError):
    pass


class MLXModel:
    def __init__(self, repo: str, cache: Cache, replay_only: bool = False):
        self.repo, self.cache, self.replay_only = repo, cache, replay_only
        if replay_only:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            self.model = None
            self.tok = AutoTokenizer.from_pretrained(snapshot_download(repo, allow_patterns=["*.json", "*.model", "*.txt"]))
        else:
            from mlx_lm import load

            self.model, self.tok = load(repo)
        self.label_ids = {lab: self._first_token_ids(lab) for lab in LABELS}

    def _first_token_ids(self, word: str) -> list[int]:
        ids = set()
        for v in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            t = self.tok.encode(v, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    def prompt_text(self, messages: list[dict]) -> str:
        try:
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, date_string=PINNED_DATE)
        except Exception:  # templates without a system role
            sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            rest = [dict(m) for m in messages if m["role"] != "system"]
            if sys_txt and rest:
                rest[0]["content"] = f"{sys_txt}\n\n{rest[0]['content']}"
            return self.tok.apply_chat_template(rest, add_generation_prompt=True, tokenize=False, date_string=PINNED_DATE)

    def _ids(self, prompt: str) -> list[int]:
        bos = getattr(self.tok, "bos_token", None)
        return list(self.tok.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos)))

    def label_probs(self, messages: list[dict]) -> dict:
        prompt = self.prompt_text(messages)
        key = _key(self.repo, "probs", prompt)
        hit = self.cache.get(key)
        if hit:
            return dict(hit, cached=True)
        if self.replay_only:
            raise CacheMiss(key[:12])
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        ids = self._ids(prompt)
        t0 = time.time()
        resp = next(iter(stream_generate(self.model, self.tok, prompt=ids, max_tokens=1, sampler=make_sampler(temp=0.0))))
        import mlx.core as mx

        lp = np.array(resp.logprobs.astype(mx.float32), dtype=np.float64)
        from scipy.special import logsumexp

        logmass = {lab: float(logsumexp(lp[self.label_ids[lab]])) for lab in LABELS}
        z = logsumexp(list(logmass.values()))
        val = dict(logits={k: v - z for k, v in logmass.items()},  # log-probabilities renormalized over labels
                   probs={k: float(np.exp(v - z)) for k, v in logmass.items()}, label_mass=float(np.exp(z)),
                   prompt_tokens=len(ids), seconds=time.time() - t0)
        self.cache.put(key, val)
        return dict(val, cached=False)

    def generate(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0, seed: int = 0) -> dict:
        prompt = self.prompt_text(messages)
        key = _key(self.repo, "gen", prompt, max_tokens, temperature, seed)
        hit = self.cache.get(key)
        if hit:
            return dict(hit, cached=True)
        if self.replay_only:
            raise CacheMiss(key[:12])
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        ids = self._ids(prompt)
        t0 = time.time()
        text = generate(self.model, self.tok, prompt=ids, max_tokens=max_tokens, sampler=make_sampler(temp=temperature), verbose=False)
        val = dict(text=text, prompt_tokens=len(ids), seconds=time.time() - t0)
        self.cache.put(key, val)
        return dict(val, cached=False)
