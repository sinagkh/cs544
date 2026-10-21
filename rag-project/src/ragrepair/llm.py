"""LLM backends with a persistent response cache.

The main backend runs a local 4-bit model on the Apple GPU through MLX. Greedy decoding is
deterministic, so identical prompts are answered from the cache and shared across policies.

`max_prompt_tokens` simulates a serving runtime with a small context window: a prompt longer than
the window is cut to its last `max_prompt_tokens` tokens before generation, and the result reports
how many tokens were actually processed, as serving APIs do (e.g. `prompt_eval_count`).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenResult:
    text: str
    prompt_tokens: int  # tokens the server processed
    completion_tokens: int
    seconds: float
    cached: bool
    sent_tokens: int = 0  # tokens the application sent
    processed_text: str = ""  # evaluation only: what survived a context-window cut


class ResponseCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, pt INTEGER, ct INTEGER, secs REAL)"
        )
        self.db.commit()

    def get(self, key: str):
        return self.db.execute("SELECT text, pt, ct, secs FROM cache WHERE key=?", (key,)).fetchone()

    def put(self, key: str, text: str, pt: int, ct: int, secs: float):
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (key, text, pt, ct, secs))
        self.db.commit()


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# Llama 3.x chat templates insert today's date into the prompt. Pin it so prompts (and cache keys)
# do not change from day to day; all recorded runs used this date.
PINNED_DATE = "20 Oct 2026"


class CacheMiss(RuntimeError):
    pass


class MLXModel:
    def __init__(self, repo: str, cache: ResponseCache, replay_only: bool = False):
        """`replay_only` serves cached responses and raises CacheMiss otherwise (for CI)."""
        from transformers import AutoTokenizer

        self.repo, self.cache, self.replay_only = repo, cache, replay_only
        if replay_only:
            from huggingface_hub import snapshot_download

            self.model, self.tok = None, AutoTokenizer.from_pretrained(snapshot_download(repo, allow_patterns=["*.json", "*.model", "*.txt"]))
        else:
            from mlx_lm import load

            self.model, self.tok = load(repo)

    def prompt_text(self, messages: list[dict]) -> str:
        try:
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, date_string=PINNED_DATE)
        except Exception:
            # Templates without a system role (e.g. Gemma): fold the system text into the first user turn.
            sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            rest = [dict(m) for m in messages if m["role"] != "system"]
            if sys_txt and rest:
                rest[0]["content"] = f"{sys_txt}\n\n{rest[0]['content']}"
            return self.tok.apply_chat_template(rest, add_generation_prompt=True, tokenize=False, date_string=PINNED_DATE)

    def encode(self, prompt: str) -> list[int]:
        bos = getattr(self.tok, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        return list(self.tok.encode(prompt, add_special_tokens=add_special))

    def count_tokens(self, messages: list[dict]) -> int:
        return len(self.encode(self.prompt_text(messages)))

    def generate(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0,
                 seed: int = 0, max_prompt_tokens: int | None = None) -> GenResult:
        prompt = self.prompt_text(messages)
        ids = self.encode(prompt)
        sent = len(ids)
        cut = max_prompt_tokens is not None and sent > max_prompt_tokens
        if cut:
            ids = ids[-max_prompt_tokens:]
        key = _key(self.repo, prompt, max_tokens, temperature, seed) if not cut else \
            _key(self.repo, prompt, max_tokens, temperature, seed, "window", max_prompt_tokens)
        processed_text = self.tok.decode(ids) if cut else prompt
        hit = self.cache.get(key)
        if hit:
            return GenResult(hit[0], hit[1], hit[2], hit[3], True, sent, processed_text)
        if self.replay_only:
            raise CacheMiss(f"no cached response for prompt (sha {key[:12]})")
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        t0 = time.time()
        text = generate(self.model, self.tok, prompt=ids, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temperature), verbose=False)
        secs = time.time() - t0
        pt, ct = len(ids), len(self.tok.encode(text))
        self.cache.put(key, text, pt, ct, secs)
        return GenResult(text, pt, ct, secs, False, sent, processed_text)


class ClaudeCLIModel:
    """Optional backend using the logged-in Claude Code CLI (subscription usage).

    No temperature or seed control is available, so results are cached per prompt.
    """

    def __init__(self, model: str, cache: ResponseCache):
        self.repo, self.cache = f"claude-cli:{model}", cache
        self.model = model

    def count_tokens(self, messages: list[dict]) -> int:
        return sum(len(m["content"]) for m in messages) // 4

    def _parts(self, messages):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        return system, user

    def cached(self, messages, seed: int = 0) -> bool:
        system, user = self._parts(messages)
        return self.cache.get(_key(self.repo, system, user, seed)) is not None

    def call(self, messages) -> tuple[str, int, int, float]:
        """Run the CLI without touching the cache (safe to call from worker threads)."""
        system, user = self._parts(messages)
        t0 = time.time()
        proc = subprocess.run(
            ["claude", "-p", user, "--system-prompt", system, "--tools", "", "--strict-mcp-config",
             "--no-session-persistence", "--model", self.model, "--output-format", "json"],
            capture_output=True, text=True, timeout=300,
        )
        out = json.loads(proc.stdout)
        u = out.get("usage", {})
        return out.get("result", ""), u.get("input_tokens", 0), u.get("output_tokens", 0), time.time() - t0

    def store(self, messages, raw, seed: int = 0):
        system, user = self._parts(messages)
        self.cache.put(_key(self.repo, system, user, seed), *raw)

    def generate(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0,
                 seed: int = 0, max_prompt_tokens: int | None = None) -> GenResult:
        system, user = self._parts(messages)
        key = _key(self.repo, system, user, seed)
        hit = self.cache.get(key)
        if hit:
            return GenResult(hit[0], hit[1], hit[2], hit[3], True, hit[1], user)
        t0 = time.time()
        proc = subprocess.run(
            ["claude", "-p", user, "--system-prompt", system, "--tools", "", "--strict-mcp-config",
             "--no-session-persistence", "--model", self.model, "--output-format", "json"],
            capture_output=True, text=True, timeout=300,
        )
        out = json.loads(proc.stdout)
        usage = out.get("usage", {})
        text = out.get("result", "")
        secs = time.time() - t0
        self.cache.put(key, text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), secs)
        return GenResult(text, usage.get("input_tokens", 0), usage.get("output_tokens", 0), secs, False,
                         usage.get("input_tokens", 0), user)
