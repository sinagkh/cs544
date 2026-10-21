"""Rewrite outdated-document values for entity answers with a local model (run after build_dataset.py)."""
import argparse
import json
from pathlib import Path

from ragrepair.data import load_dataset, refresh_entity_stale
from ragrepair.llm import MLXModel, ResponseCache

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="data/dataset.json")
ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
args = ap.parse_args()
ds = load_dataset(args.dataset)
m = MLXModel(args.model, ResponseCache("cache/llm_cache.sqlite"))
ds = refresh_entity_stale(ds, lambda msgs, t, s: m.generate(msgs, max_tokens=24, temperature=t, seed=s).text)
Path(args.dataset).write_text(json.dumps(ds, indent=1))
print(ds["meta"], "dev", len(ds["dev"]), "test", len(ds["test"]))
for q in ds["test"][:15]:
    print("-", q["answer"], "->", q["stale_answer"], f"[{q['stale_source']}]")
