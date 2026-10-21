"""Build data/dataset.json from the HotpotQA distractor validation parquet."""
import argparse
import json
from pathlib import Path

from ragrepair.data import build

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default="data/raw/distractor/validation-00000-of-00001.parquet")
ap.add_argument("--out", default="data/dataset.json")
ap.add_argument("--pool", type=int, default=1200)
ap.add_argument("--dev", type=int, default=30)
ap.add_argument("--test", type=int, default=150)
ap.add_argument("--exclude", default="archive/v1/dataset.json", help="dataset whose dev/test ids must not be reused")
args = ap.parse_args()
excl = set()
if Path(args.exclude).exists():
    old = json.loads(Path(args.exclude).read_text())
    excl = {x["id"] for x in old["dev"] + old["test"]}
d = build(args.parquet, args.out, pool_size=args.pool, n_dev=args.dev, n_test=args.test, exclude_ids=excl)
print(json.dumps(d["meta"], indent=1), "dev", len(d["dev"]), "test", len(d["test"]))
