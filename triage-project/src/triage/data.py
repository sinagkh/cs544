"""Load NLBSE'24 and fresh 2025-26 issues, clean them, and build the evaluation splits."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

LABELS = ["bug", "feature", "question"]
BODY_CHARS = 1500
REPO_LABELS = {
    "facebook/react": {"bug": "Type: Bug", "feature": "Type: Feature Request", "question": "Type: Question"},
    "tensorflow/tensorflow": {"bug": "type:bug", "feature": "type:feature", "question": "type:support"},
    "microsoft/vscode": {"bug": "bug", "feature": "feature-request", "question": "*question"},
    "opencv/opencv": {"bug": "bug", "feature": "feature", "question": "question (invalid tracker)"},
    "bitcoin/bitcoin": {"bug": "Bug", "feature": "Feature", "question": "Questions and Help"},
}
REPOS = list(REPO_LABELS)

_COMMENT = re.compile(r"<!--.*?-->", re.S)


def clean_body(body: str | None, limit: int = BODY_CHARS) -> str:
    """Deployed preprocessing: drop HTML comments left by issue templates, squash blank lines, truncate."""
    b = _COMMENT.sub("", body or "")
    b = re.sub(r"\r\n?", "\n", b)
    b = re.sub(r"\n{3,}", "\n\n", b).strip()
    return b[:limit]


def _item(source, idx, repo, created_at, label, title, body):
    return dict(id=f"{source}-{idx}", source=source, repo=repo, created_at=str(created_at), label=label,
                title=str(title).strip(), body=clean_body(body if isinstance(body, str) else ""))


def load_nlbse(raw_dir: Path, split: str) -> list[dict]:
    df = pd.read_csv(raw_dir / "nlbse24" / f"issues_{split}.csv")
    return [_item(f"nlbse-{split}", i, r.repo, r.created_at, r.label, r.title, r.body) for i, r in enumerate(df.itertuples())]


def load_fresh(raw_dir: Path) -> list[dict]:
    out = []
    for line in (raw_dir / "github" / "fresh_issues.jsonl").read_text().splitlines():
        r = json.loads(line)
        types = [k for k, v in REPO_LABELS[r["repo"]].items() if v in r["labels"]]
        if len(types) != 1:
            continue  # issues with more than one type label are ambiguous, as in NLBSE
        out.append(_item("fresh", f"{r['repo'].replace('/', '_')}-{r['number']}", r["repo"], r["created_at"],
                         types[0], r["title"], r["body"]))
    return out


def _balanced(items: list[dict], per_cell: int, rng: random.Random) -> list[dict]:
    cells = defaultdict(list)
    for it in items:
        cells[(it["repo"], it["label"])].append(it)
    out = []
    for k in sorted(cells):
        c = list(cells[k])
        rng.shuffle(c)
        out += c[:per_cell]
    return out


def build(raw_dir: str | Path = "data/raw", out_path: str | Path = "data/dataset.json", seed: int = 544) -> dict:
    raw_dir = Path(raw_dir)
    rng = random.Random(seed)
    train, test, fresh = load_nlbse(raw_dir, "train"), load_nlbse(raw_dir, "test"), load_fresh(raw_dir)
    # Calibration: about one third of the training half, balanced; the rest is the labeled history.
    calib = _balanced(train, 33, rng)
    cal_ids = {x["id"] for x in calib}
    history = [x for x in train if x["id"] not in cal_ids]
    fresh_bal = _balanced(fresh, 40, rng)
    newrepo = _balanced(test, 50, rng)  # 50 per (repo, label) = 750
    probe = _balanced(test, 14, rng)  # 14 per (repo, label) = 210, for rewording and explanations
    ds = dict(meta=dict(seed=seed, history=len(history), calib=len(calib), test=len(test), fresh_all=len(fresh),
                        fresh=len(fresh_bal), newrepo=len(newrepo), probe=len(probe), body_chars=BODY_CHARS),
              history=history, calib=calib, test=test, fresh=fresh_bal, fresh_all=fresh,
              newrepo_ids=[x["id"] for x in newrepo], probe_ids=[x["id"] for x in probe])
    Path(out_path).write_text(json.dumps(ds, indent=1))
    return ds


def load(path: str | Path = "data/dataset.json") -> dict:
    return json.loads(Path(path).read_text())
