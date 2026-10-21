"""Create meaning-preserving variants of the probe issues; writes data/variants.json.

The non-native rewrite uses a different model family (Llama-3.2-3B) than the classifier under test
(Qwen2.5-7B), so the classifier never sees its own writing style.
"""
import argparse
import json
import re
from pathlib import Path

from triage.data import clean_body, load
from triage.llm import Cache, MLXModel
from triage.perturb import REWRITE_PROMPT, mask_code, newcomer, no_code, no_template, unmask_code, valid_rewrite

ap = argparse.ArgumentParser()
ap.add_argument("--rewriter", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
args = ap.parse_args()

ds = load()
test = {x["id"]: x for x in ds["test"]}
probe = [test[i] for i in ds["probe_ids"]]
rw = MLXModel(args.rewriter, Cache("cache/llm_cache.sqlite"))
variants, stats = [], {"invalid_rewrite": 0}
for n, it in enumerate(probe, 1):
    body = it["body"]
    masked, blocks = mask_code(body)
    msgs = [{"role": "user", "content": REWRITE_PROMPT.format(body=masked)}]
    ok, attempts = False, 0
    for temp, seed in ((0.0, 0), (0.7, 1), (0.7, 2), (0.7, 3)):  # greedy first, then up to three samples
        attempts += 1
        out = rw.generate(msgs, max_tokens=700, temperature=temp, seed=seed)["text"].strip()
        out = re.sub(r"^(here is (the |a )?rewritten[^:\n]*:|(the )?rewritten (issue )?text\s*:|issue text\s*:)\s*", "",
                     out, flags=re.I).strip()  # drop echoed preambles
        rewritten = unmask_code(out, blocks)
        ok = valid_rewrite(body, rewritten)
        if ok:
            break
    stats.setdefault("attempts", []).append(attempts)
    if not ok:
        stats["invalid_rewrite"] += 1
    kinds = {"no_code": no_code(body), "no_template": no_template(body), "newcomer": newcomer(body)}
    if ok:
        nn = clean_body(rewritten)
        kinds["non_native"] = nn
        kinds["non_native+no_code"] = no_code(nn)
    for kind, new_body in kinds.items():
        variants.append(dict(it, id=f"{it['id']}::{kind}", base_id=it["id"], kind=kind, body=new_body,
                             applicable=new_body.strip() != body.strip()))
    if n % 30 == 0:
        print(n, "of", len(probe), stats, flush=True)
att = stats.pop("attempts")
stats.update(rewrites=len(att), valid=len(att) - stats["invalid_rewrite"], valid_first_try=sum(a == 1 for a in att) - 0)
Path("data/variants.json").write_text(json.dumps(dict(stats=stats, variants=variants), indent=1))
from collections import Counter  # noqa: E402

print(stats, Counter((v["kind"], v["applicable"]) for v in variants))
