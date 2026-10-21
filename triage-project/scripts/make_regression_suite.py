"""Regression suite for the deployed policy (few-shot LLM + calibrated threshold).

Saves 40 cases: 15 test issues the policy auto-labels correctly (must stay auto-labeled and correct),
15 test issues the model gets wrong but hands to a human (must not be auto-labeled), and 10 probe
issues whose non-native rewrite keeps the label (must keep it). The calibration parameters used by
the replay test are frozen into the suite.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from triage.selective import fit_temperature, label_index, logit_matrix, risk_threshold, softmax, split_fit_cert

R = defaultdict(dict)
for line in Path("results/llm_Qwen2.5-7B-Instruct-4bit.jsonl").read_text().splitlines():
    r = json.loads(line)
    R[(r["approach"], r["condition"])][r["id"]] = r
fit, cert = split_fit_cert(list(R[("few", "calib")].values()))
T = fit_temperature(logit_matrix(fit), label_index(fit))
pc = softmax(logit_matrix(cert), T)
tau = risk_threshold(pc.max(1), pc.argmax(1) == label_index(cert))


def conf(r):
    return float(softmax(logit_matrix([r]), T)[0].max())


rng = random.Random(544)
test = list(R[("few", "test")].values())
ok_auto = [r for r in test if conf(r) >= tau and r["pred"] == r["label"]]
wrong_deferred = [r for r in test if conf(r) < tau and r["pred"] != r["label"]]
stable = [v for v in R[("few", "variants")].values() if v["kind"] == "non_native" and v["applicable"]
          and v["pred"] == R[("few", "test")][v["base_id"]]["pred"] == v["label"]]
cases = ([dict(id=r["id"], kind="auto_correct", label=r["label"]) for r in rng.sample(ok_auto, 15)] +
         [dict(id=r["id"], kind="must_defer", label=r["label"]) for r in rng.sample(wrong_deferred, min(15, len(wrong_deferred)))] +
         [dict(id=v["id"], base_id=v["base_id"], kind="stable_rewording", label=v["label"]) for v in rng.sample(stable, min(10, len(stable)))])
Path("tests/regression_cases.json").write_text(json.dumps(dict(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit", temperature=T, threshold=tau, k=6, cases=cases), indent=1))
print(len(cases), "cases; T =", round(T, 3), "tau =", round(tau, 4))
