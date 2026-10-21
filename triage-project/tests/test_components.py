"""Unit tests (no model needed)."""

import numpy as np
import pytest

from triage.data import clean_body
from triage.perturb import mask_code, newcomer, no_code, no_template, unmask_code, valid_rewrite
from triage.prompts import parse_explanation
from triage.selective import (conformal_qhat, conformal_sets, ece, fit_temperature, ltt_fixed_sequence, macro_f1,
                              risk_threshold, softmax, split_fit_cert, threshold_grid)

BODY = """### Describe the bug
The app crashes on start.

### Steps
- [x] I searched existing issues
1. run `npm start`

```
TypeError: Cannot read properties of undefined
    at render (index.js:10)
```
Expected it to render."""


def test_clean_body_strips_comments_and_truncates():
    b = clean_body("<!-- template help -->\n\n\n\nreal text", limit=5)
    assert b == "real "


def test_no_code_removes_fences_and_traces():
    out = no_code(BODY)
    assert "TypeError" not in out and "```" not in out and "crashes on start" in out


def test_no_template_removes_headings_and_checkboxes():
    out = no_template(BODY)
    assert "### Describe" not in out and "[x]" not in out and "crashes on start" in out


def test_newcomer_prefix():
    assert newcomer("x").endswith("x") and "new to this project" in newcomer("x")


def test_mask_roundtrip_and_validity():
    masked, blocks = mask_code(BODY)
    assert "[CODE0]" in masked and len(blocks) == 1
    assert unmask_code(masked, blocks) == BODY
    assert unmask_code("lost the placeholder", blocks) is None
    assert valid_rewrite(BODY, BODY)
    assert not valid_rewrite(BODY, "short")


def test_parse_explanation():
    lab, q = parse_explanation('Label: Bug\nEvidence: "app crashes"; "TypeError: Cannot read"')
    assert lab == "bug" and q == ["app crashes", "TypeError: Cannot read"]


def test_temperature_and_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    z = rng.normal(0, 1, (500, 3))
    z[np.arange(500), y] += 2.0
    T = fit_temperature(z * 5, y)  # overconfident logits need T > 1
    assert T > 1
    assert ece(softmax(z * 5, T), y) < ece(softmax(z * 5), y)


def test_risk_threshold_and_conformal():
    conf = np.linspace(1, 0, 100)
    correct = np.array([True] * 60 + [False] * 40)
    tau = risk_threshold(conf, correct, target=0.1, min_auto=10)
    auto = conf >= tau
    assert (~correct[auto]).mean() <= 0.1 and auto.sum() >= 60
    p = np.array([[0.9, 0.05, 0.05]] * 50 + [[0.4, 0.35, 0.25]] * 50)
    y = np.zeros(100, int)
    q = conformal_qhat(p, y, alpha=0.1)
    S = conformal_sets(p, q)
    assert S[np.arange(100), y].mean() >= 0.9


def test_macro_f1():
    y = np.array([0, 1, 2, 0])
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_split_fit_cert_is_balanced_and_disjoint():
    rows = [dict(id=f"x{i}", repo=f"r{i % 2}", label=["bug", "feature", "question"][i % 3]) for i in range(60)]
    fit, cert = split_fit_cert(rows)
    assert len(fit) + len(cert) == 60 and not ({r["id"] for r in fit} & {r["id"] for r in cert})
    assert 15 <= len(fit) <= 25


def test_ltt_fixed_sequence_controls_risk():
    rng = np.random.default_rng(1)
    conf = rng.random(3000)
    correct = rng.random(3000) < 0.4 + 0.6 * conf  # error falls with confidence
    grid = threshold_grid(conf[:1000])
    assert grid == sorted(grid, reverse=True)
    tau = ltt_fixed_sequence(conf[1000:2000], correct[1000:2000], grid, target=0.15)
    auto = conf[2000:] >= tau
    assert auto.any() and (~correct[2000:][auto]).mean() <= 0.18
    assert ltt_fixed_sequence(conf[1000:2000], correct[1000:2000], grid, target=0.001) == float("inf")
