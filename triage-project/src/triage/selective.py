"""Calibration and selective automation: when to auto-label and when to hand the issue to a human."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

from .data import LABELS

TARGET_RISK = 0.10  # at most 10% of auto-labeled issues may be wrong
ALPHA = 0.10  # conformal miscoverage


def logit_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[r["logits"][l] for l in LABELS] for r in rows], dtype=np.float64)


def label_index(rows: list[dict]) -> np.ndarray:
    return np.array([LABELS.index(r["label"]) for r in rows])


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017) fit by NLL on the calibration split."""
    def nll(logT):
        p = softmax(z, math.exp(logT))
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
    return float(math.exp(minimize_scalar(nll, bounds=(-4, 6), method="bounded").x))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def risk_threshold(conf: np.ndarray, correct: np.ndarray, target: float = TARGET_RISK, min_auto: int = 30) -> float:
    """Lowest confidence threshold whose auto-labeled set on calibration data has error <= target."""
    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    errs = np.cumsum(~ok)
    n = np.arange(1, len(c) + 1)
    feasible = np.where((errs / n <= target) & (n >= min_auto))[0]
    if not len(feasible):
        return float("inf")
    i = feasible.max()
    return float(c[i])


def ltt_threshold(conf: np.ndarray, correct: np.ndarray, target: float = TARGET_RISK, delta: float = 0.05,
                  min_auto: int = 30) -> float:
    """Risk-controlling threshold (Learn-then-Test with fixed-sequence testing; Angelopoulos et al. 2021).

    Thresholds are tested from the most to the least confident; each is accepted while the
    Clopper-Pearson upper (1 - delta) bound on the error among auto-labeled calibration issues stays
    at or below `target`, and testing stops at the first failure. With probability 1 - delta the
    chosen threshold keeps the error at or below `target` on exchangeable data.
    """
    from scipy.stats import beta

    order = np.argsort(-conf)
    c, ok = conf[order], correct[order]
    errs = np.cumsum(~ok)
    best = float("inf")
    for i in range(len(c)):
        n, k = i + 1, int(errs[i])
        if i + 1 < len(c) and c[i + 1] == c[i]:
            continue  # only test at distinct confidence values
        ucb = 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))
        if n >= min_auto and ucb > target:
            break
        if n >= min_auto:
            best = float(c[i])
    return best


def conformal_qhat(p: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """Split conformal with the LAC score 1 - p(true label)."""
    s = 1 - p[np.arange(len(y)), y]
    n = len(s)
    k = min(n, int(math.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(s)[k - 1])


def conformal_sets(p: np.ndarray, qhat: float) -> np.ndarray:
    return (1 - p) <= qhat


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    f = []
    for c in range(len(LABELS)):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def selective_metrics(auto: np.ndarray, correct: np.ndarray) -> dict:
    cov = float(auto.mean())
    risk = float((~correct[auto]).mean()) if auto.any() else float("nan")
    return dict(coverage=cov, risk=risk, auto_errors_per_100=float((auto & ~correct).mean() * 100))
