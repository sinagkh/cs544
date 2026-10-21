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


def split_fit_cert(rows: list[dict], frac_fit: float = 1 / 3) -> tuple[list[dict], list[dict]]:
    """Deterministic split of calibration rows, balanced within (repo, label) cells.

    The fit part sets the temperature and the candidate threshold grid; the certification part
    is used only to test thresholds and compute conformal quantiles, so certification is done on
    data independent of everything that was tuned.
    """
    import hashlib
    from collections import defaultdict

    cells = defaultdict(list)
    for r in rows:
        cells[(r["repo"], r["label"])].append(r)
    fit, cert = [], []
    for k in sorted(cells):
        c = sorted(cells[k], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        n_fit = int(round(len(c) * frac_fit))
        fit += c[:n_fit]
        cert += c[n_fit:]
    return fit, cert


def threshold_grid(conf_fit: np.ndarray, start: float = 0.10, step: float = 0.025) -> list[float]:
    """Candidate thresholds from the fit split, at coverage levels start, start+step, ..., 1 (descending)."""
    levels = np.arange(start, 1.0 + 1e-9, step)
    return [float(np.quantile(conf_fit, 1 - c)) for c in levels]


def ltt_fixed_sequence(conf: np.ndarray, correct: np.ndarray, grid: list[float], target: float = TARGET_RISK,
                       delta: float = 0.05) -> float:
    """Learn-then-Test with fixed-sequence testing over a pre-specified grid (Angelopoulos et al.).

    For each threshold in order, test H0: risk > target with the exact binomial p-value
    P(Bin(n, target) <= k), where n issues on the certification split pass the threshold and k of
    them are wrong. Certify while p <= delta; stop at the first failure. Returns the lowest certified
    threshold, or inf if none is certified.
    """
    from scipy.stats import binom

    best = float("inf")
    for t in grid:
        auto = conf >= t
        n, k = int(auto.sum()), int((auto & ~correct).sum())
        if n == 0 or binom.cdf(k, n, target) > delta:
            break
        best = t
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
