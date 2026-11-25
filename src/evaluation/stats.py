"""Paired inference. Bootstrap, Wilcoxon and McNemar answer different questions."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, wilcoxon


def paired_bootstrap_metric_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    y = np.asarray(y)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    observed = metric_fn(y, a) - metric_fn(y, b)
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx]))
    if not deltas:
        return {"observed": float(observed), "ci_low": float("nan"), "ci_high": float("nan")}
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"observed": float(observed), "ci_low": float(low), "ci_high": float(high)}


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > 1e-12)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(wilcoxon(a[mask], b[mask]).pvalue)
    except Exception:
        return float("nan")


def mcnemar_paired(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    correct_a, correct_b = a == y, b == y
    n01 = int((~correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm_adjust(pvalues) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(values)
    out = np.ones_like(values)
    idx = np.flatnonzero(finite)
    if len(idx) == 0:
        return out
    order = idx[np.argsort(values[idx])]
    running = 0.0
    for rank, position in enumerate(order, start=1):
        running = max(running, (len(order) - rank + 1) * values[position])
        out[position] = min(1.0, running)
    return out
