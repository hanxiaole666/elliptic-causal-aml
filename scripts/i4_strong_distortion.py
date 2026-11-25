#!/usr/bin/env python3
"""C1 (round 16): I4 same-score check with STRONG monotone distortion.

The previous implementation fitted temperatures T_val=0.968 and T_test=1.003
on a synthetic vector; both are near-identity, so ECE moved by only 0.002 and
Brier did not move at all. That made the manuscript's claim "ECE and Brier move"
self-contradictory with its own table.

This version keeps the frozen score vector and the ranking metrics, but uses
two deliberately strong distortions of the SAME strictly monotone family:
  identity : p = sigmoid(z)                 (no scaling)
  h_val    : p = sigmoid(0.5 * z)            (T=0.5, over-confident)
  h_test   : p = sigmoid(2.0 * z)            (T=2.0, under-confident; illegal test fit)
Both are strictly increasing in z, so PR-AUC and ROC-AUC must be bit-identical,
while ECE and Brier change substantially. This is an IMPLEMENTATION CHECK of
I4's monotonicity -- not a claim that T=0.5/2.0 are optimal temperatures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v6" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
n = 20000
y = (rng.random(n) < 0.025).astype(int)
z = rng.normal(loc=np.where(y == 1, 1.0, -0.5), scale=1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def row(p):
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "ece": ece(y, p),
        "brier": float(np.mean((p - y) ** 2)),
    }


res = {
    "design": "same frozen score vector; identity vs T=0.5 (val-frozen) vs T=2.0 (illegal test fit)",
    "note": "implementation check of monotone invariance; T values are deliberately strong distortions, not fitted optima",
    "n": n, "base_rate": float(y.mean()),
    "identity": row(sigmoid(z)),
    "h_val_T05": row(sigmoid(0.5 * z)),
    "h_test_T20": row(sigmoid(2.0 * z)),
}
p0, p1, p2 = sigmoid(z), sigmoid(0.5 * z), sigmoid(2.0 * z)
res["pr_auc_bit_identical"] = (
    res["identity"]["pr_auc"] == res["h_val_T05"]["pr_auc"] == res["h_test_T20"]["pr_auc"]
)
res["roc_auc_bit_identical"] = (
    res["identity"]["roc_auc"] == res["h_val_T05"]["roc_auc"] == res["h_test_T20"]["roc_auc"]
)
res["rank_order_identical"] = bool(
    np.array_equal(np.argsort(p0), np.argsort(p1)) and np.array_equal(np.argsort(p0), np.argsort(p2))
)
(OUT / "i4_strong_distortion.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k in (
    "design", "identity", "h_val_T05", "h_test_T20",
    "pr_auc_bit_identical", "roc_auc_bit_identical", "rank_order_identical")}, indent=2))
