#!/usr/bin/env python3
"""Fourth-round requirement 3: I4 calibration invariance on the SAME scores.

Take one frozen score vector z_test. Apply (a) validation-frozen calibration
h_val and (b) test-fitted calibration h_test. PR-AUC/ROC-AUC are invariant to
any strictly monotone map, so they must be bit-identical; only ECE/Brier change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v5" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
n = 20000
# Imbalanced score vector mimicking a test window (base rate ~2.5%).
y = (rng.random(n) < 0.025).astype(int)
z = rng.normal(loc=np.where(y == 1, 1.0, -0.5), scale=1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(zval, yval):
    from scipy.optimize import minimize_scalar
    def loss(t):
        p = sigmoid(t * zval)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(yval * np.log(p) + (1 - yval) * np.log(1 - p))
    r = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


# Validation window has a different positive rate (8%) than test (2.5%),
# so the two fitted temperatures differ substantially (not a near-identity check).
yval = (rng.random(4000) < 0.08).astype(int)
zval = rng.normal(loc=np.where(yval == 1, 1.6, -0.4), scale=1.0)
t_val = fit_temperature(zval, yval)
t_test = fit_temperature(z, y)  # illegal: uses test labels
p_val = sigmoid(t_val * z)
p_test = sigmoid(t_test * z)

p_raw = sigmoid(z)  # no temperature scaling (identity on logits)
res = {
    "t_val": t_val, "t_test": t_test,
    "ece_raw": ece(y, p_raw), "brier_raw": float(np.mean((p_raw - y) ** 2)),
    "pr_auc_h_val": float(average_precision_score(y, p_val)),
    "pr_auc_h_test": float(average_precision_score(y, p_test)),
    "roc_auc_h_val": float(roc_auc_score(y, p_val)),
    "roc_auc_h_test": float(roc_auc_score(y, p_test)),
    "ece_h_val": ece(y, p_val),
    "ece_h_test": ece(y, p_test),
    "brier_h_val": float(np.mean((p_val - y) ** 2)),
    "brier_h_test": float(np.mean((p_test - y) ** 2)),
}
res["pr_auc_identical"] = abs(res["pr_auc_h_val"] - res["pr_auc_h_test"]) < 1e-12
res["roc_auc_identical"] = abs(res["roc_auc_h_val"] - res["roc_auc_h_test"]) < 1e-12
(OUT / "i4_same_scores.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}, indent=2))
