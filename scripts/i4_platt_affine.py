#!/usr/bin/env python3
"""D1 (round 17): I4 as a REAL calibration-contrast experiment.

Round 16 used T=0.5/2.0 temperature scaling, labelled 'val-frozen' and
'illegal refit' even though the temperatures were hand-picked distortions,
not fitted parameters -- self-contradictory with its own note.

This version uses the logit-affine (Platt) family p = sigmoid(a*z + b) and
actually FITS (a,b):
  h_val  : fitted on a validation window with base rate 8%;
  h_test : fitted ILLEGALLY on the test window (base rate 2.5%).
Both maps are strictly monotone in z, so PR-AUC, ROC-AUC and ranking are
bit-identical; ECE/Brier improve substantially. This is the correct evidence
for I4: the operating point (threshold metrics) can move without changing
ranking, and an illegal test fit exploits the test base rate to move it further.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v7" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
n = 20000
# Test window: 2.5% base rate; the raw logits are mis-calibrated (over-confident).
y = (rng.random(n) < 0.025).astype(int)
z = rng.normal(loc=np.where(y == 1, 1.0, -0.5), scale=1.0)

# Validation window: different base rate (8%) -- the source of the operating-point shift.
nval = 4000
yval = (rng.random(nval) < 0.08).astype(int)
zval = rng.normal(loc=np.where(yval == 1, 1.6, -0.4), scale=1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fit_platt(zf, yf):
    """Fit (a,b) by NLL on the given window's labels -- the map is frozen there."""
    def nll(theta):
        p = np.clip(sigmoid(theta[0] * zf + theta[1]), 1e-6, 1 - 1e-6)
        return -np.mean(yf * np.log(p) + (1 - yf) * np.log(1 - p))
    r = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000})
    return float(r.x[0]), float(r.x[1])


def ece(yy, pp, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pp, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(pp[m].mean() - yy[m].mean())
    return float(e)


def row(pp):
    return {
        "pr_auc": float(average_precision_score(y, pp)),
        "roc_auc": float(roc_auc_score(y, pp)),
        "ece": ece(y, pp),
        "brier": float(np.mean((pp - y) ** 2)),
    }


a_val, b_val = fit_platt(zval, yval)   # legit: validation-only
a_test, b_test = fit_platt(z, y)       # illegal: uses test labels
p_id = sigmoid(z)
p_val = sigmoid(a_val * z + b_val)
p_test = sigmoid(a_test * z + b_test)

res = {
    "design": "logit-affine Platt fit: h_val on 8% validation window; h_test ILLEGALLY on 2.5% test window",
    "family": "p = sigmoid(a*z + b), strictly monotone for a>0",
    "n_test": n, "test_base_rate": float(y.mean()),
    "n_val": nval, "val_base_rate": float(yval.mean()),
    "coefficients": {
        "h_val": {"a": round(a_val, 4), "b": round(b_val, 4)},
        "h_test": {"a": round(a_test, 4), "b": round(b_test, 4)},
    },
    "identity": row(p_id),
    "h_val": row(p_val),
    "h_test": row(p_test),
}
r0, r1, r2 = res["identity"], res["h_val"], res["h_test"]
res["deltas"] = {
    "dECE_h_val": round(r1["ece"] - r0["ece"], 4),
    "dECE_h_test": round(r2["ece"] - r0["ece"], 4),
    "dBrier_h_val": round(r1["brier"] - r0["brier"], 4),
    "dBrier_h_test": round(r2["brier"] - r0["brier"], 4),
}
res["pr_auc_bit_identical"] = r0["pr_auc"] == r1["pr_auc"] == r2["pr_auc"]
res["roc_auc_bit_identical"] = r0["roc_auc"] == r1["roc_auc"] == r2["roc_auc"]
res["ranking_identical"] = bool(
    np.array_equal(np.argsort(p_id), np.argsort(p_val))
    and np.array_equal(np.argsort(p_id), np.argsort(p_test))
)
(OUT / "i4_platt_affine.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k in (
    "coefficients", "identity", "h_val", "h_test", "deltas",
    "pr_auc_bit_identical", "roc_auc_bit_identical", "ranking_identical")}, indent=2))
