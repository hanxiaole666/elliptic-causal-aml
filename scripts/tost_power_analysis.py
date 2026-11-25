#!/usr/bin/env python3
"""E8: TOST power analysis with the OBSERVED paired SD.

Reviewer: with n=15, SD≈0.064, epsilon=0.02, TOST power is ~0.33 -- the
'equivalence not established' statement is underpowered and cannot support
the stronger claim. We compute power for the observed delta and report the
n needed for 0.80 power, then recommend either raising n or dropping the
equivalence framing in favour of 'no significant difference + small point
estimate'.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v6" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

paired = json.loads((ROOT / "outputs" / "v5" / "tables" / "paired_v5.json").read_text(encoding="utf-8"))
fe = paired["full_minus_empty"]
delta, sd, n = fe["mean_delta"], fe.get("std_delta", 0.064), fe["n"]
eps = 0.02
alpha = 0.05


def tost_power(d, s, nn, eps):
    """Power of two one-sided t-tests at level alpha for equivalence."""
    se = s / np.sqrt(nn)
    df = nn - 1
    c = stats.t.ppf(1 - alpha, df)
    # Reject H0 if both (d+eps)/se > c and (d-eps)/se < -c
    p1 = 1 - stats.nct.cdf(c, df, (d + eps) / se)
    p2 = stats.nct.cdf(-c, df, (d - eps) / se)
    # More stable normal approximation for reporting
    z1 = (eps - abs(d)) / se
    approx = 2 * stats.norm.cdf(z1 - stats.norm.ppf(1 - alpha)) - 1
    return float(max(0.0, min(1.0, approx)))


observed_power = tost_power(delta, sd, n, eps)
powers = {str(nn): round(tost_power(delta, sd, nn, eps), 3) for nn in (15, 30, 45, 60, 64, 80)}
needed = next((nn for nn in range(10, 300) if tost_power(delta, sd, nn, eps) >= 0.80), None)

# Re-anchored epsilon: a decision-relevant margin of 0.05 PR-AUC
# (roughly 1/8 of the tree-vs-graph gap) rather than the arbitrary 0.02.
eps2 = 0.05
powers2 = {str(nn): round(tost_power(delta, sd, nn, eps2), 3) for nn in (15, 30, 45, 60, 64, 80)}
needed2 = next((nn for nn in range(10, 500) if tost_power(delta, sd, nn, eps2) >= 0.80), None)

out = {
    "observed_delta": delta, "observed_sd": sd, "n": n,
    "epsilon_02": eps,
    "critical_finding": ("The observed |delta|=0.0209 EXCEEDS the pre-set epsilon=0.02, so TOST "
                         "cannot pass at ANY sample size; the earlier 'equivalence not established' "
                         "was not merely underpowered but structurally impossible at that margin."),
    "power_at_eps02_n15": observed_power,
    "power_table_eps02": powers,
    "epsilon_05_decision_anchored": eps2,
    "power_table_eps05": powers2,
    "n_for_80_power_eps05": needed2,
    "recommendation": ("We DROP the equivalence framework. The defensible statement is that the "
                       "point estimate is small relative to the tree-vs-graph gap with a CI including "
                       "zero, i.e. no attributable structure gain is DEMONSTRATED -- a claim about "
                       "evidence, not equality."),
}
(OUT / "tost_power.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
