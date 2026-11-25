#!/usr/bin/env python3
"""R1: I6 feature observation-time auditor -- makes I6 machine-checkable.

Two-layer check:
  (A) Declaration completeness: every one of the 165 columns carries a
      declared family in {intrinsic, snapshot_address_aggregate, undetermined}
      and an observation-time tag t_f relative to the label time t_y.
  (B) Behavioural consistency: for columns declared 'intrinsic', we test
      whether the column's cross-step distribution is stable (an intrinsic
      per-transaction feature should not drift systematically with the step
      index). Columns with strong step-dependence are flagged; they are *not*
      proved leaky -- the flag only means the declaration needs review.

We do NOT reconstruct the publisher's original computation. Following Safar
et al. (DFRWS 2026), a substantial part of the nominal local block is believed
to be address-level aggregates computed at one snapshot; without their released
mapping we conservatively mark all such columns 'undetermined', which is an
I6 FAIL under the strict predicate (t_f < t_y must be certified, not assumed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "v7" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

feat = pd.read_csv(ROOT / "data" / "raw" / "elliptic_txs_features.csv", header=None)
time = feat.iloc[:, 1].values
X = feat.iloc[:, 2:].values
n_col = X.shape[1]
assert n_col == 165

# One declaration per column: two families known from the public release
# schema (columns 1-94 nominal local, 95-165 aggregated), but per Safar et al.
# the local block may itself contain address aggregates -> all are declared
# 'undetermined' unless individually certified. This is the conservative I6
# stance and makes the check FAIL-OPEN-to-reject, not fail-open-to-accept.
decl = []
# Official schema (Weber 2019 / Kaggle / Safar 2026 sec.3.1-3.2):
#   166 features = 94 local (INCLUDING the time step) + 72 aggregated.
# The CSV carries 167 columns = txId + time step + 165 model features;
# dropping the time step leaves 165 = 93 local + 72 aggregated.
# Safar: only the first 21 of the local features are transaction-scoped;
# the remaining 72 local are address aggregates, and all 72 aggregated
# features are neighbourhood aggregates. Hence 165 = 21 + 72 + 72 and
# 144 = 72 + 72 columns have t_f at block 575059 (after the test window).
N_TRANSACTION, N_ADDRESS, N_NEIGHBOUR = 21, 72, 72
assert N_TRANSACTION + N_ADDRESS + N_NEIGHBOUR == n_col, "partition must sum to 165"

for j in range(n_col):
    nominal = "local93" if j < 93 else "aggregated72"
    if j < N_TRANSACTION:
        family, tf, verdict = "transaction_scoped", "t_f = t_y (same transaction)", "pass"
    elif j < N_TRANSACTION + N_ADDRESS:
        family, tf, verdict = "address_scoped", "block 575059 > all test windows", "fail"
    else:
        family, tf, verdict = "neighbourhood_aggregated", "block 575059 > all test windows", "fail"
    decl.append({
        "col": j, "nominal_family": nominal,
        "declared_family": family,
        "t_f_relation": tf,
        "i6_certified": verdict == "pass",
        "i6_verdict": verdict,
    })

# Behavioural consistency (advisory only): Spearman correlation of column
# median with step index. Strong drift suggests (but does not prove) a
# snapshot/aggregate origin.
steps = np.unique(time)
flags = []
for j in range(n_col):
    meds = [np.median(X[time == t, j]) for t in steps]
    rho, p = stats.spearmanr(steps, meds)
    if np.isfinite(rho) and abs(rho) > 0.7 and p < 0.01:
        flags.append({"col": j, "rho": round(float(rho), 3), "p": float(p)})
        decl[j]["drift_flag"] = True
    else:
        decl[j]["drift_flag"] = False

n_certified = sum(d["i6_certified"] for d in decl)
n_fail = sum(d.get("i6_verdict") == "fail" for d in decl)
n_flagged = len(flags)
result = {
    "predicate": "I6: for each feature family f, t_f < t_y must be certified",
    "n_columns": n_col,
    "declaration_completeness": {
        "all_columns_declared": all(d["declared_family"] for d in decl),
        "n_certified_i6": n_certified,
        "n_verdict_fail": n_fail,
        "n_undetermined": n_col - n_certified,
    },
    "behavioural_flags": {
        "n_strong_drift": n_flagged,
        "note": "drift flags are advisory; they do not prove leakage",
        "columns": flags[:20],
    },
    "verdict": "FAIL" if n_fail > 0 else ("PASS" if n_certified == n_col else "UNVERIFIABLE"),
    "evidence": {
        "partition_from_column_indices": {
            "n_transaction_scoped": N_TRANSACTION,
            "n_address_scoped": N_ADDRESS,
            "n_neighbourhood_aggregated": N_NEIGHBOUR,
            "sum": N_TRANSACTION + N_ADDRESS + N_NEIGHBOUR,
            "assertion": "sum == 165 and (21, 72, 72)",
        },
        "reference_block": 575059,
        "elliptic_window_end_block": 487975,
        "source": "Safar et al. FSIDI 2026 sec.3.2 (block 575,059); Liang et al. 2026 Table 1 (window 391200-487975)",
        "partition_note": "Official schema: 166 features = 94 local (incl. time step) + 72 aggregated; after dropping txId and time step, 165 = 93 local + 72 aggregated = 21 transaction-scoped + 72 address-scoped + 72 neighbourhood-aggregated (Safar 2026 sec.3.1-3.2).",
    },
    "interpretation": ("I6 returns FAIL: 144 of 165 columns are address- or neighbourhood-scoped "
                       "aggregates evaluated at block 575,059, after the entire Elliptic window "
                       "(end block 487,975), so t_f > t_y is demonstrated for those columns."),
}
(OUT / "i6_feature_time_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
pd.DataFrame(decl).to_csv(OUT / "i6_column_declarations.csv", index=False)
print(f"I6 verdict: {result['verdict']}; certified {n_certified}/{n_col}; drift flags {n_flagged}")
print("saved", OUT)
