#!/usr/bin/env python3
"""E2 proxy: within-step temporal-order RECOVERY sensitivity.

Reviewer M2: native Elliptic has only step-level timestamps, so the strict
boundary tau(u)<tau(v) removes all edges. But each step is a ~3h block-time
window; if finer within-step order were recovered, cross-transaction edges
would survive. We cannot obtain real block timestamps in this environment, so
we run an HONEST UPPER-BOUND proxy: assign each node a pseudo within-step rank
(uniform random, fixed seed) and define recovered time tau'(v)=tau(v)+rank(v)/M.
Then tau'(u)<tau'(v) retains roughly half the intra-step edges.

This is NOT a claim about real Elliptic chronology. It answers the narrower
question: 'does the contract collapse when within-step order is available?'
If the graph survives and the GNN gains nothing, the negative result is robust
to this modelling choice; if it gains, we must qualify the claim.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.elliptic import load_elliptic  # noqa: E402

paths = {
    "elliptic_txs_features.csv": ROOT / "data" / "raw" / "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv": ROOT / "data" / "raw" / "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv": ROOT / "data" / "raw" / "elliptic_txs_edgelist.csv",
}
g = load_elliptic(paths)
OUT = ROOT / "outputs" / "v6" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

edge = g.edge_index
time = g.time
results = {}

# Native: all intra-step -> strict keeps 0
lag_native = time[edge[1]] - time[edge[0]]
results["native"] = {
    "n_edges": int(edge.shape[1]),
    "strict_retained": int((lag_native > 0).sum()),
    "same_step_fraction": round(float((lag_native == 0).mean()), 4),
}

# Recovered within-step order: pseudo-rank per node, fixed seed.
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    rank = np.zeros(len(time), dtype=np.float64)
    for t in np.unique(time):
        idx = np.flatnonzero(time == t)
        perm = rng.permutation(len(idx))
        rank[idx] = perm / max(1, len(idx) - 1)  # in [0,1)
    tau2 = time.astype(float) + rank
    lag2 = tau2[edge[1]] - tau2[edge[0]]
    results[f"recovered_seed{seed}"] = {
        "strict_retained": int((lag2 > 0).sum()),
        "retained_fraction": round(float((lag2 > 0).mean()), 4),
        "forward_edges": int((lag2 > 0).sum()),
        "backward_edges": int((lag2 < 0).sum()),
        "mean_abs_lag": round(float(np.mean(np.abs(lag2[lag2 != 0]))), 4),
    }

out = {
    "status": "proxy, not real block timestamps",
    "assumption": "within-step order recovered as uniform pseudo-rank (fixed seed)",
    "question": "does the contract's strict boundary collapse when within-step order exists?",
    "results": results,
    "interpretation": ("Under recovered within-step order the graph does NOT collapse: "
                       "roughly half of intra-step edges satisfy tau'(u)<tau'(v). "
                       "This converts 'strict filtering is vacuous' from a data property "
                       "into a resolution property of the released timestamps."),
}
(OUT / "within_step_recovery.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({"native": results["native"],
                  "recovered_seed0": results["recovered_seed0"]}, indent=2))
print("saved", OUT / "within_step_recovery.json")
