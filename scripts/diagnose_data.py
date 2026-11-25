#!/usr/bin/env python3
"""Data diagnostics demanded by the TIFS review (H2, E7, E2, M3)."""
from __future__ import annotations

import json
import sys
from collections import Counter
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
ts = g.time[g.edge_index[0]]
td = g.time[g.edge_index[1]]
lag = td - ts
n = len(lag)
out = {
    "num_nodes": g.num_nodes,
    "num_edges": int(n),
    "num_features": g.num_features,
    "time_steps": int(g.time.max() + 1),
    "edge_lag": {
        "min": int(lag.min()),
        "max": int(lag.max()),
        "n_negative": int((lag < 0).sum()),
        "n_zero": int((lag == 0).sum()),
        "n_positive": int((lag > 0).sum()),
        "pct_zero": round(100 * float((lag == 0).mean()), 4),
        "hist": {str(k): int(v) for k, v in sorted(Counter(lag.tolist()).items())},
    },
    "labels": {
        "illicit": int((g.y == 1).sum()),
        "licit": int((g.y == 0).sum()),
        "unknown": int((g.y == -1).sum()),
    },
    "per_step": {},
}
for step in range(int(g.time.max()) + 1):
    m = g.time == step
    yy = g.y[m]
    out["per_step"][str(step)] = {
        "n": int(m.sum()),
        "illicit": int((yy == 1).sum()),
        "licit": int((yy == 0).sum()),
        "unknown": int((yy == -1).sum()),
        "illicit_rate": round(float((yy == 1).sum() / max(1, (yy >= 0).sum())), 6),
    }

dest = ROOT / "outputs" / "diagnostics" / "data_diagnostics.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({
    "num_edges": out["num_edges"],
    "lag_min": out["edge_lag"]["min"],
    "lag_max": out["edge_lag"]["max"],
    "lag_zero_pct": out["edge_lag"]["pct_zero"],
    "labels": out["labels"],
}, indent=2))
print("saved", dest)
