#!/usr/bin/env python3
"""H2 rebuttal: prove the t_src<=t_dst filter is non-vacuous.

Reviewer claim: 'all 234,355 Elliptic edges are same-time-step, so the
future-edge invariant is vacuous.' We confirm the lag=0 fact AND show the
filter removes injected future edges on (a) the toy graph which has genuine
cross-time edges and (b) Elliptic with a controlled 5% future-edge injection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.causal_edges import filter_causal_edges, audit_edges  # noqa: E402
from src.data.elliptic import load_elliptic, make_toy_data  # noqa: E402


def audit(graph, label: str, inject_future: float = 0.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    edge = graph.edge_index.copy()
    time = graph.time
    if inject_future > 0 and edge.size:
        # Rewire a fraction of edges to point backwards in time (t_src > t_dst).
        n = edge.shape[1]
        k = int(n * inject_future)
        pick = rng.choice(n, size=k, replace=False)
        for idx in pick:
            src = int(edge[0, idx])
            earlier = np.flatnonzero(time < time[src])
            if len(earlier):
                edge[1, idx] = int(rng.choice(earlier))
    lag = time[edge[1]] - time[edge[0]]
    before_violations = int((lag < 0).sum())
    filtered, removed = filter_causal_edges(edge, time)
    audit_obj = audit_edges(filtered, time, "real", removed)
    return {
        "graph": label,
        "inject_future": inject_future,
        "edges_before": int(edge.shape[1]),
        "future_edges_before": before_violations,
        "edges_after": int(filtered.shape[1]),
        "future_edges_removed": int(removed),
        "causal_violations_after": audit_obj.causal_violations,
        "lag_min": int(lag.min()) if len(lag) else None,
        "lag_max": int(lag.max()) if len(lag) else None,
    }


def main():
    paths = {
        "elliptic_txs_features.csv": ROOT / "data" / "raw" / "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv": ROOT / "data" / "raw" / "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv": ROOT / "data" / "raw" / "elliptic_txs_edgelist.csv",
    }
    ell = load_elliptic(paths)
    toy = make_toy_data(seed=0)
    results = [
        audit(ell, "elliptic-real", inject_future=0.0),
        audit(ell, "elliptic-injected-5pct", inject_future=0.05, seed=1),
        audit(toy, "toy-real", inject_future=0.0),
        audit(toy, "toy-injected-10pct", inject_future=0.10, seed=2),
    ]
    # Hard assertions: the filter must remove ALL injected future edges.
    for r in results:
        assert r["causal_violations_after"] == 0, f"filter failed on {r['graph']}"
    assert results[1]["future_edges_removed"] > 0, "injected Elliptic future edges were not detected"
    assert results[3]["future_edges_removed"] > 0, "injected toy future edges were not detected"

    dest = ROOT / "outputs" / "diagnostics" / "future_edge_filter_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"status": "ok", "runs": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("saved", dest)


if __name__ == "__main__":
    main()
