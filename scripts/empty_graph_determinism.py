#!/usr/bin/env python3
"""Fourth-round requirement 4: determinism of the zero-edge control.

Two code paths (strict tau(u)<tau(v) filter on native Elliptic, and explicit
empty-world construction) both yield an empty graph. This test constructs the
empty edge_index via both paths with a fixed seed and data order, and verifies
bit-identical sampled batches and model scores. If identical, the three
reported zero-edge values must be replaced by ONE value with std.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sampling.causal_sampler import CausalNeighborSampler  # noqa: E402
from src.models.gnn import build_gnn  # noqa: E402

OUT = ROOT / "outputs" / "v5" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
n, d = 500, 10
time = np.repeat(np.arange(50), 10)  # all edges intra-step
x = torch.randn(n, d)
# Path A: strict filter on native intra-step edges -> all removed
edges_native = np.array([[i, i + 1] for i in range(0, n - 1, 2)]).T  # same-step pairs
keep = time[edges_native[0]] < time[edges_native[1]]
edges_a = edges_native[:, keep]
# Path B: explicit empty world
edges_b = np.empty((2, 0), dtype=np.int64)

roots = np.arange(60)
rng_a = np.random.default_rng(42)
rng_b = np.random.default_rng(42)
sa = CausalNeighborSampler(edges_a, time, alpha=0.5, strict=True).sample(roots, [10, 10], rng_a)
sb = CausalNeighborSampler(edges_b, time, alpha=0.5, strict=False).sample(roots, [10, 10], rng_b)

edges_identical = np.array_equal(sa.edge_index, sb.edge_index)
nodes_identical = np.array_equal(sa.local_nodes, sb.local_nodes)

model = build_gnn("causal_sage", d, 32, 2, 0.3, degree_weight=True)
model.eval()
with torch.no_grad():
    za = model(x[sa.local_nodes],
               torch.from_numpy(sa.edge_index), torch.from_numpy(sa.target_positions))
    zb = model(x[sb.local_nodes],
               torch.from_numpy(sb.edge_index), torch.from_numpy(sb.target_positions))
scores_identical = torch.allclose(za, zb, atol=0, rtol=0)

res = {
    "path_a_edges": int(edges_a.shape[1]),
    "path_b_edges": int(edges_b.shape[1]),
    "sampled_edges_identical": bool(edges_identical),
    "sampled_nodes_identical": bool(nodes_identical),
    "scores_bit_identical": bool(scores_identical),
    "max_abs_score_diff": float(torch.abs(za - zb).max().item()),
    "conclusion": "both paths produce identical empty-graph behaviour" if scores_identical else "paths differ; investigate",
}
(OUT / "empty_graph_determinism.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
