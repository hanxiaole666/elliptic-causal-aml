"""Pure-NumPy causal neighbor sampler.

We deliberately avoid PyG's NeighborLoader here: in this environment it needs
pyg-lib / torch-sparse, and we want the causal constraint to be explicit and
auditable. Every sampled edge satisfies time[source] <= time[target].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledBatch:
    local_nodes: np.ndarray
    edge_index: np.ndarray
    target_positions: np.ndarray


class CausalNeighborSampler:
    """Sample temporal predecessors of each target.

    ``strict=False`` uses the non-future-step boundary tau(src) <= tau(dst).
    ``strict=True`` requires tau(src) < tau(dst), excluding same-time-step
    neighbours whose within-step ordering is unknown from the coarse labels.
    """

    def __init__(self, edge_index: np.ndarray, time: np.ndarray, alpha: float = 0.0, strict: bool = False):
        self.time = np.asarray(time)
        self.alpha = float(alpha)
        self.strict = bool(strict)
        n = len(self.time)
        incoming: list[list[int]] = [[] for _ in range(n)]
        for src, dst in np.asarray(edge_index).T:
            ok = self.time[src] < self.time[dst] if self.strict else self.time[src] <= self.time[dst]
            if ok:
                incoming[int(dst)].append(int(src))
        self.incoming = [np.array(x, dtype=np.int64) for x in incoming]
        if edge_index.size:
            self.degree = np.bincount(edge_index[0], minlength=n).astype(np.float32)
        else:
            self.degree = np.zeros(n, dtype=np.float32)

    def sample(
        self,
        roots: np.ndarray,
        num_neighbors: list[int] | tuple[int, ...],
        rng: np.random.Generator,
    ) -> SampledBatch:
        roots = np.asarray(roots, dtype=np.int64)
        local_nodes: list[int] = [int(node) for node in roots]
        local_index = {node: i for i, node in enumerate(local_nodes)}
        edges: list[tuple[int, int]] = []
        current = [int(node) for node in roots]

        for hop, k_max in enumerate(num_neighbors):
            k_max = int(k_max)
            next_nodes: list[int] = []
            for target in current:
                candidates = self.incoming[target]
                k = min(k_max, len(candidates))
                if k <= 0:
                    continue
                if self.alpha > 0:
                    weights = np.power(np.maximum(1.0, self.degree[candidates]), -self.alpha)
                    probs = weights / weights.sum()
                    picked = rng.choice(len(candidates), size=k, replace=False, p=probs)
                else:
                    picked = rng.choice(len(candidates), size=k, replace=False)
                for idx in picked:
                    source = int(candidates[idx])
                    if source not in local_index:
                        local_index[source] = len(local_nodes)
                        local_nodes.append(source)
                    edges.append((local_index[source], local_index[target]))
                    next_nodes.append(source)
            current = list(dict.fromkeys(next_nodes))
            if not current:
                break

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        local_nodes_arr = np.array(local_nodes, dtype=np.int64)
        if edge_index.size:
            src_global = local_nodes_arr[edge_index[0]]
            dst_global = local_nodes_arr[edge_index[1]]
            if np.any(self.time[src_global] > self.time[dst_global]):
                raise AssertionError("CAUSAL VIOLATION: future neighbor in sampled batch")
        return SampledBatch(
            local_nodes=local_nodes_arr,
            edge_index=edge_index,
            target_positions=np.arange(len(roots), dtype=np.int64),
        )


def iterate_batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
    shuffle: bool = False,
):
    indices = np.asarray(indices, dtype=np.int64)
    order = rng.permutation(indices) if shuffle and rng is not None else indices
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]
