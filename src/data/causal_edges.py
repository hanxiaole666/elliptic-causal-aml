"""Causal edge construction and the counterfactual edge worlds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeAudit:
    mode: str
    n_edges_in: int
    n_edges_out: int
    n_future_edges_removed: int
    max_source_degree: int
    max_target_degree: int
    causal_violations: int


def filter_causal_edges(edge_index: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep only edges with t_src <= t_dst. Returns (edges, removed)."""
    if edge_index.size == 0:
        return edge_index.copy(), 0
    src, dst = edge_index[0], edge_index[1]
    keep = time[src] <= time[dst]
    return edge_index[:, keep].copy(), int((~keep).sum())


def audit_edges(edge_index: np.ndarray, time: np.ndarray, mode: str, removed: int = 0) -> EdgeAudit:
    if edge_index.size == 0:
        return EdgeAudit(mode, 0, 0, removed, 0, 0, 0)
    src, dst = edge_index[0], edge_index[1]
    src_deg = np.bincount(src, minlength=len(time))
    dst_deg = np.bincount(dst, minlength=len(time))
    return EdgeAudit(
        mode=mode,
        n_edges_in=int(edge_index.shape[1]) + removed,
        n_edges_out=int(edge_index.shape[1]),
        n_future_edges_removed=removed,
        max_source_degree=int(src_deg.max()),
        max_target_degree=int(dst_deg.max()),
        causal_violations=int((time[src] > time[dst]).sum()),
    )


def degree_preserving_rewire(
    edge_index: np.ndarray,
    time: np.ndarray,
    rng: np.random.Generator,
    swaps_factor: int = 5,
) -> np.ndarray:
    """Markov-chain edge swaps that preserve both degree sequences and causality.

    A swap exchanges the destinations of two edges. This keeps each source's
    out-degree and each target's in-degree exactly unchanged; the time filter
    additionally enforces t_src <= t_dst for both new edges.
    """
    real, _ = filter_causal_edges(edge_index, time)
    if real.size == 0:
        return real.copy()
    edges = list(dict.fromkeys(tuple(int(v) for v in pair) for pair in real.T))
    edge_set = set(edges)
    n_swaps = max(1, len(edges) * swaps_factor)
    for _ in range(n_swaps):
        i, j = (int(v) for v in rng.integers(0, len(edges), size=2))
        if i == j:
            continue
        s1, d1 = edges[i]
        s2, d2 = edges[j]
        if s1 == d2 or s2 == d1:
            continue
        if time[s1] > time[d2] or time[s2] > time[d1]:
            continue
        new1, new2 = (s1, d2), (s2, d1)
        if new1 == new2 or new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove((s1, d1))
        edge_set.remove((s2, d2))
        edge_set.add(new1)
        edge_set.add(new2)
        edges[i], edges[j] = new1, new2
    return np.array(edges, dtype=np.int64).T


def build_edge_world(
    edge_index: np.ndarray,
    time: np.ndarray,
    mode: str,
    rng: np.random.Generator | None = None,
    future_fraction: float = 0.1,
) -> tuple[np.ndarray, EdgeAudit]:
    """Construct one edge world. All returned edges satisfy t_src <= t_dst."""
    real, removed = filter_causal_edges(edge_index, time)
    if mode == "real":
        chosen, rem = real, removed
    elif mode == "no_time":
        same = real[:, time[real[0]] == time[real[1]]]
        chosen, rem = same, 0
    elif mode == "empty":
        chosen, rem = np.empty((2, 0), dtype=np.int64), 0
    elif mode == "perm":
        if rng is None:
            raise ValueError("mode='perm' requires rng")
        permuted_dst = rng.permutation(real[1])
        candidate = np.vstack([real[0], permuted_dst])
        chosen, rem = filter_causal_edges(candidate, time)
        rem += removed
    elif mode == "rewire":
        if rng is None:
            raise ValueError("mode='rewire' requires rng")
        chosen = degree_preserving_rewire(edge_index, time, rng)
        chosen, re_removed = filter_causal_edges(chosen, time)
        rem = removed + re_removed
    elif mode == "future":
        if rng is None:
            raise ValueError("mode='future' requires rng")
        n_add = max(1, int(real.shape[1] * future_fraction))
        src = rng.integers(0, len(time), size=n_add)
        dst = rng.integers(0, len(time), size=n_add)
        bad = np.vstack([src, dst])
        candidate = np.hstack([real, bad])
        chosen, rem = filter_causal_edges(candidate, time)
        rem += removed + n_add
    else:
        raise ValueError(f"Unknown edge world: {mode}")

    audit = audit_edges(chosen, time, mode, rem)
    if audit.causal_violations:
        raise AssertionError("edge world contains future edges")
    return chosen, audit
