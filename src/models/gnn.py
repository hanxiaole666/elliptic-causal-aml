"""Small GNN models with explicit causal sampling and zero-parameter degree correction."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


def time_encoding(time: np.ndarray, k: int = 8) -> np.ndarray:
    """Fixed periodic encoding; no learnable parameters and no future normalisation."""
    import numpy as np

    time = np.asarray(time, dtype=np.float64)
    lo, hi = float(time.min()), float(time.max())
    rho = (time - lo) / max(1.0, hi - lo)
    columns = []
    for j in range(1, k + 1):
        frequency = 2.0 ** (j - 1)
        angle = 2.0 * math.pi * frequency * rho
        columns.extend([np.cos(angle), np.sin(angle)])
    return np.column_stack(columns).astype(np.float32)


def _scatter_mean(source: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros((num_nodes, source.size(1)), dtype=source.dtype, device=source.device)
    count = torch.zeros(num_nodes, dtype=source.dtype, device=source.device)
    out.index_add_(0, index, source)
    count.index_add_(0, index, torch.ones_like(index, dtype=source.dtype))
    return out / count.clamp(min=1.0).unsqueeze(-1)


def _robust_aggregate(
    source: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
    mode: str,
    trim_alpha: float = 0.1,
) -> torch.Tensor:
    """Median or trimmed-mean aggregation for test-time robustness probes."""
    if mode == "mean":
        return _scatter_mean(source, index, num_nodes)
    out = torch.zeros((num_nodes, source.size(1)), dtype=source.dtype, device=source.device)
    for node in torch.unique(index):
        values = source[index == node]
        if values.size(0) == 0:
            continue
        if mode == "median":
            out[node] = torch.median(values, dim=0).values
        elif mode == "trim":
            order = torch.argsort(values.norm(dim=1))
            values = values[order]
            k = int(trim_alpha * values.size(0))
            trimmed = values[k : values.size(0) - k] if values.size(0) - 2 * k > 0 else values
            out[node] = trimmed.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregator: {mode}")
    return out


class MeanSAGE(nn.Module):
    """Mean aggregation with optional 1/sqrt(d_u d_v) edge weighting.

    Parameter count is identical with and without degree weighting.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        layers: int,
        dropout: float,
        degree_weight: bool = False,
        aggregator: str = "mean",
        trim_alpha: float = 0.1,
    ):
        super().__init__()
        self.self_lins = nn.ModuleList()
        self.neigh_lins = nn.ModuleList()
        dims = [in_channels] + [hidden] * (layers - 1)
        for dim in dims:
            self.self_lins.append(nn.Linear(dim, hidden))
            self.neigh_lins.append(nn.Linear(dim, hidden))
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.degree_weight = degree_weight
        self.aggregator = aggregator
        self.trim_alpha = trim_alpha
        self.num_layers = layers

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_positions: torch.Tensor,
        deg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for layer in range(self.num_layers):
            if edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                messages = h[src]
                if self.degree_weight and deg is not None:
                    weight = 1.0 / torch.sqrt(deg[src] * deg[dst] + 1.0)
                    messages = messages * weight.unsqueeze(-1)
                aggregated = _robust_aggregate(messages, dst, h.size(0), self.aggregator, self.trim_alpha)
            else:
                aggregated = torch.zeros_like(h)
            h = self.self_lins[layer](h) + self.neigh_lins[layer](aggregated)
            if layer < self.num_layers - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return self.head(h[target_positions]).squeeze(-1)


class SmallGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden] * max(0, layers - 1)
        for dim in dims:
            self.convs.append(GCNConv(dim, hidden))
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, target_positions, deg=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = torch.relu(conv(h, edge_index))
            if i < len(self.convs) - 1:
                h = self.dropout(h)
        return self.head(h[target_positions]).squeeze(-1)


class SmallGAT(nn.Module):
    def __init__(self, in_channels: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        heads = 2
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, max(1, hidden // heads), heads=heads, dropout=dropout))
        for _ in range(max(0, layers - 1)):
            self.convs.append(GATConv(hidden, hidden, heads=1, concat=False, dropout=dropout))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, target_positions, deg=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = torch.relu(conv(h, edge_index))
        return self.head(h[target_positions]).squeeze(-1)


def build_gnn(
    name: str,
    in_channels: int,
    hidden: int,
    layers: int,
    dropout: float,
    degree_weight: bool = False,
    aggregator: str = "mean",
):
    if name in {"causal_sage", "graphsage", "sage"}:
        return MeanSAGE(in_channels, hidden, layers, dropout, degree_weight, aggregator)
    if name == "gcn":
        return SmallGCN(in_channels, hidden, layers, dropout)
    if name in {"gat", "tgat"}:
        # TGAT-style baseline: GAT attention over causally sampled neighbours,
        # with periodic time encoding appended to the node features upstream.
        return SmallGAT(in_channels, hidden, layers, dropout)
    raise ValueError(f"Unknown GNN: {name}")
