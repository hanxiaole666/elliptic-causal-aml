"""Loaders for the official Elliptic CSVs and a clearly-marked toy fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GraphData:
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    edge_index: np.ndarray
    node_ids: np.ndarray
    dataset_version: str

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])


def _map_label(value) -> int:
    text = str(value).strip().lower()
    if text in {"1", "illicit"}:
        return 1
    if text in {"2", "licit"}:
        return 0
    return -1


def load_elliptic(paths: dict[str, Path]) -> GraphData:
    features = pd.read_csv(paths["elliptic_txs_features.csv"], header=None)
    classes = pd.read_csv(paths["elliptic_txs_classes.csv"])
    edges = pd.read_csv(paths["elliptic_txs_edgelist.csv"])

    node_ids = features.iloc[:, 0].to_numpy()
    time = features.iloc[:, 1].to_numpy(dtype=np.int64)
    x = features.iloc[:, 2:].to_numpy(dtype=np.float32)

    label_map = {
        str(row[classes.columns[0]]): _map_label(row[classes.columns[1]])
        for _, row in classes.iterrows()
    }
    y = np.array([label_map.get(str(node_id), -1) for node_id in node_ids], dtype=np.int64)

    index = {str(node_id): i for i, node_id in enumerate(node_ids)}
    src = edges.iloc[:, 0].map(lambda value: index.get(str(value), -1)).to_numpy(dtype=np.int64)
    dst = edges.iloc[:, 1].map(lambda value: index.get(str(value), -1)).to_numpy(dtype=np.int64)
    valid = (src >= 0) & (dst >= 0) & (src < len(node_ids)) & (dst < len(node_ids))
    edge_index = np.vstack([src[valid], dst[valid]]).astype(np.int64)

    return GraphData(
        x=x,
        y=y,
        time=time,
        edge_index=edge_index,
        node_ids=node_ids,
        dataset_version="elliptic",
    )


def make_toy_data(num_nodes: int = 3000, num_steps: int = 12, seed: int = 0) -> GraphData:
    """Synthetic DAG-like graph. NEVER use toy results as paper evidence."""
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, num_steps, size=num_nodes)).astype(np.int64)
    x = rng.standard_normal((num_nodes, 24)).astype(np.float32)

    edges: list[tuple[int, int]] = []
    order = np.argsort(time, kind="stable")
    position = {int(node): i for i, node in enumerate(order)}
    for target in order:
        earlier = order[: position[int(target)]]
        if len(earlier) == 0:
            continue
        count = min(6, len(earlier))
        sources = rng.choice(earlier, size=count, replace=False)
        for source in sources:
            if time[int(source)] <= time[int(target)]:
                edges.append((int(source), int(target)))

    logit = 0.8 * x[:, 0] + 0.5 * x[:, 1] - 0.3 * time.astype(float) / max(1, num_steps)
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(num_nodes) < probability).astype(np.int64)
    unknown = rng.random(num_nodes) < 0.35
    y[unknown] = -1

    edge_index = (
        np.array(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64)
    )
    return GraphData(
        x=x,
        y=y,
        time=time,
        edge_index=edge_index,
        node_ids=np.arange(num_nodes),
        dataset_version="toy",
    )
