from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.causal_edges import build_edge_world, degree_preserving_rewire, filter_causal_edges
from src.data.download import ensure_data
from src.data.elliptic import make_toy_data
from src.data.label_protocol import apply_label_protocol
from src.data.split import chronological_split, random_split
from src.evaluation.metrics import binary_metrics
from src.models.gnn import MeanSAGE
from src.sampling.causal_sampler import CausalNeighborSampler
from src.utils.seed import rng_for

ROOT = Path(__file__).resolve().parents[1]


def test_no_future_edges_filtered():
    time = np.array([0, 1, 2, 1, 3])
    edges = np.array([[0, 1, 3, 4], [1, 2, 1, 2]])
    filtered, removed = filter_causal_edges(edges, time)
    assert removed == 1
    assert np.all(time[filtered[0]] <= time[filtered[1]])


def test_chronological_split_disjoint_and_covers():
    time = np.repeat(np.arange(10), 5)
    split = chronological_split(time)
    assert not set(split.train_times) & set(split.valid_times)
    assert not set(split.train_times) & set(split.test_times)
    assert not set(split.valid_times) & set(split.test_times)
    assert len(split.train_idx) + len(split.valid_idx) + len(split.test_idx) == len(time)


def test_random_split_is_disjoint_and_covers_all_nodes():
    time = np.repeat(np.arange(10), 5)
    split = random_split(time, seed=9)
    combined = np.concatenate([split.train_idx, split.valid_idx, split.test_idx])
    assert len(combined) == len(time)
    assert len(np.unique(combined)) == len(time)
    assert not set(split.train_idx) & set(split.valid_idx)
    assert not set(split.train_idx) & set(split.test_idx)


def test_degree_preserving_rewire_keeps_degree_sequences():
    graph = make_toy_data(num_nodes=300, num_steps=6, seed=5)
    real, _ = filter_causal_edges(graph.edge_index, graph.time)
    rewired = degree_preserving_rewire(real, graph.time, rng_for(5, "rewire"), swaps_factor=1)
    out_before = np.bincount(real[0], minlength=graph.num_nodes)
    out_after = np.bincount(rewired[0], minlength=graph.num_nodes)
    in_before = np.bincount(real[1], minlength=graph.num_nodes)
    in_after = np.bincount(rewired[1], minlength=graph.num_nodes)
    assert np.array_equal(out_before, out_after)
    assert np.array_equal(in_before, in_after)
    assert np.all(graph.time[rewired[0]] <= graph.time[rewired[1]])


def test_unknown_excluded_from_loss_and_metrics():
    y = np.array([1, 0, -1, -1, 1])
    effective, mask = apply_label_protocol(y, "A")
    assert mask.sum() == 3
    assert effective[mask].min() >= 0
    effective_b, mask_b = apply_label_protocol(y, "B")
    assert mask_b.all()
    assert (effective_b == 0).sum() == 3
    effective_d, mask_d = apply_label_protocol(y, "D")
    assert mask_d.all()
    assert effective_d.min() == -1


def test_binary_metrics_and_ece_finite():
    y = np.array([1, 0, 1, 0, 1, 0])
    score = np.array([0.9, 0.1, 0.8, 0.4, 0.7, 0.2])
    metrics = binary_metrics(y, score, k_list=(2, 4), ece_bins=5)
    assert np.isfinite(metrics["pr_auc"])
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["precision@2"] <= 1


def test_sampler_never_returns_future_neighbors():
    graph = make_toy_data(num_nodes=400, num_steps=8, seed=3)
    edges, audit = build_edge_world(graph.edge_index, graph.time, "future", rng=rng_for(3, "attack"))
    assert audit.causal_violations == 0
    sampler = CausalNeighborSampler(edges, graph.time, alpha=0.5)
    roots = np.arange(20)
    batch = sampler.sample(roots, [5, 5], rng_for(3, "sample"))
    src = batch.local_nodes[batch.edge_index[0]]
    dst = batch.local_nodes[batch.edge_index[1]]
    assert np.all(graph.time[src] <= graph.time[dst])


def test_degree_weight_and_robust_aggregators_add_no_parameters():
    count = lambda m: sum(p.numel() for p in m.parameters())
    plain = MeanSAGE(10, 32, 2, 0.5, degree_weight=False)
    corrected = MeanSAGE(10, 32, 2, 0.5, degree_weight=True)
    median = MeanSAGE(10, 32, 2, 0.5, degree_weight=True, aggregator="median")
    trim = MeanSAGE(10, 32, 2, 0.5, degree_weight=True, aggregator="trim")
    assert count(plain) == count(corrected) == count(median) == count(trim)


def test_ensure_data_skips_download_when_files_exist(tmp_path: Path):
    for name in ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"):
        (tmp_path / name).write_text("txId,class\n1,1\n", encoding="utf-8")
    paths, version = ensure_data(tmp_path, download=False, allow_toy=False)
    assert version == "elliptic"
    assert set(paths) == {
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
    }
    assert (tmp_path / "SHA256SUMS").exists()


def test_checkpoint_roundtrip(tmp_path: Path):
    from scripts.run_pipeline import save_checkpoint

    model = MeanSAGE(10, 16, 2, 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {
        "epoch": 4,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": 0.7,
    })
    restored = torch.load(path, map_location="cpu")
    assert restored["epoch"] == 4
    assert restored["best_score"] == 0.7
    reloaded = MeanSAGE(10, 16, 2, 0.1)
    reloaded.load_state_dict(restored["model"])
    for key in reloaded.state_dict():
        assert torch.equal(reloaded.state_dict()[key], model.state_dict()[key])


def test_end_to_end_toy_smoke(tmp_path: Path):
    output = tmp_path / "outputs"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--data-mode", "toy",
        "--model", "causal_sage",
        "--seed", "0",
        "--epochs", "1",
        "--batch-size", "128",
        "--neighbors", "5", "5",
        "--device", "cpu",
        "--calibration", "none",
        "--output-dir", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stderr
    results = list((output / "results").glob("*.json"))
    results = [p for p in results if p.name != "leaderboard.json"]
    assert results, "no result JSON produced"
    payload = json.loads(results[0].read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["dataset_version"] == "toy"
    assert np.isfinite(payload["metrics"]["calibrated"]["pr_auc"])
