#!/usr/bin/env python3
"""Reproducible Elliptic causal-AML runner with auto-download and resume.

Examples
--------
# Auto-download if missing, then train with checkpoint resume:
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --model causal_sage --seed 0 --epochs 2

# Force toy smoke test (never use toy numbers in a paper):
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --data-mode toy --model causal_sage --seed 0 --epochs 2

# Tree baseline:
C:/Users/apple/anaconda3/python.exe scripts/run_pipeline.py --model lightgbm --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.causal_edges import build_edge_world  # noqa: E402
from src.data.download import DEFAULT_URL, ensure_data  # noqa: E402
from src.data.elliptic import load_elliptic, make_toy_data  # noqa: E402
from src.data.label_protocol import apply_label_protocol, unknown_distribution_audit  # noqa: E402
from src.data.split import canonical_split, chronological_split, random_split, save_split_manifest  # noqa: E402
from src.evaluation.calibration import apply_calibration  # noqa: E402
from src.evaluation.metrics import binary_metrics, subgroup_feature_proxy, subgroup_metrics  # noqa: E402
from src.models.gnn import build_gnn, time_encoding  # noqa: E402
from src.models.tabular import fit_tabular, predict_proba  # noqa: E402
from src.sampling.causal_sampler import CausalNeighborSampler, iterate_batches  # noqa: E402
from src.utils.io import atomic_write_json, read_json, stable_hash  # noqa: E402
from src.utils.seed import rng_for, seed_everything  # noqa: E402

TABULAR = {"lr", "rf", "xgboost", "lightgbm", "mlp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-mode", choices=["auto", "elliptic", "toy"], default="auto")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading when data are missing")
    parser.add_argument("--allow-toy", action="store_true", help="fall back to toy data if download fails")
    parser.add_argument("--model", default="causal_sage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pos-weight-mode", choices=["balanced", "sqrt", "none"], default="balanced")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5, help="degree-aware sampling exponent; 0 = uniform")
    parser.add_argument("--strict-time", action="store_true",
                        help="exclude same-time-step edges: keep only tau(src) < tau(dst) (strict causal boundary)")
    parser.add_argument("--degree-weight", action="store_true", help="use 1/sqrt(d_u d_v) aggregation weighting")
    parser.add_argument("--no-time-encoding", action="store_true")
    parser.add_argument("--no-standardize", action="store_true", help="disable train-only feature standardization for neural models")
    parser.add_argument("--time-k", type=int, default=8)
    parser.add_argument("--split-mode", choices=["chrono", "random"], default="chrono")
    parser.add_argument("--canonical-split", choices=["none", "1-34_35-42", "1-34_35-49", "1-34_35-49-strict"], default="none",
                        help="canonical literature split; overrides --split-mode ratios")
    parser.add_argument("--feature-view", choices=["all", "local93", "local94", "local94_agg72", "all_plus_struct"], default="all",
                        help="E2/E3 structural ablation: local features, local+neighbour aggregates, or structural augmentation")
    parser.add_argument("--edge-world", choices=["real", "no_time", "empty", "perm", "rewire", "future"], default="real")
    parser.add_argument("--attack", choices=["none", "edge_delete", "high_degree_delete", "feature_flip"], default="none")
    parser.add_argument("--attack-budget", type=float, default=0.1)
    parser.add_argument("--aggregator", choices=["mean", "median", "trim"], default="mean")
    parser.add_argument("--label-protocol", choices=["A", "B", "C", "D"], default="A")
    parser.add_argument("--pu-prior", type=float, default=None, help="positive prior for PU protocol D; default = labelled illicit prevalence")
    parser.add_argument("--calibration", choices=["none", "temperature", "isotonic"], default="temperature")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-checkpoint", action="store_true", help="retain the last checkpoint after completion (useful for audit/resume tests)")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap samples for paired CI in smoke runs")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def prepare(args: argparse.Namespace) -> dict:
    if args.data_mode == "toy":
        graph = make_toy_data(seed=args.seed)
        dataset_version = "toy"
        paths = {}
    else:
        paths, dataset_version = ensure_data(
            args.data_dir,
            url=args.source_url,
            download=not args.no_download,
            allow_toy=args.allow_toy or args.data_mode == "auto",
        )
        if paths is None:
            graph = make_toy_data(seed=args.seed)
            dataset_version = "toy"
        else:
            graph = load_elliptic(paths)

    rng = rng_for(args.seed, f"edge-world:{args.edge_world}")
    edge_index, audit = build_edge_world(graph.edge_index, graph.time, args.edge_world, rng=rng)
    if args.strict_time and edge_index.size:
        # Strict boundary: drop same-time-step edges entirely. On native
        # Elliptic this leaves an empty graph (all edges have lag 0).
        keep = graph.time[edge_index[0]] < graph.time[edge_index[1]]
        n_dropped = int((~keep).sum())
        edge_index = edge_index[:, keep].copy()
        audit.n_future_edges_removed += n_dropped
        audit.n_edges_out = int(edge_index.shape[1])
        audit.causal_violations = int((graph.time[edge_index[0]] > graph.time[edge_index[1]]).sum()) if edge_index.size else 0
    if args.canonical_split != "none":
        split = canonical_split(graph.time, args.canonical_split)
        split_tag = args.canonical_split.replace("-", "_")
    elif args.split_mode == "random":
        split = random_split(graph.time, seed=args.seed)
        split_tag = "random"
    else:
        split = chronological_split(graph.time)
        split_tag = "chrono"
    save_split_manifest(
        str(Path(args.output_dir) / "manifest" / f"splits_{split_tag}_{args.seed}.json"),
        split,
    )

    use_time = not args.no_time_encoding
    x = graph.x
    # Feature-family ablation. The shipped CSV has 165 model columns =
    # 93 transaction/local features (public "94 local" minus the time step)
    # + 72 aggregated features. local93 is the transaction-scoped-only arm;
    # local94 is the legacy arm (93 local + 1 aggregate) kept for audit only.
    if args.feature_view == "local93":
        x = x[:, :93]
    elif args.feature_view == "local94":
        x = x[:, :94]
    elif args.feature_view == "local94_agg72":
        x = x[:, :94 + 71]
    elif args.feature_view == "all_plus_struct":
        deg = np.bincount(edge_index[0], minlength=graph.num_nodes).astype(np.float32) if edge_index.size else np.zeros(graph.num_nodes, dtype=np.float32)
        indeg = np.bincount(edge_index[1], minlength=graph.num_nodes).astype(np.float32) if edge_index.size else np.zeros(graph.num_nodes, dtype=np.float32)
        logdeg = np.log1p(deg + indeg)[:, None]
        x = np.concatenate([x, logdeg], axis=1).astype(np.float32)
    scaler = None
    if args.model not in TABULAR and not args.no_standardize:
        mu = x[split.train_idx].mean(axis=0)
        sigma = x[split.train_idx].std(axis=0)
        sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)
        x = ((x - mu) / sigma).astype(np.float32)
        scaler = {"fit_on": "train_nodes_only", "n_train": int(len(split.train_idx))}
    if use_time:
        x = np.concatenate([x, time_encoding(graph.time, args.time_k)], axis=1).astype(np.float32)

    degree = np.bincount(edge_index[0], minlength=graph.num_nodes).astype(np.float32) if edge_index.size else np.zeros(graph.num_nodes, dtype=np.float32)
    y_eff, label_mask = apply_label_protocol(graph.y, args.label_protocol)
    if args.label_protocol == "C":
        known_train = split.train_idx[graph.y[split.train_idx] >= 0]
        temp_model = fit_tabular("rf", x[known_train], graph.y[known_train], args.seed)
        unknown_idx = np.flatnonzero(graph.y < 0)
        if len(unknown_idx):
            pseudo_proba = predict_proba(temp_model, x[unknown_idx])
            y_eff[unknown_idx] = (pseudo_proba >= 0.8).astype(np.int64)
            label_mask[unknown_idx] = np.abs(pseudo_proba - 0.5) >= 0.3

    return {
        "graph": graph,
        "x": x,
        "y_eff": y_eff,
        "label_mask": label_mask,
        "edge_index": edge_index,
        "edge_audit": audit,
        "degree": degree,
        "split": split,
        "dataset_version": dataset_version,
        "paths": paths,
        "use_time_encoding": use_time,
        "feature_scaler": scaler,
    }


def run_config(args: argparse.Namespace) -> dict:
    relevant = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "layers": args.layers,
        "neighbors": args.neighbors,
        "alpha": args.alpha,
        "strict_time": args.strict_time,
        "degree_weight": args.degree_weight,
        "time_encoding": not args.no_time_encoding,
        "standardize": not args.no_standardize,
        "time_k": args.time_k,
        "edge_world": args.edge_world,
        "split_mode": args.split_mode,
        "canonical_split": args.canonical_split,
        "feature_view": args.feature_view,
        "attack": args.attack,
        "attack_budget": args.attack_budget,
        "aggregator": args.aggregator,
        "label_protocol": args.label_protocol,
        "pu_prior": args.pu_prior,
        "calibration": args.calibration,
        "threshold": args.threshold,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "pos_weight_mode": args.pos_weight_mode,
        "dropout": args.dropout,
        "keep_checkpoint": args.keep_checkpoint,
    }
    return relevant


def logit_from_probability(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def evaluate_scores(
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    test_time: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, np.ndarray]:
    calibrated_test, calibration_info = apply_calibration(args.calibration, val_logits, y_val, test_logits)
    y = y_test
    raw_metrics = binary_metrics(y, 1 / (1 + np.exp(-np.clip(test_logits, -30, 30))),
                                 (50, 100, 500), 10, args.threshold)
    calibrated_metrics = binary_metrics(y, calibrated_test, (50, 100, 500), 10, args.threshold)

    by_time = {}
    for step in np.unique(test_time):
        mask = test_time == step
        by_time[str(int(step))] = binary_metrics(y[mask], calibrated_test[mask], (50, 100), 10, args.threshold)
    slopes = []
    steps = sorted(int(k) for k in by_time)
    values = [by_time[str(k)]["pr_auc"] for k in steps]
    if len(values) >= 2 and all(np.isfinite(values)):
        slopes = np.polyfit(steps, values, 1).tolist()
    return (
        {"raw_threshold": raw_metrics, "calibrated": calibrated_metrics},
        {"by_time": by_time, "pr_auc_slope_per_step": slopes},
        calibration_info,
        calibrated_test,
    )


def train_tabular(prep: dict, args: argparse.Namespace, run_id: str) -> dict:
    start = time.time()
    split = prep["split"]
    y_eff, mask = prep["y_eff"], prep["label_mask"]
    x = prep["x"]
    train_mask = mask[split.train_idx] & (y_eff[split.train_idx] >= 0)
    if train_mask.sum() < 2:
        raise RuntimeError("Not enough labelled training nodes")

    if args.label_protocol == "B":
        eval_mask = np.ones_like(y_eff, dtype=bool)
    else:
        eval_mask = prep["graph"].y >= 0
    val_known = split.valid_idx[eval_mask[split.valid_idx]]
    test_known = split.test_idx[eval_mask[split.test_idx]]

    model = fit_tabular(args.model, x[split.train_idx[train_mask]], y_eff[split.train_idx[train_mask]], args.seed)
    val_prob = predict_proba(model, x[val_known])
    test_prob = predict_proba(model, x[test_known])
    metrics, drift, cal, calibrated_test = evaluate_scores(
        y_eff[val_known], y_eff[test_known], logit_from_probability(val_prob),
        logit_from_probability(test_prob), prep["graph"].time[test_known], args,
    )
    return {
        "run_id": run_id,
        "status": "done",
        "family": "tabular",
        "dataset_version": prep["dataset_version"],
        "model": args.model,
        "seed": args.seed,
        "config": run_config(args),
        "edge_audit": prep["edge_audit"].__dict__,
        "unknown_audit": unknown_distribution_audit(prep["graph"].x, prep["graph"].time, prep["graph"].y),
        "metrics": metrics,
        "drift": drift,
        "calibration": cal,
        "subgroup_degree": subgroup_metrics(prep["degree"][test_known], y_eff[test_known], calibrated_test),
        "subgroup_local_feature_proxy": subgroup_feature_proxy(
            prep["graph"].x[test_known], y_eff[test_known], calibrated_test
        ),
        "wall_time_seconds": time.time() - start,
        "n_labelled_train": int(train_mask.sum()),
        "n_labelled_valid": int(len(val_known)),
        "n_labelled_test": int(len(test_known)),
    }


def predict_gnn(model, sampler, prep, indices, args, rng, apply_attack: bool = False) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(prep["x"])
    degree = torch.from_numpy(prep["degree"])
    target_device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for roots in iterate_batches(indices, args.eval_batch_size, rng=rng, shuffle=False):
            batch = sampler.sample(roots, args.neighbors, rng)
            local_features = prep["x"][batch.local_nodes].copy()
            edges = batch.edge_index.copy()

            if apply_attack and args.attack != "none" and edges.size:
                src_global = batch.local_nodes[edges[0]]
                if args.attack == "edge_delete":
                    keep = rng.random(edges.shape[1]) >= args.attack_budget
                    edges = edges[:, keep]
                elif args.attack == "high_degree_delete":
                    degree_scores = prep["degree"][src_global]
                    cutoff = np.quantile(degree_scores, 1.0 - args.attack_budget)
                    edges = edges[:, degree_scores < cutoff]
                elif args.attack == "feature_flip":
                    neighbour_positions = np.arange(len(roots), len(batch.local_nodes))
                    if len(neighbour_positions):
                        width = min(10, local_features.shape[1])
                        mask = rng.random((len(neighbour_positions), width)) < args.attack_budget
                        block = local_features[neighbour_positions][:, :width]
                        block[mask] = -block[mask]
                        local_features[neighbour_positions][:, :width] = block

            xb = torch.from_numpy(local_features).to(target_device)
            edge_index = torch.from_numpy(edges).long().to(target_device)
            target = torch.from_numpy(batch.target_positions).long().to(target_device)
            deg = degree[batch.local_nodes].to(target_device)
            logits = model(xb, edge_index, target, deg)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_gnn(prep: dict, args: argparse.Namespace, run_id: str, result_path: Path) -> dict:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    split = prep["split"]
    y_eff, mask = prep["y_eff"], prep["label_mask"]
    is_pu = args.label_protocol == "D"
    if is_pu:
        train_idx = split.train_idx[mask[split.train_idx]]
    else:
        train_idx = split.train_idx[mask[split.train_idx] & (y_eff[split.train_idx] >= 0)]
    if args.label_protocol == "B":
        eval_mask = np.ones_like(y_eff, dtype=bool)
    else:
        eval_mask = prep["graph"].y >= 0
    val_idx = split.valid_idx[eval_mask[split.valid_idx]]
    test_idx = split.test_idx[eval_mask[split.test_idx]]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("A temporal partition has no labelled nodes")

    sampler = CausalNeighborSampler(prep["edge_index"], prep["graph"].time, alpha=args.alpha, strict=args.strict_time)
    in_dim = prep["x"].shape[1]
    model = build_gnn(
        args.model, in_dim, args.hidden, args.layers, args.dropout,
        args.degree_weight, args.aggregator,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos = int((y_eff[train_idx] == 1).sum())
    neg = int((y_eff[train_idx] == 0).sum())
    if args.pos_weight_mode == "none":
        weight_value = 1.0
    elif args.pos_weight_mode == "sqrt":
        weight_value = max(1.0, float(np.sqrt(neg / max(1, pos))))
    else:
        weight_value = max(1.0, neg / max(1, pos))
    pos_weight = torch.tensor([weight_value], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    pu_prior = args.pu_prior
    if pu_prior is None:
        pu_prior = pos / max(1, pos + neg)
    pu_prior = float(np.clip(pu_prior, 1e-4, 1 - 1e-4))
    pu_bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    checkpoint_path = Path(args.output_dir) / "checkpoints" / f"{run_id}.pt"
    start_epoch = 1
    best_score = -math.inf
    history: list[dict] = []
    best_state = None
    if checkpoint_path.exists() and not args.no_resume:
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        best_score = float(payload["best_score"])
        history = payload.get("history", [])
        best_state = payload.get("best_state")
        print(f"[RESUME] {run_id} from epoch {start_epoch}")

    train_rng = rng_for(args.seed, f"train:{run_id}")
    if checkpoint_path.exists() and not args.no_resume:
        train_rng = np.random.default_rng()
        train_rng.bit_generator.state = payload["numpy_rng_state"]
    eval_rng = rng_for(args.seed, f"eval:{run_id}")
    x_tensor = torch.from_numpy(prep["x"])
    degree_tensor = torch.from_numpy(prep["degree"])
    patience, no_improve = args.epochs, 0
    started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_rng.shuffle(train_idx)
        losses = []
        for roots in iterate_batches(train_idx, args.batch_size, rng=train_rng, shuffle=False):
            batch = sampler.sample(roots, args.neighbors, train_rng)
            xb = x_tensor[batch.local_nodes].to(device)
            ei = torch.from_numpy(batch.edge_index).long().to(device)
            target = torch.from_numpy(batch.target_positions).long().to(device)
            deg = degree_tensor[batch.local_nodes].to(device)
            root_labels = y_eff[roots]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, ei, target, deg)
            if is_pu:
                is_pos = torch.from_numpy(root_labels == 1).to(device)
                one = torch.ones_like(logits)
                zero = torch.zeros_like(logits)
                risk_p = pu_bce(logits[is_pos], one[is_pos]) if is_pos.any() else logits.sum() * 0.0
                risk_p_neg = pu_bce(logits[is_pos], zero[is_pos]) if is_pos.any() else logits.sum() * 0.0
                risk_u_neg = pu_bce(logits[~is_pos], zero[~is_pos]) if (~is_pos).any() else logits.sum() * 0.0
                loss = pu_prior * risk_p + torch.clamp(risk_u_neg - pu_prior * risk_p_neg, min=0.0)
            else:
                labels = torch.from_numpy(root_labels.astype(np.float32)).to(device)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_prob = predict_gnn(model, sampler, prep, val_idx, args, eval_rng)
        val_metrics = binary_metrics(y_eff[val_idx], val_prob, (50, 100), 10, args.threshold)
        score = val_metrics["pr_auc"] if np.isfinite(val_metrics["pr_auc"]) else val_metrics["f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pr_auc": float(score)})
        if score > best_score:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(checkpoint_path, {
            "run_id": run_id,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_score": best_score,
            "best_state": best_state,
            "history": history,
            "numpy_rng_state": train_rng.bit_generator.state,
            "config": run_config(args),
        })
        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_logits = logit_from_probability(predict_gnn(model, sampler, prep, val_idx, args, eval_rng))
    test_logits = logit_from_probability(
        predict_gnn(model, sampler, prep, test_idx, args, eval_rng, apply_attack=True)
    )
    metrics, drift, cal, calibrated_test = evaluate_scores(
        y_eff[val_idx], y_eff[test_idx], val_logits, test_logits,
        prep["graph"].time[test_idx], args,
    )
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    payload = {
        "run_id": run_id,
        "status": "done",
        "family": "gnn",
        "dataset_version": prep["dataset_version"],
        "model": args.model,
        "seed": args.seed,
        "config": run_config(args),
        "edge_audit": prep["edge_audit"].__dict__,
        "unknown_audit": unknown_distribution_audit(prep["graph"].x, prep["graph"].time, prep["graph"].y),
        "metrics": metrics,
        "drift": drift,
        "calibration": cal,
        "subgroup_degree": subgroup_metrics(prep["degree"][test_idx], y_eff[test_idx], calibrated_test),
        "subgroup_local_feature_proxy": subgroup_feature_proxy(
            prep["graph"].x[test_idx], y_eff[test_idx], calibrated_test
        ),
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "feature_standardization": prep["feature_scaler"],
        "pu_prior": pu_prior if is_pu else None,
        "pu_risk": "nnPU (SCAR sensitivity only; SAR not identified)" if is_pu else None,
        "epochs_run": len(history),
        "history": history,
        "wall_time_seconds": time.time() - started,
        "peak_memory_mb": peak_mb,
        "n_labelled_train": int(len(train_idx)),
        "n_labelled_valid": int(len(val_idx)),
        "n_labelled_test": int(len(test_idx)),
        "checkpoint": str(checkpoint_path),
    }
    atomic_write_json(result_path, payload)
    # Checkpoints are retained for audit and resume; deletion is intentionally
    # avoided so a bulk-delete guard can never break an otherwise valid run.
    return payload


def main() -> None:
    args = parse_args()
    if args.data_mode == "toy":
        args.allow_toy = True
    config = run_config(args)
    run_id = stable_hash({"config": config, "data_mode": args.data_mode, "source_url": args.source_url})[:12]
    result_path = Path(args.output_dir) / "results" / f"{run_id}.json"
    manifest_path = Path(args.output_dir) / "manifest" / f"{run_id}.json"

    if result_path.exists() and not args.no_resume:
        print(f"[SKIP] result already exists: {result_path}")
        print(json.dumps(read_json(result_path), indent=2, ensure_ascii=False)[:800])
        return

    prep = prepare(args)
    manifest = {
        "run_id": run_id,
        "dataset_version": prep["dataset_version"],
        "data_paths": {k: str(v) for k, v in prep["paths"].items()},
        "config": config,
        "seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed, "sampler": args.seed},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
        "command": " ".join(sys.argv),
    }
    atomic_write_json(manifest_path, manifest)

    if args.model in TABULAR:
        result = train_tabular(prep, args, run_id)
    else:
        result = train_gnn(prep, args, run_id, result_path)

    result["manifest"] = str(manifest_path)
    if args.model in TABULAR:
        atomic_write_json(result_path, result)
    leaderboard_path = Path(args.output_dir) / "results" / "leaderboard.json"
    leaderboard = read_json(leaderboard_path, {"runs": []})
    leaderboard.setdefault("runs", [])
    leaderboard["runs"] = [r for r in leaderboard["runs"] if r.get("run_id") != run_id]
    leaderboard["runs"].append({
        "run_id": run_id,
        "model": args.model,
        "dataset_version": prep["dataset_version"],
        "pr_auc": result["metrics"]["calibrated"]["pr_auc"],
        "brier": result["metrics"]["calibrated"]["brier"],
        "ece": result["metrics"]["calibrated"]["ece"],
        "edge_world": args.edge_world,
        "split_mode": args.split_mode,
        "label_protocol": args.label_protocol,
    })
    atomic_write_json(leaderboard_path, leaderboard)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
