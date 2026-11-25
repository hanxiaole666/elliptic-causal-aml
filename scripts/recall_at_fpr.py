#!/usr/bin/env python3
"""C3: recall at fixed FPR from an existing C1 checkpoint (no retraining).

Loads the registered C1-chrono checkpoint, rebuilds the same data/split via
run_pipeline.prepare, recomputes test scores with predict_gnn, and reports
recall at FPR = 1%, 5%, 10% over known-label test nodes. This is an operational
metric complement to P@100 and U@0.5, both of which are uninformative at a
~2.5% base rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.run_pipeline import parse_args, prepare, predict_gnn  # noqa: E402
from src.models.gnn import build_gnn  # noqa: E402
from src.sampling.causal_sampler import CausalNeighborSampler  # noqa: E402

OUT = ROOT / "outputs" / "v7" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# Find the registered C1-chrono run (known-label protocol A, elliptic).
runs = []
for p in (ROOT / "outputs" / "v5" / "results").glob("*.json"):
    r = json.loads(p.read_text(encoding="utf-8"))
    c = r.get("config", {})
    if (r.get("model") == "causal_sage" and c.get("split_mode") == "chrono"
            and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none"
            and r.get("dataset_version") == "elliptic" and r.get("status") == "done"
            and r.get("checkpoint")):
        runs.append(r)
if not runs:
    raise SystemExit("no registered C1-chrono checkpoint found")
run = sorted(runs, key=lambda r: r["seed"])[0]
cfg = run["config"]
ckpt = run["checkpoint"]

argv = [
    "--data-mode", "elliptic", "--model", str(cfg["model"]),
    "--seed", str(cfg["seed"]), "--epochs", str(cfg["epochs"]),
    "--batch-size", str(cfg["batch_size"]), "--hidden", str(cfg["hidden"]),
    "--layers", str(cfg["layers"]), "--neighbors", *[str(n) for n in cfg["neighbors"]],
    "--alpha", str(cfg["alpha"]), "--time-k", str(cfg["time_k"]),
    "--split-mode", str(cfg["split_mode"]), "--canonical-split", str(cfg["canonical_split"]),
    "--feature-view", str(cfg["feature_view"]), "--edge-world", str(cfg["edge_world"]),
    "--attack", str(cfg["attack"]), "--attack-budget", str(cfg["attack_budget"]),
    "--aggregator", str(cfg["aggregator"]), "--label-protocol", str(cfg["label_protocol"]),
    "--calibration", str(cfg["calibration"]), "--threshold", str(cfg["threshold"]),
    "--lr", str(cfg["lr"]), "--weight-decay", str(cfg["weight_decay"]),
    "--patience", str(cfg["patience"]), "--pos-weight-mode", str(cfg["pos_weight_mode"]),
    "--dropout", str(cfg["dropout"]), "--device", "cpu", "--keep-checkpoint",
    "--output-dir", str(ROOT / "outputs" / "v5"),
]
if cfg["strict_time"]: argv.append("--strict-time")
if cfg["degree_weight"]: argv.append("--degree-weight")
if not cfg["time_encoding"]: argv.append("--no-time-encoding")
if not cfg["standardize"]: argv.append("--no-standardize")
if cfg.get("pu_prior") is not None: argv += ["--pu-prior", str(cfg["pu_prior"])]
old_argv = sys.argv
sys.argv = ["run_pipeline.py"] + argv
try:
    args = parse_args()
finally:
    sys.argv = old_argv

prep = prepare(args)
sampler = CausalNeighborSampler(prep["edge_index"], prep["graph"].time,
                                alpha=args.alpha, strict=args.strict_time)
model = build_gnn(cfg["model"], prep["x"].shape[1], cfg["hidden"], cfg["layers"],
                  cfg["dropout"], cfg["degree_weight"], cfg["aggregator"])
state = torch.load(ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(state["best_state"])
rng = np.random.default_rng(cfg["seed"] + 999)
test_idx = prep["split"].test_idx
y = prep["y_eff"]
mask = np.isin(test_idx, np.flatnonzero(prep["label_mask"]))
idx = test_idx[mask]
scores = predict_gnn(model, sampler, prep, idx, args, rng)
labels = y[idx].astype(int)

pos, neg = scores[labels == 1], scores[labels == 0]
rows = []
for fpr in (0.01, 0.05, 0.10):
    # threshold on negatives: recall = fraction of positives above it
    thr = np.quantile(neg, 1.0 - fpr)
    recall = float((pos >= thr).mean())
    rows.append({"fpr": fpr, "n_neg": int(len(neg)), "n_pos": int(len(pos)),
                 "threshold": round(float(thr), 4), "recall": round(recall, 4)})
out = {"run_id": run["run_id"], "seed": cfg["seed"],
       "source_checkpoint": str(Path(ckpt).name), "protocol": "C1 chrono / protocol A",
       "definition": "FPR measured over known-label test negatives; recall over known-label test positives",
       "rows": rows}
(OUT / "recall_at_fpr.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
