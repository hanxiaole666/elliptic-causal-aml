#!/usr/bin/env python3
"""Build ONE canonical per-seed main table used by every v2 manuscript number.

Resolves reviewer H1: seed 0-4 have multiple variant runs in outputs/full, so
run-level averaging (0.492 for CausalSAGE) disagrees with paired per-seed
averaging (0.523). We define a single standard configuration and average runs
within each seed first, then across seeds. Every v2 number derives from here.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "outputs" / "full" / "results"
OUT = ROOT / "outputs" / "rebuttal" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ("pr_auc", "roc_auc", "brier", "ece", "precision@50", "precision@100", "precision@500")

# Standard main configuration: chrono, real edges, protocol A, no attack,
# default architecture (hidden 32 / 2 layers), no rebuttal-view modifications.
def is_standard(cfg: dict) -> bool:
    return (cfg.get("split_mode") in ("chrono", "random")
            and cfg.get("edge_world") == "real"
            and cfg.get("label_protocol", "A") == "A"
            and cfg.get("attack", "none") == "none"
            and cfg.get("canonical_split", "none") == "none"
            and cfg.get("feature_view", "all") == "all"
            and cfg.get("hidden", 32) == 32
            and cfg.get("layers", 2) == 2)


def main():
    by_model_split = defaultdict(lambda: defaultdict(list))
    for p in RESULTS.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "done" or r.get("dataset_version") == "toy":
            continue
        cfg = r.get("config", {})
        # Include both chrono and random split modes (random is a diagnostic ladder row).
        split_mode = cfg.get("split_mode", "chrono")
        standard = (split_mode in ("chrono", "random") and cfg.get("edge_world") == "real"
                    and cfg.get("label_protocol", "A") == "A" and cfg.get("attack", "none") == "none"
                    and cfg.get("canonical_split", "none") == "none" and cfg.get("feature_view", "all") == "all")
        if r.get("model") not in ("rf", "lightgbm", "xgboost", "mlp", "lr", "causal_sage", "graphsage"):
            continue
        if not standard:
            continue
        if r.get("model") in ("causal_sage", "graphsage") and not is_standard(cfg):
            continue
        key = (r["model"], split_mode)
        by_model_split[key][r["seed"]].append(r["metrics"]["calibrated"])

    rows = []
    for (model, split), per_seed in sorted(by_model_split.items()):
        seeds = sorted(per_seed)
        # Average runs WITHIN each seed, then across seeds (per-seed semantics).
        seed_means = []
        for s in seeds:
            runs = per_seed[s]
            seed_means.append({m: float(np.mean([r.get(m, np.nan) for r in runs])) for m in METRICS})
        row = {"model": model, "split_mode": split, "n_seeds": len(seeds),
               "seeds": ",".join(str(s) for s in seeds)}
        for m in METRICS:
            vals = [sm[m] for sm in seed_means if np.isfinite(sm[m])]
            row[m] = float(np.mean(vals)) if vals else float("nan")
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if m == "pr_auc" and len(vals) > 1:
                rng = np.random.default_rng(7)
                boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
                row["pr_auc_ci_low"], row["pr_auc_ci_high"] = np.percentile(boots, [2.5, 97.5])
        rows.append(row)
    df = pd.DataFrame(rows)
    dest = OUT / "Table1_unified_perseed.csv"
    df.to_csv(dest, index=False)

    # Paired delta causal_sage - rf, per-seed, common seeds.
    def seeds_for(model, split="chrono"):
        return {s: np.mean([r.get("pr_auc", np.nan) for r in by_model_split[(model, split)][s]])
                for s in by_model_split[(model, split)]}
    cs = seeds_for("causal_sage"); rf = seeds_for("rf")
    common = sorted(set(cs) & set(rf))
    diffs = np.array([cs[s] - rf[s] for s in common])
    paired = {
        "status": "ok", "n": len(common),
        "mean_rf": float(np.mean([rf[s] for s in common])),
        "mean_cs": float(np.mean([cs[s] for s in common])),
        "mean_delta": float(diffs.mean()),
        "per_seed_delta": {str(s): float(cs[s] - rf[s]) for s in common},
        "definition": "per-seed mean, averaged over runs within seed, then paired across seeds",
    }
    (OUT / "paired_effect_unified.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(df[["model", "split_mode", "n_seeds", "pr_auc", "pr_auc_std"]].to_string(index=False))
    print("\npaired:", {k: paired[k] for k in ("n", "mean_rf", "mean_cs", "mean_delta")})
    print("saved", dest)


if __name__ == "__main__":
    main()
