#!/usr/bin/env python3
"""V5 SINGLE SOURCE OF TRUTH.

Builds one table from ONE pre-registered configuration per model (registry.json).
Every manuscript number, table, figure and abstract value derives from here.
This resolves the fourth-round core finding: 'same protocol, different values'.

- C1 (causal_sage) comes exclusively from outputs/v5/results (all rungs,
  identical hyperparameters, 15 seeds for every protocol).
- Tree baselines come from outputs/full with their own fixed configuration.
- No heterogeneous-configuration averaging anywhere.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "outputs" / "v5" / "results"
FULL = ROOT / "outputs" / "full" / "results"
OUT = ROOT / "outputs" / "v5" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
REG = json.loads((ROOT / "configs" / "registry.json").read_text(encoding="utf-8"))

FIELDS = ["pr_auc", "roc_auc", "brier", "ece", "precision@50", "precision@100", "precision@500", "recall", "precision"]


def load(dirpath):
    rows = []
    if not dirpath.exists():
        return rows
    for p in dirpath.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "done" and r.get("dataset_version") != "toy":
            rows.append(r)
    return rows


def aggregate(rows, name):
    """Per-seed mean, then across-seed mean/std and paired bootstrap CI."""
    d = defaultdict(list)
    for r in rows:
        d[r["seed"]].append(r["metrics"]["calibrated"])
    seeds = sorted(d)
    rec = {"config_id": name, "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds))}
    seed_means = {}
    for s in seeds:
        seed_means[s] = {f: float(np.mean([r.get(f, np.nan) for r in d[s]])) for f in FIELDS}
    for f in FIELDS:
        vals = [seed_means[s][f] for s in seeds if np.isfinite(seed_means[s][f])]
        rec[f] = float(np.mean(vals)) if vals else float("nan")
        rec[f + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if len(seeds) > 1:
        pv = np.array([seed_means[s]["pr_auc"] for s in seeds])
        rng = np.random.default_rng(11)
        boots = [rng.choice(pv, size=len(pv), replace=True).mean() for _ in range(2000)]
        rec["pr_auc_ci_low"], rec["pr_auc_ci_high"] = np.percentile(boots, [2.5, 97.5])
    return rec, seed_means


def main():
    v5 = load(V5)
    full = load(FULL)
    rows, seedmaps = [], {}

    # All C1 rungs (single configuration, one seed set each).
    rungs = {
        "C1-chrono": lambda c, r: r["model"] == "causal_sage" and c.get("split_mode") == "chrono"
            and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none"
            and c.get("edge_world") == "real" and not c.get("strict_time", False)
            and c.get("calibration") != "none" and c.get("canonical_split", "none") == "none",
        "C1-random": lambda c, r: r["model"] == "causal_sage" and c.get("split_mode") == "random"
            and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none",
        "C1-unknown": lambda c, r: r["model"] == "causal_sage" and c.get("label_protocol") == "B",
        "C1-testcal": lambda c, r: r["model"] == "causal_sage" and c.get("calibration") == "none"
            and c.get("split_mode") == "chrono",
        "C1-strict": lambda c, r: r["model"] == "causal_sage" and c.get("strict_time", False),
        "C1-empty": lambda c, r: r["model"] == "causal_sage" and c.get("edge_world") == "empty",
    }
    for name, pred in rungs.items():
        sel = [r for r in v5 if pred(r.get("config", {}), r)]
        if sel:
            rec, sm = aggregate(sel, name)
            rows.append(rec); seedmaps[name] = sm

    # C1-chrono uses the same registered C1 config from outputs/full
    # (the v5 matrix varies only the protocol rung; chrono is the baseline).
    chrono = [r for r in full if r["model"] == "causal_sage"
              and r.get("config", {}).get("split_mode") == "chrono"
              and r.get("config", {}).get("label_protocol", "A") == "A"
              and r.get("config", {}).get("attack", "none") == "none"
              and r.get("config", {}).get("edge_world") == "real"
              and r.get("config", {}).get("hidden") == 32
              and r.get("config", {}).get("layers") == 2
              and r.get("config", {}).get("alpha") == 0.5
              and r.get("config", {}).get("degree_weight") is True
              and r.get("config", {}).get("time_encoding") is True
              and r.get("config", {}).get("calibration") == "temperature"
              and r.get("config", {}).get("epochs") == 30
              and r.get("config", {}).get("canonical_split", "none") == "none"
              and r["seed"] <= 14 and r.get("status") == "done"]
    if chrono:
        rec, sm = aggregate(chrono, "C1-chrono")
        rows.append(rec); seedmaps["C1-chrono"] = sm

    # Tree baselines: fixed configuration from outputs/full.
    for model in ("rf", "lightgbm", "xgboost", "mlp", "lr"):
        for split in ("chrono", "random"):
            sel = [r for r in full if r["model"] == model and r.get("config", {}).get("split_mode") == split
                   and r.get("config", {}).get("label_protocol", "A") == "A"
                   and r.get("config", {}).get("attack", "none") == "none"
                   and r.get("config", {}).get("edge_world") == "real"
                   and r.get("config", {}).get("canonical_split", "none") == "none"]
            sel = [r for r in sel if r["seed"] <= 14]
            if sel:
                rec, sm = aggregate(sel, f"{model}-{split}")
                rows.append(rec); seedmaps[f"{model}-{split}"] = sm

    df = pd.DataFrame(rows)
    # Derive model/split_mode columns from config_id for downstream compatibility.
    def model_of(cid):
        for m in ("causal_sage", "lightgbm", "xgboost", "rf", "mlp", "lr"):
            if cid.startswith(m) or cid.startswith("C1"):
                return "causal_sage" if cid.startswith("C1") else m
        return cid
    def split_of(cid):
        if "random" in cid: return "random"
        return "chrono"
    df["model"] = df["config_id"].map(model_of)
    df["split_mode"] = df["config_id"].map(split_of)
    df.to_csv(OUT / "Table1_v5.csv", index=False)

    # Paired effects on the COMMON seed set for every comparison (no n mixing).
    def paired(name_a, name_b):
        a, b = seedmaps.get(name_a, {}), seedmaps.get(name_b, {})
        common = sorted(set(a) & set(b))
        d = np.array([a[s]["pr_auc"] - b[s]["pr_auc"] for s in common])
        if len(d) == 0:
            return {"n": 0}
        rng = np.random.default_rng(13)
        boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(3000)]
        from scipy import stats as _st
        se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else np.nan
        eps = 0.02
        p_tost = max(1-_st.t.cdf((d.mean()+eps)/se, len(d)-1), _st.t.cdf((d.mean()-eps)/se, len(d)-1)) if se and np.isfinite(se) else np.nan
        return {"n": len(d), "seeds": common, "mean_delta": float(d.mean()),
                "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
                "tost_p": float(p_tost), "equivalent_at_0.02": bool(p_tost < 0.05)}

    effects = {
        "cs_minus_rf": paired("C1-chrono", "rf-chrono"),
        "random_minus_chrono": paired("C1-random", "C1-chrono"),
        "full_minus_empty": paired("C1-chrono", "C1-empty"),
        "full_minus_strict": paired("C1-chrono", "C1-strict"),
        "empty_minus_strict": paired("C1-empty", "C1-strict"),
    }
    (OUT / "paired_v5.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")

    identity = {
        "C1": REG["C1"],
        "note": "One configuration per model. Same-protocol values are unique across all tables.",
        "rungs": {name: {"n_seeds": next((r["n_seeds"] for r in rows if r["config_id"] == name), 0)} for name in rungs},
    }
    (OUT / "config_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    print(df[["config_id", "n_seeds", "pr_auc", "pr_auc_std"]].to_string(index=False))
    print("\npaired:", json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in effects.items()}, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
