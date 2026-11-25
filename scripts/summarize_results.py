#!/usr/bin/env python3
"""Turn run-level results.json files into paper tables, statistics and figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.stats import holm_adjust, wilcoxon_paired  # noqa: E402
from src.utils.io import atomic_write_json  # noqa: E402

METRICS = ("pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "ece", "precision@50", "precision@100", "precision@500")


def load_runs(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "leaderboard.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "done" or payload.get("dataset_version") == "toy":
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def flatten(run: dict) -> dict:
    cfg = run.get("config", {})
    calibrated = run.get("metrics", {}).get("calibrated", {})
    row = {
        "run_id": run.get("run_id"),
        "dataset_version": run.get("dataset_version"),
        "model": run.get("model"),
        "seed": run.get("seed"),
        "split_mode": cfg.get("split_mode", "chrono"),
        "edge_world": cfg.get("edge_world", "real"),
        "label_protocol": cfg.get("label_protocol", "A"),
        "attack": cfg.get("attack", "none"),
        "attack_budget": cfg.get("attack_budget", 0.0),
        "aggregator": cfg.get("aggregator", "mean"),
        "alpha": cfg.get("alpha"),
        "degree_weight": cfg.get("degree_weight"),
        "time_encoding": cfg.get("time_encoding"),
        "layers": cfg.get("layers"),
        "hidden": cfg.get("hidden"),
        "neighbors": str(cfg.get("neighbors")),
        "parameters": run.get("parameters"),
        "wall_time_seconds": run.get("wall_time_seconds"),
        "peak_memory_mb": run.get("peak_memory_mb"),
        "n_test": run.get("n_labelled_test"),
    }
    for metric in METRICS:
        row[metric] = calibrated.get(metric, np.nan)
    return row


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def aggregate_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=keys + ["n_seeds"] + list(METRICS))
    grouped = df.groupby(keys, dropna=False)
    records = []
    for key, group in grouped:
        if isinstance(key, tuple):
            row = {column: value for column, value in zip(keys, key)}
        else:
            row = {keys[0]: key}
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[metric] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if metric == "pr_auc":
                low, high = bootstrap_ci_mean(values)
                row["pr_auc_ci_low"], row["pr_auc_ci_high"] = low, high
        records.append(row)
    return pd.DataFrame(records).sort_values(keys).reset_index(drop=True)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    points = df[["run_id", "model", "edge_world", "pr_auc", "wall_time_seconds", "peak_memory_mb"]].dropna()
    front = []
    for _, candidate in points.iterrows():
        dominated = False
        for _, other in points.iterrows():
            if candidate["run_id"] == other["run_id"]:
                continue
            better_or_equal = (
                other["pr_auc"] >= candidate["pr_auc"]
                and other["wall_time_seconds"] <= candidate["wall_time_seconds"]
                and other["peak_memory_mb"] <= candidate["peak_memory_mb"]
            )
            strictly_better = (
                other["pr_auc"] > candidate["pr_auc"]
                or other["wall_time_seconds"] < candidate["wall_time_seconds"]
                or other["peak_memory_mb"] < candidate["peak_memory_mb"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate.to_dict())
    return pd.DataFrame(front)


def statistical_tests(df: pd.DataFrame) -> dict:
    clean = df[df["split_mode"] == "chrono"]
    models = sorted(clean["model"].dropna().unique())
    tabular = [m for m in models if m in {"lr", "rf", "xgboost", "lightgbm", "mlp"}]
    if not tabular:
        return {"status": "not_enough_baselines"}
    best_baseline = clean[clean["model"].isin(tabular)].groupby("model")["pr_auc"].mean().idxmax()
    baseline = clean[clean["model"] == best_baseline].groupby("seed")["pr_auc"].mean()
    tests = []
    for model in models:
        if model == best_baseline:
            continue
        target = clean[clean["model"] == model].groupby("seed")["pr_auc"].mean()
        common = sorted(set(target.index) & set(baseline.index))
        if len(common) < 2:
            continue
        a, b = target.loc[common].to_numpy(), baseline.loc[common].to_numpy()
        diff = a - b
        low, high = bootstrap_ci_mean(diff, n_boot=2000, seed=17)
        tests.append({
            "model": model,
            "baseline": best_baseline,
            "n_seeds": len(common),
            "mean_delta_pr_auc": float(diff.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "wilcoxon_p": wilcoxon_paired(a, b),
        })
    if tests:
        adjusted = holm_adjust([t["wilcoxon_p"] for t in tests])
        for test, p in zip(tests, adjusted):
            test["holm_p"] = float(p)
    return {
        "status": "ok",
        "best_tabular_baseline": best_baseline,
        "aggregation_note": "Paired deltas use per-seed means across chronological runs; Table 1 means are run-level averages and may differ slightly when run counts per seed are unequal.",
        "comparisons": tests,
    }


def write_figures(output_dir: Path, df: pd.DataFrame, runs: list[dict]) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    drift_rows = []
    for run in runs:
        by_time = run.get("drift", {}).get("by_time", {})
        for step, payload in by_time.items():
            drift_rows.append({
                "run_id": run["run_id"],
                "model": run["model"],
                "seed": run["seed"],
                "step": int(step),
                "pr_auc": payload.get("pr_auc"),
                "ece": payload.get("ece"),
            })
    pd.DataFrame(drift_rows).to_csv(figure_dir / "drift.csv", index=False)

    pareto_df = df[["run_id", "model", "edge_world", "pr_auc", "wall_time_seconds", "peak_memory_mb", "parameters"]].copy()
    pareto_df.to_csv(figure_dir / "pareto.csv", index=False)

    subgroup_rows = []
    for run in runs:
        for group, payload in run.get("subgroup_degree", {}).items():
            subgroup_rows.append({
                "run_id": run["run_id"], "model": run["model"], "group": group,
                "n": payload.get("n"), "positive_n": payload.get("positive_n"),
                "pr_auc": payload.get("pr_auc"), "precision@100": payload.get("precision@100"),
                "ece": payload.get("ece"),
            })
    pd.DataFrame(subgroup_rows).to_csv(figure_dir / "subgroup_degree.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        drift = pd.DataFrame(drift_rows)
        if not drift.empty:
            for model, group in drift.groupby("model"):
                agg = group.groupby("step")["pr_auc"].mean()
                plt.plot(agg.index, agg.values, marker="o", label=str(model))
            plt.xlabel("time step")
            plt.ylabel("PR-AUC")
            plt.title("Out-of-time PR-AUC drift")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(figure_dir / "drift.png", dpi=150)
            plt.close()

        pareto = pd.read_csv(figure_dir / "pareto.csv")
        if not pareto.empty and pareto["peak_memory_mb"].notna().any():
            plt.scatter(pareto["wall_time_seconds"], pareto["pr_auc"], s=24)
            plt.xlabel("wall time (s)")
            plt.ylabel("PR-AUC")
            plt.title("Accuracy-time Pareto")
            plt.tight_layout()
            plt.savefig(figure_dir / "pareto.png", dpi=150)
            plt.close()
    except Exception as exc:
        print(f"[WARN] figure rendering skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--tables-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = Path(args.tables_dir) if args.tables_dir else output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(output_dir / "results")
    if not runs:
        raise SystemExit(f"No real-data results found under {output_dir / 'results'}")
    df = pd.DataFrame([flatten(run) for run in runs])

    aggregate_table(df, ["model", "split_mode"]).to_csv(tables_dir / "Table1_main.csv", index=False)
    aggregate_table(df, ["split_mode", "model", "alpha", "degree_weight", "time_encoding", "aggregator", "layers", "hidden", "neighbors"]).to_csv(
        tables_dir / "Table2_ablation.csv", index=False
    )
    aggregate_table(df, ["split_mode", "model", "attack", "attack_budget", "aggregator"]).to_csv(
        tables_dir / "Table3_robustness.csv", index=False
    )
    aggregate_table(df, ["split_mode", "model", "label_protocol"]).to_csv(tables_dir / "Table4_label_protocol.csv", index=False)
    aggregate_table(df, ["split_mode", "model", "edge_world"]).to_csv(tables_dir / "Table5_edge_world.csv", index=False)

    pareto_df = pareto_front(df[df["split_mode"] == "chrono"])
    pareto_df.to_csv(tables_dir / "Table6_pareto_front.csv", index=False)
    stats = statistical_tests(df)
    atomic_write_json(tables_dir / "statistical_tests.json", stats)
    write_figures(output_dir, df, runs)

    manifest = {
        "n_runs": len(runs),
        "dataset_versions": sorted({run["dataset_version"] for run in runs}),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "tables": [p.name for p in sorted(tables_dir.glob("*.csv"))],
        "best_tabular_baseline": stats.get("best_tabular_baseline"),
    }
    atomic_write_json(tables_dir / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
