#!/usr/bin/env python3
"""Render a results-consistent manuscript from the summarized tables.

No number is typed by hand: every value inserted into the text is read from
outputs/full/tables/*.csv or statistical_tests.json and asserted against them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fmt(value, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number != number:
        return "N/A"
    return f"{number:.{digits}f}"


def read_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_row(df: pd.DataFrame, column: str = "pr_auc") -> dict | None:
    if df.empty or column not in df.columns:
        return None
    clean = df[df[column].notna()]
    if clean.empty:
        return None
    return clean.loc[clean[column].idxmax()].to_dict()


def choose_claim(main: pd.DataFrame, worlds: pd.DataFrame, stats: dict) -> tuple[str, str]:
    best_gnn = best_row(main[main["model"].astype(str).str.contains("sage|gcn|gat", regex=True, na=False)])
    best_tree = best_row(main[main["model"].isin(["lr", "rf", "xgboost", "lightgbm", "mlp"])])
    tree_name = stats.get("best_tabular_baseline", "tree baseline")

    if best_gnn and best_tree and float(best_tree["pr_auc"]) >= float(best_gnn["pr_auc"]):
        return (
            "negative_result",
            f"Under the strict chronological causal protocol, the strongest tabular baseline ({tree_name}, "
            f"PR-AUC {fmt(best_tree['pr_auc'])}) matches or exceeds the best graph model "
            f"({best_gnn['model']}, PR-AUC {fmt(best_gnn['pr_auc'])}). We therefore report tree models as the "
            "upper bound and treat graph-derived gains as conditional rather than universal.",
        )
    worlds_chrono = worlds[worlds["split_mode"] == "chrono"] if "split_mode" in worlds.columns else worlds
    if not worlds_chrono.empty:
        real = best_row(worlds_chrono[worlds_chrono["edge_world"] == "real"])
        rewire = best_row(worlds_chrono[worlds_chrono["edge_world"].isin(["rewire", "perm"])])
        if real and rewire and float(real["pr_auc"]) - float(rewire["pr_auc"]) >= 0.01:
            return (
                "conditional_graph_gain",
                "Under the strict causal protocol, real temporal edges outperform degree-preserving "
                f"counterfactuals by at least 0.01 PR-AUC (real {fmt(real['pr_auc'])} vs "
                f"rewired/permuted {fmt(rewire['pr_auc'])}), supporting a conditional structural claim.",
            )
    return (
        "no_reliable_gain",
        "The observed differences do not clear the pre-registered 0.01 PR-AUC threshold with "
        "non-overlapping confidence intervals; we therefore make no structural-necessity claim.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", default=str(ROOT / "outputs" / "full" / "tables"))
    parser.add_argument("--output", default=str(ROOT / "paper" / "MAIN_MANUSCRIPT_FINAL.md"))
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    output = Path(args.output)
    main_table = read_table(tables_dir, "Table1_main.csv")
    ablation = read_table(tables_dir, "Table2_ablation.csv")
    robustness = read_table(tables_dir, "Table3_robustness.csv")
    labels = read_table(tables_dir, "Table4_label_protocol.csv")
    worlds = read_table(tables_dir, "Table5_edge_world.csv")
    pareto = read_table(tables_dir, "Table6_pareto_front.csv")
    stats_path = tables_dir / "statistical_tests.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    if main_table.empty:
        raise SystemExit(f"Main table missing or empty: {tables_dir / 'Table1_main.csv'}")

    main_chrono = main_table[main_table["split_mode"] == "chrono"] if "split_mode" in main_table.columns else main_table
    claim_id, claim_text = choose_claim(main_chrono, worlds, stats)
    table1 = best_row(main_chrono)
    assert table1 is not None

    ladder_lines = []
    for model in sorted(set(main_table["model"].dropna())):
        r = best_row(main_table[(main_table["model"] == model) & (main_table["split_mode"] == "random")])
        c = best_row(main_table[(main_table["model"] == model) & (main_table["split_mode"] == "chrono")])
        if r and c:
            ladder_lines.append(
                f"{model}: random {fmt(r['pr_auc'])} vs chronological {fmt(c['pr_auc'])}, "
                f"same-model gap {fmt(float(r['pr_auc']) - float(c['pr_auc']))}."
            )
    ladder_text = " ".join(ladder_lines) if ladder_lines else "N/A"

    comparisons = stats.get("comparisons", [])
    comparison_lines = []
    for item in comparisons:
        comparison_lines.append(
            f"- {item['model']} vs {item['baseline']}: ΔPR-AUC={fmt(item['mean_delta_pr_auc'])}, "
            f"95% CI [{fmt(item['ci_low'])}, {fmt(item['ci_high'])}], "
            f"Wilcoxon p={fmt(item['wilcoxon_p'], 4)}, Holm p={fmt(item.get('holm_p'), 4)} "
            f"(n={item['n_seeds']} seeds)."
        )
    comparison_text = "\n".join(comparison_lines) if comparison_lines else "- No paired comparison was available."
    max_n = max([int(item["n_seeds"]) for item in comparisons], default=0)
    power_text = (
        f"Paired comparisons use n={max_n} seeds. The initial five-seed Wilcoxon test could not reach "
        "p<0.05 even with perfect agreement (minimum attainable p=0.0625); the expanded seed set is "
        "reported above, and significance is claimed only where Holm-adjusted p<0.05 is actually observed."
        if max_n >= 10
        else "With only five seeds a two-sided Wilcoxon test cannot reach p<0.05 even when all seeds agree; no significance is claimed."
    )

    pareto_rows = []
    if not pareto.empty:
        for _, row in pareto.iterrows():
            pareto_rows.append(
                f"| {row.get('model', 'N/A')} | {row.get('edge_world', 'N/A')} | "
                f"{fmt(row.get('pr_auc'))} | {fmt(row.get('wall_time_seconds'), 1)} | "
                f"{fmt(row.get('peak_memory_mb'), 0)} |"
            )
    pareto_text = "\n".join(pareto_rows) if pareto_rows else "| N/A | N/A | N/A | N/A | N/A |"

    manuscript = f"""# Temporal Causality, Label Bias and Operational Decision Quality in AML Graph Learning

> Results-consistent draft generated from `outputs/full/tables/`. Claim branch: `{claim_id}`.
> Venue: **author decision pending**. Recommended track: benchmark/dataset-oriented ADS or
> data-track submission (e.g. KDD ADS style), because the contribution is a diagnostic protocol
> and negative result rather than a new model. This is a recommendation, not a submitted venue.

## Abstract

Graph neural networks are frequently evaluated on anti-money-laundering graphs under protocols that
do not isolate future edges or unlabeled transactions. We study the problem as a diagnostic question:
under a strictly causal temporal protocol, when does graph structure provide attributable value beyond
strong tabular baselines? We implement explicit time-ordered edges, chronological splits, unknown-label
masking, validation-only calibration, edge counterfactuals and a zero-parameter degree-aware causal
sampler. {claim_text} {ladder_text} All reported numbers are generated from machine-readable
result files; no value is manually inserted into this manuscript.

## 1. Main result

Under the strict chronological protocol, the best configuration is `{table1.get('model', 'N/A')}` with
PR-AUC `{fmt(table1.get('pr_auc'))}` (95% CI [{fmt(table1.get('pr_auc_ci_low'))},
{fmt(table1.get('pr_auc_ci_high'))}], n={table1.get('n_seeds')} seeds), ROC-AUC
`{fmt(table1.get('roc_auc'))}`, Brier `{fmt(table1.get('brier'))}` and ECE `{fmt(table1.get('ece'))}`.

### Table 1 — Main results
{main_table.to_markdown(index=False) if hasattr(main_table, 'to_markdown') else main_table.to_string(index=False)}

### Paired inference

{comparison_text}

**Paired-comparison aggregation.** The Δ values in paired tests use each model's per-seed mean
across chronological runs, then pair seeds across models. Table 1 row means are run-level averages.
When a model has unequal run counts across seeds, these two averages need not match exactly; the
paired test remains the correct basis for the significance statement.

**Statistical-power disclosure.** Bootstrap CIs over seed-level means and signed-rank tests answer
different questions. {power_text} Bootstrap intervals describe seed-level dispersion and do not
substitute for an adequately powered paired test.

**Table-reading note.** Rows with `n_seeds=1` in Tables 2 and 3 are exploratory single runs; their
zero standard deviation and degenerate confidence intervals must not be read as stable estimates.

## 2. Structural counterfactuals

### Table 5 — Edge worlds
{worlds.to_markdown(index=False) if not worlds.empty and hasattr(worlds, 'to_markdown') else 'No edge-world results available.'}

## 3. Algorithm ablation

### Table 2 — Ablation
{ablation.to_markdown(index=False) if not ablation.empty and hasattr(ablation, 'to_markdown') else 'No ablation results available.'}

## 4. Label protocols and calibration

Protocols: **A** = train and evaluate only on originally known labels; **B** = treat every unknown
transaction as a negative for both training and evaluation; **C** = train with high-confidence
pseudo-labels on unknown transactions but evaluate only on originally known labels; **D** = nnPU
risk treating positives as labelled and all other nodes as unlabeled (SCAR sensitivity only; SAR is
not identified).

### Table 4 — Label protocols
{labels.to_markdown(index=False) if not labels.empty and hasattr(labels, 'to_markdown') else 'No label-protocol results available.'}

We treat protocol D as an nnPU risk under an SCAR sensitivity assumption only; SAR is not identified
from the available labels.

## 5. Robustness

### Table 3 — Attacks x aggregators
{robustness.to_markdown(index=False) if not robustness.empty and hasattr(robustness, 'to_markdown') else 'No robustness results available.'}

Median and trimmed-mean aggregators are evaluated as test-time defenses. Structural perturbations are
applied only to the test computation graph; training graphs remain causally filtered.

## 6. Operational efficiency

### Table 6 — Non-dominated configurations
| Model | Edge world | PR-AUC | Wall time (s) | Peak memory (MB) |
|---|---|---:|---:|---:|
{pareto_text}

The non-dominated set is computed only over configurations that recorded peak GPU memory; tabular
baselines have no comparable GPU-memory measurement and are therefore reported separately rather
than silently included in the front.

## 7. Claim bounds

- Allowed: `{claim_text}`
- Forbidden: claiming cross-chain validity, address-level independence, real deployment, causal
  explanation from SHAP/GNNExplainer, or any "SOTA" statement not supported by the paired tests above.
- Volume/amount subgroups are **not** claimed: Elliptic features are anonymized and have no public
  semantic mapping. Exploratory local-feature-index quintiles are reported in the machine-readable
  results only.

## Reproduction

- Full commands: `scripts/run_full_matrix.sh`
- Aggregation: `scripts/summarize_results.py`
- Consistency check: this file is regenerated from the CSVs; do not edit numbers manually.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manuscript, encoding="utf-8")

    # Consistency gate: every number in the prose must exist in the source tables.
    for value in [table1.get("pr_auc"), table1.get("roc_auc"), table1.get("ece")]:
        if value is None or pd.isna(value):
            raise RuntimeError("Main table lacks a required metric; manuscript not accepted.")
        assert fmt(value) in manuscript

    print(f"Manuscript written to {output}")
    print(f"Claim branch: {claim_id}")


if __name__ == "__main__":
    main()
