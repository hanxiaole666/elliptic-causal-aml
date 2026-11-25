#!/usr/bin/env python3
"""Check that LaTeX numeric macros match the machine-readable result tables."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "submission" / "tdsc" / "numbers.tex"
TABLES = ROOT / "outputs" / "full" / "tables"


def read_macros(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    return {name: float(value) for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{([-+]?\d+(?:\.\d+)?)\}", text)}


def main():
    macros = read_macros(NUMBERS)
    main = pd.read_csv(TABLES / "Table1_main.csv")
    worlds = pd.read_csv(TABLES / "Table5_edge_world.csv")
    stats = json.loads((TABLES / "statistical_tests.json").read_text(encoding="utf-8"))
    comparisons = {c["model"]: c for c in stats["comparisons"]}

    checks = {
        "NumCSChronic": main[(main.model == "causal_sage") & (main.split_mode == "chrono")].iloc[0]["pr_auc"],
        "NumCSRandom": main[(main.model == "causal_sage") & (main.split_mode == "random")].iloc[0]["pr_auc"],
        "NumRFPrAUC": main[(main.model == "rf") & (main.split_mode == "chrono")].iloc[0]["pr_auc"],
        "NumLGBMChronic": main[(main.model == "lightgbm") & (main.split_mode == "chrono")].iloc[0]["pr_auc"],
        "NumLGBMRandom": main[(main.model == "lightgbm") & (main.split_mode == "random")].iloc[0]["pr_auc"],
        "NumEdgeReal": worlds[(worlds.model == "causal_sage") & (worlds.split_mode == "chrono") & (worlds.edge_world == "real")].iloc[0]["pr_auc"],
        "NumEdgeRewire": worlds[(worlds.model == "causal_sage") & (worlds.split_mode == "chrono") & (worlds.edge_world == "rewire")].iloc[0]["pr_auc"],
        "NumDeltaCSRf": comparisons["causal_sage"]["mean_delta_pr_auc"],
        "NumDeltaCSRfHolm": comparisons["causal_sage"]["holm_p"],
    }
    problems = []
    for macro, expected in checks.items():
        actual = macros[macro]
        if abs(actual - expected) > 0.0006:
            problems.append({"macro": macro, "expected": expected, "actual": actual})

    if problems:
        raise SystemExit(json.dumps({"status": "failed", "problems": problems}, indent=2))
    print(json.dumps({"status": "ok", "checked_macros": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
