#!/usr/bin/env python3
"""V4 figures whose numeric labels are read from the SAME numbers.tex that
drives the manuscript tables and abstract. This guarantees figure==table==
abstract by construction (third-round requirement 3.1).

A cross-check JSON records every plotted value so automated tests can verify.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "submission" / "tifs_v4" / "numbers.tex"
OUT = ROOT / "submission" / "tifs_v4" / "figures"
CHECK = ROOT / "outputs" / "v4" / "tables" / "figure_values.json"
OUT.mkdir(parents=True, exist_ok=True)
CHECK.parent.mkdir(parents=True, exist_ok=True)

macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", NUMBERS.read_text(encoding="utf-8")))


def val(name):
    raw = macros[name]
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"macro {name} is not numeric: {raw}")


plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
record = {}

# ---- Figure 1: same-step boundary (values from Table I macros) ----
f1 = {
    "labels": ["Non-future-step\n(full graph)", "Strict causal\n(empty graph)", "Empty-world\ncontrol"],
    "values": [val("NumCSChronic"), val("NumStrictPr"), val("NumEmptyPr")],
    "source_macros": ["NumCSChronic", "NumStrictPr", "NumEmptyPr"],
}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
xs = np.arange(3)
bars = ax.bar(xs, f1["values"], color=["#2980b9", "#c0392b", "#95a5a6"], alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(f1["labels"], fontsize=7)
ax.set_ylabel("PR-AUC")
ax.set_title("Same-step edge boundary", pad=14)
ax.set_ylim(0, max(f1["values"]) * 1.28)  # headroom for labels, away from title
for i, b in enumerate(bars):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012, f"{f1['values'][i]:.3f}",
            ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "same_step_boundary.png"); plt.close(fig)
record["same_step_boundary"] = f1

# ---- Figure 2: contract ladder (values from Table V macros) ----
f2 = {
    "labels": ["Random\nknown-only", "Chronological\n(I1)", "Unknown-as-neg\n(I3)", "Test-calibration\n(I4)"],
    "values": [val("NumLadOne"), val("NumLadTwo"), val("NumLadThree"), val("NumLadFour")],
    "source_macros": ["NumLadOne", "NumLadTwo", "NumLadThree", "NumLadFour"],
}
fig, ax = plt.subplots(figsize=(3.6, 2.3))
xs = np.arange(4)
bars = ax.bar(xs, f2["values"], color=["#c0392b", "#2980b9", "#7f8c8d", "#7f8c8d"], alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(f2["labels"], fontsize=6.5)
ax.set_ylabel("PR-AUC")
ax.set_title("Contract ladder (CausalSAGE)", pad=14)
ax.set_ylim(0, max(f2["values"]) * 1.28)
for i, b in enumerate(bars):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012, f"{f2['values'][i]:.3f}",
            ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "contract_ladder.png"); plt.close(fig)
record["contract_ladder"] = f2

# ---- Figure 3: calibration summary (measured ECE/Brier from Table I macros) ----
f3 = {
    "models": ["Random Forest", "LightGBM", "CausalSAGE"],
    "ece": [val("NumRFECE"), val("NumLGBMECE"), val("NumCSEce")],
    "brier": [val("NumRFBrier"), val("NumLGBMBrier"), val("NumCSBrier")],
}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
xs = np.arange(3); w = 0.35
ax.bar(xs - w/2, f3["ece"], w, label="ECE (15 bins)", color="#2980b9")
ax.bar(xs + w/2, f3["brier"], w, label="Brier", color="#e67e22")
ax.set_xticks(xs); ax.set_xticklabels(f3["models"], fontsize=7)
ax.set_ylabel("Score (lower is better)"); ax.set_title("Calibration summary", pad=14)
ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "calibration_summary.png"); plt.close(fig)
record["calibration_summary"] = f3

(CHECK).write_text(json.dumps(record, indent=2), encoding="utf-8")
print("figures drawn from numbers.tex macros:")
for k, v in record.items():
    print(" ", k, v.get("values", v.get("ece")))
print("saved", CHECK)
