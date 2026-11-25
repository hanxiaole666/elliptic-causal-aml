#!/usr/bin/env python3
"""V5 figures drawn from the SAME numbers.tex macros as tables/abstract.

Fixes fourth-round layout complaints:
  F1: error bars present (matches caption), n annotated from macros.
  F2: horizontal labels (no crowding).
  F3: ylim auto-scaled (no clipped bar); reliability diagram from the
      measured I4 same-score experiment instead of an invented curve.
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
NUMBERS = ROOT / "submission" / "tifs_v14" / "numbers.tex"
OUT = ROOT / "submission" / "tifs_v14" / "figures"
CHECK = ROOT / "outputs" / "v7" / "tables" / "figure_values.json"
OUT.mkdir(parents=True, exist_ok=True)
CHECK.parent.mkdir(parents=True, exist_ok=True)

macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", NUMBERS.read_text(encoding="utf-8")))
def val(name):
    try: return float(macros[name])
    except (KeyError, ValueError): raise RuntimeError(f"macro {name} not numeric: {macros.get(name)}")

plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
record = {}

# ---- F1: same-step boundary, WITH error bars from macros ----
f1 = {
    "labels": ["Full graph\n(non-future-step)", "Zero-edge control\n(C1-E)"],
    "values": [val("NumCSChronic"), val("NumEmptyPr")],
    "err": [val("NumCSChronicStd"), val("NumEmptyStd")],
    "n": [val("NumCSChronicN"), val("NumEmptyN")],
    "source_macros": ["NumCSChronic", "NumEmptyPr"],
}
fig, ax = plt.subplots(figsize=(3.6, 2.4))
xs = np.arange(2)
ax.bar(xs, f1["values"], yerr=f1["err"], capsize=3,
       color=["#2980b9", "#95a5a6"], alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(f1["labels"], fontsize=7)
ax.set_ylabel("PR-AUC (mean $\\pm$ std)"); ax.set_title("Same-step edge boundary", pad=12)
ax.set_ylim(0, max(np.array(f1["values"]) + np.array(f1["err"])) * 1.25)
for i, x in enumerate(xs):
    ax.text(x, f1["values"][i] + f1["err"][i] + 0.015,
            f"{f1['values'][i]:.3f} (n={int(f1['n'][i])})", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "same_step_boundary.png"); plt.close(fig)
record["same_step_boundary"] = f1

# ---- F2: protocol configurations, horizontal labels ----
f2 = {
    "labels": ["Random", "Chrono", "Unknown-neg"],
    "values": [val("NumLadOne"), val("NumLadTwo"), val("NumLadThree")],
    "source_macros": ["NumLadOne", "NumLadTwo", "NumLadThree"],
}
fig, ax = plt.subplots(figsize=(4.6, 2.2))
xs = np.arange(3)
ax.bar(xs, f2["values"], color="#2980b9", alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(f2["labels"], fontsize=7)  # short labels, no rotation/crop
ax.set_ylabel("PR-AUC"); ax.set_title("Protocol configurations (CausalSAGE, C1)", pad=12)
ax.set_ylim(0, max(f2["values"]) * 1.22)
for i, x in enumerate(xs):
    ax.text(x, f2["values"][i] + 0.012, f"{f2['values'][i]:.3f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "contract_ladder.png"); plt.close(fig)
record["contract_ladder"] = f2

# ---- F3: measured ECE/Brier, auto ylim ----
f3 = {
    "models": ["Random Forest", "LightGBM", "CausalSAGE"],
    "ece": [val("NumRFECE"), val("NumLGBMECE"), val("NumCSEce")],
    "brier": [val("NumRFBrier"), val("NumLGBMBrier"), val("NumCSBrier")],
}
fig, ax = plt.subplots(figsize=(3.6, 2.4))
xs = np.arange(3); w = 0.35
ax.bar(xs - w/2, f3["ece"], w, label="ECE (15 bins)", color="#2980b9")
ax.bar(xs + w/2, f3["brier"], w, label="Brier", color="#e67e22")
ax.set_xticks(xs); ax.set_xticklabels(f3["models"], fontsize=7)
ax.set_ylabel("Score (lower is better)"); ax.set_title("Calibration summary", pad=12)
ax.legend(fontsize=6)
ax.set_ylim(0, max(max(f3["ece"]), max(f3["brier"])) * 1.25)  # no clipping
fig.tight_layout(); fig.savefig(OUT / "calibration_summary.png"); plt.close(fig)
record["calibration_summary"] = f3

(CHECK).write_text(json.dumps(record, indent=2), encoding="utf-8")
print("figures from numbers.tex:")
for k, v in record.items():
    print(" ", k, v.get("values", v.get("ece")))
print("saved", CHECK)
