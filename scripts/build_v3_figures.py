#!/usr/bin/env python3
"""Generate the three figures required by the second-round review:
  F1: contract-ladder PR-AUC by rung (with seed error bars)
  F2: same-step boundary comparison (full vs strict/empty)
  F3: reliability diagram (calibration quality)
All data come from machine-readable results; figures carry sample sizes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "tifs_v3" / "figures_src"
FULL = ROOT / "outputs" / "full" / "results"
V3 = ROOT / "outputs" / "v3" / "results"
REB = ROOT / "outputs" / "rebuttal" / "results"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def load(dirs):
    rows = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") == "done":
                rows.append(r)
    return rows


rows = load([FULL, V3, REB])


def pr(pred):
    v = [r["metrics"]["calibrated"]["pr_auc"] for r in rows if pred(r.get("config", {}), r)]
    return np.array(v) if v else np.array([np.nan])


# F1: contract ladder
labels = ["Random\nknown-only", "Chrono\nknown-only", "Chrono\nunknown-neg", "Chrono\ntest-cal"]
preds = [
    lambda c, r: c.get("split_mode") == "random" and c.get("label_protocol", "A") == "A" and r["model"] == "causal_sage",
    lambda c, r: c.get("split_mode") == "chrono" and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none" and c.get("calibration") != "none" and r["model"] == "causal_sage",
    lambda c, r: c.get("label_protocol") == "B" and r["model"] == "causal_sage",
    lambda c, r: c.get("calibration") == "none" and c.get("split_mode") == "chrono" and r["model"] == "causal_sage",
]
vals = [pr(p) for p in preds]
means = [np.nanmean(v) if len(v) and np.isfinite(v).any() else np.nan for v in vals]
stds = [np.nanstd(v, ddof=1) if len(v) > 1 else 0.0 for v in vals]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
xs = np.arange(len(labels))
ax.bar(xs, means, yerr=stds, capsize=3, color=["#c0392b", "#2980b9", "#7f8c8d", "#7f8c8d"], alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("PR-AUC"); ax.set_title("Contract ladder (CausalSAGE)")
ax.set_ylim(0, max([m for m in means if np.isfinite(m)] + [0.6]) * 1.2)
for i, (m, v) in enumerate(zip(means, vals)):
    if np.isfinite(m):
        ax.text(i, m + (stds[i] or 0) + 0.02, f"{m:.3f}\nn={len(v)}", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "contract_ladder.png"); plt.close(fig)

# F2: same-step boundary
full_pr = pr(lambda c, r: r["model"] == "causal_sage" and c.get("split_mode") == "chrono" and c.get("strict_time", False) is False and c.get("attack", "none") == "none" and c.get("label_protocol", "A") == "A" and c.get("hidden", 32) == 32)
strict_pr = pr(lambda c, r: r["model"] == "causal_sage" and c.get("strict_time", False))
empty_pr = pr(lambda c, r: r["model"] == "causal_sage" and c.get("edge_world") == "empty")
names = ["Non-future-step\n(all same-step kept)", "Strict causal\n(no same-step)", "Empty-world\ncontrol"]
vs = [full_pr, strict_pr, empty_pr]
ms = [np.nanmean(v) if np.isfinite(v).any() else np.nan for v in vs]
ss = [np.nanstd(v, ddof=1) if len(v) > 1 else 0.0 for v in vs]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
xs = np.arange(3)
ax.bar(xs, ms, yerr=ss, capsize=3, color=["#2980b9", "#c0392b", "#95a5a6"], alpha=0.85)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel("PR-AUC"); ax.set_title("Same-step edge boundary")
for i, (m, v) in enumerate(zip(ms, vs)):
    if np.isfinite(m):
        ax.text(i, m + (ss[i] or 0) + 0.02, f"{m:.3f}\nn={len(v)}", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "same_step_boundary.png"); plt.close(fig)

# F3: calibration summary using MEASURED ECE and Brier (no invented bin curves).
def ece_brier(pred):
    es = [r["metrics"]["calibrated"]["ece"] for r in rows if pred(r.get("config", {}), r)]
    bs = [r["metrics"]["calibrated"]["brier"] for r in rows if pred(r.get("config", {}), r)]
    es = [v for v in es if np.isfinite(v)]; bs = [v for v in bs if np.isfinite(v)]
    return (float(np.mean(es)), float(np.std(es, ddof=1)) if len(es) > 1 else 0.0,
            float(np.mean(bs)), len(es))

models = [
    ("Random Forest", lambda c, r: r["model"] == "rf" and c.get("split_mode") == "chrono"),
    ("LightGBM", lambda c, r: r["model"] == "lightgbm" and c.get("split_mode") == "chrono"),
    ("CausalSAGE", lambda c, r: r["model"] == "causal_sage" and c.get("split_mode") == "chrono" and c.get("attack", "none") == "none"),
]
names, ece_m, ece_s, brier_m = [], [], [], []
for nm, p in models:
    e, es, b, n = ece_brier(p)
    names.append(nm); ece_m.append(e); ece_s.append(es); brier_m.append(b)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
xs = np.arange(len(names)); w = 0.35
ax.bar(xs - w/2, ece_m, w, yerr=ece_s, capsize=2, label="ECE (15 bins)", color="#2980b9")
ax.bar(xs + w/2, brier_m, w, label="Brier", color="#e67e22")
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel("Score (lower is better)"); ax.set_title("Calibration summary")
ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "calibration_summary.png"); plt.close(fig)

print("figures written to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name)
