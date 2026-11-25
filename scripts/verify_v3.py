#!/usr/bin/env python3
"""V3 final verification: numbers, N/A macros, no review-letter language, figures."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "tifs_v3"
REB = ROOT / "outputs" / "rebuttal" / "tables"

failures, passes = [], []

def check(cond, msg):
    (passes if cond else failures).append(msg)

macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (OUT / "numbers.tex").read_text(encoding="utf-8")))
tex = (OUT / "main.tex").read_text(encoding="utf-8")
unified = pd.read_csv(REB / "Table1_unified_perseed.csv")

def near(a, b, tol=0.0015):
    try: return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError): return False

def pick(model, split, field):
    r = unified[(unified.model == model) & (unified.split_mode == split)]
    return float(r.iloc[0][field]) if len(r) else None

# Core numbers
for name, exp in [
    ("NumRFPrAUC", pick("rf", "chrono", "pr_auc")),
    ("NumCSChronic", pick("causal_sage", "chrono", "pr_auc")),
    ("NumCSRandom", pick("causal_sage", "random", "pr_auc")),
]:
    check(macros.get(name) is not None and near(macros[name], exp), f"{name}={macros.get(name)} vs {exp}")

# H1 reconstructibility
cs, rf = pick("causal_sage", "chrono", "pr_auc"), pick("rf", "chrono", "pr_auc")
check(near(cs - rf, macros.get("NumDeltaPaired"), 0.002), f"H1 reconstructible: {cs:.3f}-{rf:.3f} vs delta {macros.get('NumDeltaPaired')}")

# No N/A macros
nas = [k for k, v in macros.items() if "N/A" in v]
check(not nas, f"N/A macros: {nas}" if nas else "no N/A macros")

# No review-letter language (desk-reject risk)
banned = ["reviewer", "earlier draft", "earlier drafts", "outputs/rebuttal", "(M2)", "(M4)", "(H2)", "(H3)",
          "(M1)", "(M3)", "(M7)", "We do not hide", "honest positioning", "previously described", "now reported"]
hits = [b for b in banned if b.lower() in tex.lower()]
check(not hits, f"review-letter language: {hits}" if hits else "no review-letter language")

# Broken glyphs
check("¡" not in tex, "no broken glyphs")
check("g<br>" not in tex, "no HTML break residue")

# Figures present and referenced
for fig in ["same_step_boundary", "contract_ladder", "calibration_summary"]:
    check((OUT / "figures" / f"{fig}.png").exists(), f"figure {fig}.png exists")

# Same-step boundary: strict macro must exist (graph collapses -> feature-only)
check(macros.get("NumStrictPr") is not None and "N/A" not in str(macros.get("NumStrictPr")), f"NumStrictPr={macros.get('NumStrictPr')}")

for p in passes: print("OK ", p)
for f in failures: print("FAIL", f)
if failures:
    sys.exit(1)
print(f"\nALL V3 CHECKS PASSED ({len(passes)})")
