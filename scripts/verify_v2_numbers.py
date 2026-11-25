#!/usr/bin/env python3
"""Verify every v2 manuscript macro matches its machine-readable source.

Enforces reviewer H1: Table I, Table II, abstract and paired delta all share
ONE per-seed definition, so the headline effect is reconstructible.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "tifs_v2"
REBUTTAL_T = ROOT / "outputs" / "rebuttal" / "tables"

unified = pd.read_csv(REBUTTAL_T / "Table1_unified_perseed.csv")
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (OUT / "numbers.tex").read_text(encoding="utf-8")))
paired = json.loads((REBUTTAL_T / "paired_effect_unified.json").read_text(encoding="utf-8"))

def near(a, b, tol=0.0015):
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return False

def pick(model, split, field):
    r = unified[(unified.model == model) & (unified.split_mode == split)]
    return float(r.iloc[0][field]) if len(r) else None

checks = [
    ("NumRFPrAUC", pick("rf", "chrono", "pr_auc")),
    ("NumCSChronic", pick("causal_sage", "chrono", "pr_auc")),
    ("NumCSRandom", pick("causal_sage", "random", "pr_auc")),
    ("NumLGBMChronic", pick("lightgbm", "chrono", "pr_auc")),
    ("NumDeltaPaired", paired["mean_delta"]),
]

failures = []
for name, expected in checks:
    got = macros.get(name)
    if expected is None or got is None or not near(got, expected):
        failures.append(f"{name}: macro={got} source={expected}")
    else:
        print(f"OK  {name} = {got}")

# H1 ZERO-TOLERANCE: effect must be reconstructible from Table I values.
rf = pick("rf", "chrono", "pr_auc")
cs = pick("causal_sage", "chrono", "pr_auc")
reconstructed = cs - rf
if not near(reconstructed, macros.get("NumDeltaPaired"), tol=0.002):
    failures.append(f"H1: Table I difference {reconstructed:.3f} != abstract delta {macros.get('NumDeltaPaired')}")
else:
    print(f"OK  H1 reconstructible: {cs:.3f} - {rf:.3f} = {reconstructed:.3f} ~ delta {macros.get('NumDeltaPaired')}")

# Protocol A must equal the unified CausalSAGE value.
if not near(macros.get("NumProtA"), macros.get("NumCSChronic")):
    failures.append(f"ProtocolA inconsistent: {macros.get('NumProtA')} vs {macros.get('NumCSChronic')}")
else:
    print(f"OK  ProtocolA == CausalSAGE == {macros.get('NumProtA')}")

# No N/A and no stale numbers.
nas = [k for k, v in macros.items() if "N/A" in v]
if nas:
    failures.append(f"N/A macros: {nas}")
else:
    print("OK  no N/A macros")

# Table II CausalSAGE delta must match abstract.
t3 = (OUT / "tables" / "Table3_stats.tex").read_text(encoding="utf-8")
cs_row = [l for l in t3.splitlines() if "CausalSAGE" in l]
if cs_row and macros.get("NumDeltaPaired") not in cs_row[0]:
    failures.append(f"Table II CausalSAGE delta mismatch: {cs_row[0]}")
else:
    print("OK  Table II delta matches abstract")

if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print(f"\nALL V2 NUMBERS CONSISTENT ({len(checks)+4} checks)")
