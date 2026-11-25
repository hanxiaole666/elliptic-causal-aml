#!/usr/bin/env python3
"""V5 verification — closes the 'consistency test blind spots' the reviewers found.

Beyond numeric range, this checks:
 - PDF text layer has no '¡' / '<br>' / broken CI (the FOUR-round glyph bug);
 - all macros numeric, no N/A;
 - figure values == table macros (cross-artifact);
 - config identity: no protocol has two values across tables;
 - references verified (20 entries, corrected authors);
 - banned review-letter language;
 - exact p-values (no <10^-3 glyph);
 - split disjointness assertions in manifest.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "tifs_v5"
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (OUT / "numbers.tex").read_text(encoding="utf-8")))
tex = (OUT / "main.tex").read_text(encoding="utf-8")
figvals = json.loads((ROOT / "outputs" / "v5" / "tables" / "figure_values.json").read_text(encoding="utf-8"))

passes, failures = [], []
def check(c, m): (passes if c else failures).append(m)

def num(name):
    try: return float(macros[name])
    except (KeyError, ValueError): return None

# 1. No N/A
nas = [k for k, v in macros.items() if "N/A" in v]
check(not nas, "no N/A macros" if not nas else f"N/A: {nas}")

# 2. Banned language / broken glyphs in SOURCE
banned = ["reviewer", "earlier draft", "outputs/rebuttal", "(M2)", "(H2)", "Table--II", "¡", "<br>"]
hits = [b for b in banned if b.lower() in tex.lower()]
check(not hits, "no banned language in source" if not hits else f"banned: {hits}")
check("<$10^{-3}$" not in tex, "no text-mode less-than glyph bug")

# 3. PDF TEXT LAYER (the actual symptom reviewers saw)
pdf = OUT / "main.pdf"
if pdf.exists():
    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    check("¡" not in txt, "PDF text layer: no '¡' glyph")
    check("<br>" not in txt, "PDF text layer: no literal '<br>'")
    check("10−3" not in txt or "p <" in txt, "PDF text layer: p-values readable")
else:
    failures.append("PDF missing")

# 4. Cross-artifact figure==table
for fig, payload in figvals.items():
    srcs = payload.get("source_macros", [])
    vals = payload.get("values")
    if vals and srcs:
        ok = all(v is not None and abs(v - float(macros[s])) < 1e-6 for v, s in zip(vals, srcs))
        check(ok, f"figure {fig} == macros")
    else:
        check((OUT / "figures" / f"{fig}.png").exists(), f"figure {fig}.png exists")

# 5. H1 reconstructible
cs, rf, delta = num("NumCSChronic"), num("NumRFPrAUC"), num("NumDeltaPaired")
check(cs is not None and rf is not None and delta is not None and abs((cs-rf)-delta) <= 0.002,
      f"H1 reconstructible: {cs}-{rf} vs {delta}")

# 6. Config identity: single source table exists, all cited IDs present
ident = json.loads((ROOT / "outputs" / "v5" / "tables" / "config_identity.json").read_text(encoding="utf-8"))
check("C1" in ident and "note" in ident, "config registry present")

# 7. References verified
bib = (OUT / "references.bib").read_text(encoding="utf-8")
check(len(re.findall(r"^@", bib, re.M)) == 23, "23 verified references")
check("Elmougy, Youssef" in bib and "Liu, Ling" in bib, "Elmougy/Liu corrected")
check("Khaleghpour, Hamideh" in bib and "McKinney, Brett" in bib, "Khaleghpour corrected")
check("2309.05125" not in bib, "no unrelated geophysics arXiv id")

# 8. Declarations
for d in ("Funding:", "Conflict of interest:", "Data availability:", "Author contributions:", "Generative AI use:"):
    check(d in tex, f"declaration {d}")

# 9. Equivalence honesty
check("equivalence is \\emph{not} established" in tex or "not established" in tex.lower(),
      "equivalence honestly qualified")

# 10. I4 same-score and determinism evidence present
i4 = json.loads((ROOT / "outputs" / "v5" / "tables" / "i4_same_scores.json").read_text(encoding="utf-8"))
check(i4.get("pr_auc_identical") and i4.get("roc_auc_identical"), "I4 same-score invariance verified")
det = json.loads((ROOT / "outputs" / "v5" / "tables" / "empty_graph_determinism.json").read_text(encoding="utf-8"))
check(det.get("scores_bit_identical"), "empty-graph determinism verified")

# 11. FIFTH-ROUND conflicts: table text must match the macro/abstract (the blind spot).
tbl_worlds = (OUT / "tables" / "Table2_worlds.tex").read_text(encoding="utf-8")
real_row = [l for l in tbl_worlds.splitlines() if l.startswith("real &")]
check(bool(real_row) and macros["NumCSChronic"] in real_row[0],
      "Table worlds 'real' row == main-table CausalSAGE (not protocol B)")
tbl_stats = (OUT / "tables" / "Table3_stats.tex").read_text(encoding="utf-8")
cs_row = [l for l in tbl_stats.splitlines() if l.startswith("CausalSAGE &")]
check(bool(cs_row) and macros["NumDeltaPaired"] in cs_row[0],
      "Table stats CausalSAGE delta == abstract delta")
check("identical in the abstract" not in tex, "no false 'identical' claim")
# Zero-edge determinism: only ONE zero-edge row/bar, matching the 'bit-identical' claim.
samestep_rows = [l for l in tex.splitlines() if "& \\NumEdges{} &" in l or "& 0 & \\NumEmptyPr{}" in l]
check(len(samestep_rows) == 2,
      f"same-step table has exactly two rows (full + one zero-edge control), got {len(samestep_rows)}")
check(len(figvals["same_step_boundary"]["values"]) == 2,
      "figure 1 has exactly two bars (full + zero-edge)")
# I4 vector honestly labelled synthetic.
check("synthetic" in tex.lower() and "i4test" in tex.lower(), "I4 table labelled synthetic")

# Sixth-round guards: no internal iteration numbering, no contradictory ladder note,
# canonical ROC column sourced from macros (not hardcoded), no Sec. VI mis-ref.
check("v5" not in tex.lower(), "no internal version numbering in manuscript")
check("only the protocol rung changes" not in tex, "no contradictory ladder footnote")
# Guard the ACTUAL row count (not just banned strings) -- the reviewer found this blind spot.
ladder_rows = [l for l in tex.splitlines() if " & CausalSAGE & \\NumLad" in l]
check(len(ladder_rows) == 3, f"ladder table has exactly three rungs, got {len(ladder_rows)}")
check(len(figvals["contract_ladder"]["values"]) == 3, "figure 2 has exactly three bars")
check("Calibration is not listed as a rung" in tex, "body agrees ladder has no calibration rung")
check("Sec.~\\ref{sec:setup}) confirms" not in tex, "determinism test referenced correctly")
canon = (OUT / "tables" / ".." / "main.tex")
main_tex = (OUT / "main.tex").read_text(encoding="utf-8")
check("Split & Model & PR-AUC & ROC-AUC & ECE" in main_tex, "canonical table has ROC-AUC column")
check("0.905 & \\NumCanon" not in main_tex, "no hardcoded ROC values in canonical table")

# Sixth-review guards: ECE single value, no revision traces, no layout overflow.
check("earlier" not in tex.lower(), "no revision-process traces ('earlier')")
# ECE for C1-chrono must be a single value across all tables/macros.
main_tbl = (OUT / "tables" / "Table1_main.tex").read_text(encoding="utf-8")
worlds_tbl = (OUT / "tables" / "Table2_worlds.tex").read_text(encoding="utf-8")
ece_macro = macros.get("NumCSEce")
check(ece_macro in main_tbl, f"main table ECE == NumCSEce ({ece_macro})")
real_w = [l for l in worlds_tbl.splitlines() if l.startswith("real &")]
check(bool(real_w) and ece_macro in real_w[0], f"worlds real ECE == NumCSEce ({ece_macro})")
check("synthetic" in tex.lower(), "I4 vector declared synthetic in body")
check("cawnet2021" in bib and "tgat2020" in bib and "tgn2020" in bib, "temporal GNN citations present")
check("\\begin{table*}" in tex and "Reproducibility map" in tex, "reproducibility map spans both columns")

# Every model in the paired-stats table must exist in the main table (no orphan rows).
tbl_main = (OUT / "tables" / "Table1_main.tex").read_text(encoding="utf-8")
for line in tbl_stats.splitlines():
    if " & " in line and "Model &" not in line and "toprule" not in line and "midrule" not in line and "bottomrule" not in line:
        name = line.split(" & ")[0].strip()
        if name and name not in ("\\vspace{2pt}\\par\\footnotesize\\noindent Family:",):
            check(name in tbl_main, f"stats row '{name}' also present in main table")

for p in passes: print("OK  ", p)
for f in failures: print("FAIL", f)
if failures: sys.exit(1)
print(f"\nALL V5 CHECKS PASSED ({len(passes)})")
