#!/usr/bin/env python3
"""V4 verification including the third-round CROSS-TABLE/FIGURE consistency test.

Checks: (1) all macros numeric, no N/A; (2) no review-letter language / broken glyphs;
(3) H1 reconstructibility; (4) figure labels == table macros (figure_values.json);
(5) bibliography count; (6) statutory declarations present; (7) equivalence honesty.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "tifs_v4"
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (OUT / "numbers.tex").read_text(encoding="utf-8")))
tex = (OUT / "main.tex").read_text(encoding="utf-8")
figvals = json.loads((ROOT / "outputs" / "v4" / "tables" / "figure_values.json").read_text(encoding="utf-8"))

passes, failures = [], []
def check(c, m): (passes if c else failures).append(m)

def num(name):
    try: return float(macros[name])
    except (KeyError, ValueError): return None

# 1. No N/A / all numeric
nas = [k for k, v in macros.items() if "N/A" in v]
check(not nas, f"no N/A macros" if not nas else f"N/A: {nas}")

# 2. Review-letter language / broken glyphs
banned = ["reviewer", "earlier draft", "outputs/rebuttal", "(M2)", "(M4)", "(H2)", "(H3)",
          "Table--II", "11/14 compliant", "g<br>", "¡", "within noise"]
hits = [b for b in banned if b.lower() in tex.lower()]
check(not hits, "no review-letter/glyph residue" if not hits else f"banned: {hits}")

# 3. H1 reconstructibility
cs, rf, delta = num("NumCSChronic"), num("NumRFPrAUC"), num("NumDeltaPaired")
check(cs is not None and rf is not None and delta is not None and abs((cs-rf)-delta) <= 0.002,
      f"H1 reconstructible {cs}-{rf} vs {delta}")

# 4. CROSS-TABLE/FIGURE: every plotted value must equal the corresponding macro
for fig, payload in figvals.items():
    srcs = payload.get("source_macros", [])
    vals = payload.get("values")
    if vals and srcs:
        ok = all(v is not None and abs(v - float(macros[s])) < 1e-6 for v, s in zip(vals, srcs))
        check(ok, f"figure {fig} values == macros {srcs}")
    else:
        check((OUT / "figures" / f"{fig}.png").exists(), f"figure {fig}.png exists")

# 5. Bibliography
bib = (OUT / "references.bib").read_text(encoding="utf-8")
check(len(re.findall(r"^@", bib, re.M)) == 20, "20 verified references")
check("Elmougy, Youssef" in bib and "Liu, Ling" in bib, "Elmougy/Liu corrected")
check("Khaleghpour, Hamideh" in bib and "McKinney, Brett" in bib and "2603.06632" in bib, "Khaleghpour corrected")
check("2309.05125" not in bib, "no unrelated geophysics arXiv id")

# 6. Declarations
for decl in ("Funding:", "Conflict of interest:", "Data availability:", "Author contributions:", "Generative AI use:"):
    check(decl in tex, f"declaration {decl}")

# 7. Equivalence honesty: must NOT claim equivalence
check("equivalence is \\emph{not} established" in tex or "equivalence is not established" in tex.lower(),
      "equivalence honestly qualified")

for p in passes: print("OK  ", p)
for f in failures: print("FAIL", f)
if failures: sys.exit(1)
print(f"\nALL V4 CHECKS PASSED ({len(passes)})")
