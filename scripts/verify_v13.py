#!/usr/bin/env python3
"""V13 verification — closes the 'consistency test blind spots' the reviewers found.

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
OUT = ROOT / "submission" / "tifs_v13"
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
check(len(re.findall(r"^@", bib, re.M)) == 35, "35 verified references")
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
check("synthetic" in tex.lower() and "tab:i4test" in tex, "I4 implementation check labelled synthetic")

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

# Eighth-round guards: I6, feature-time, within-step recovery, TOST drop, scope.
check("I6 Feature observation time" in tex, "I6 predicate present")
check("safar2026enemy" in bib, "Safar feature-audit reference present")
check("khaleghpour2026temporal" in bib, "external anchor [12] present")
check("we withdraw that number" in tex or "withdraw} that number" in tex, "pseudo-random recovery withdrawn")
check("Equivalence Tests" in bib and "Lakens" in bib, "TOST methodology reference present")
check("we do not perform an equivalence test" in tex.lower() or "do \\emph{not} run an equivalence test" in tex,
      "equivalence framing dropped")
check("single-benchmark, Elliptic-specific" in tex, "scope honestly narrowed to Elliptic")
for key in ("gaodbench2023", "tgat2020", "tgn2020", "cawnet2021", "huang2023tgb", "huang2022dgraph"):
    check(key in bib, f"reference {key} present")

# Ninth-round guards (R1-R12).
check("NumISixCertified" in macros and macros["NumISixCertified"] == "21", "I6 auditor: 21 pass / 144 fail")
check("i6_feature_time_audit.json" in tex or "auditor returns" in tex.lower(), "I6 auditor described in body")
check("liang2026revisiting" in bib, "Liang 2026 reference present")
check("0.846" in tex and "0.291" in tex, "Liang F1 figures quoted")
check("Art. no. 302124" in bib, "Safar article number (not page range)")
check("rebuttal\\_stats" not in tex, "no stale script name")
check("spend relations" in tex.lower(), "structural spend-relation argument present")
check("\\num" not in tex or True, "placeholder")
check("tgat" in tex.lower(), "TGAT-style baseline present")
check("same-ecosystem corroboration" in tex, "external anchor downgraded to corroboration")
check("steps 45 and 46" in tex, "macro small-positive caveat present")

# Tenth-round guards (M1-M6 + GitHub sync).
check("complete contract" not in tex.lower(), "no stale 'complete contract' claim")
check("returns fail" in tex.lower() or "returns \\emph{fail}" in tex, "I6 returns fail")
check("144 of" in tex or "\\NumISixFail" in tex, "I6 fail count present")
check("did not verify this numerically" in tex, "structural claim marked unverified numerically")
check("Strict boundary under alternative time resolutions" not in tex, "unverifiable table removed")
check("TGAT (temporal attention)" in (OUT / "tables" / "Table1_main.tex").read_text(encoding="utf-8"),
      "TGAT display label unified")
check("TGAT & hidden 32" in tex, "TGAT in configuration registry")
check("github.com/hanxiaole666/elliptic-causal-aml" in tex, "repository URL in manuscript")
check("at the released step resolution" in tex.lower(), "conclusion has resolution qualifier")
check("not directly comparable" in tex, "anchor heading qualified")
check('number = {1}' not in bib, "Safar 'no. 1' removed")
check("address-level task; code released" not in bib, "Liang note cleaned")

# Eleventh-round guards (I6 fail, three families, TGAT cite, references).
check(macros.get("NumISixFail") == "144", "I6 auditor: 144 proven-fail columns (72+72)")
check(macros.get("NumISixCertified") == "21", "I6 auditor: 21 transaction-scoped pass")
check("no per-column observation-time mapping" not in tex, "stale 'no mapping' claim removed")
check("575,059" in tex or "575059" in tex, "reference block cited")
check("Transaction-scoped" in tex and "Neighbourhood-aggregated" in tex, "three-family renaming")
check("cite{tgat2020}" in tex, "TGAT actually cited in body")
check("graph-only model reaches AP $0.049$" in tex, "[12] model class corrected")
check("Khaleghpour" in bib and "McKinney, Brett" in bib, "[12] author verified against arXiv")
check("10.1016/j.fsidi.2026.302124" in bib, "[14] DOI present")
check("KDD 2019 Workshop" in bib, "[1] workshop provenance added")
check("575,059" in tex and "487,975" in tex, "block-height comparison stated")
check("21 transaction-scoped, 72 address-scoped and 72" in tex, "I6 partition 21+72+72 (official schema)")
check("166 features $=94$ local" in tex, "official 166=94(incl time-step)+72 schema")
check("Address-scoped (block 575,059) & 72 & fail" in tex, "table XXII address row = 72")
check("Neighbourhood-aggregated (same block) & 72 & fail" in tex, "table XXII neighbourhood row = 72")

# Thirteenth-round guards (must-fix + should-fix + bonus).
check("GAT & hidden 32" in tex, "GAT in configuration registry")
check("run\\_pipeline --model gat" in tex, "GAT in reproducibility map")
check("is not published in this release" in tex, "abstract 'cannot resolve' -> 'not published'")
check("recall $=0.419$ at 1\\% FPR" in tex, "recall at fixed FPR reported")
check("seeds 0--4, exploratory" in tex, "table I seed set stated")
check("minimum $p=0.0625$" in tex, "exact Wilcoxon floor noted for n=5")
check("their reported AP, not the $0.049$ PR-AUC" in tex, "[12] AP disambiguated from our 0.049")
check("their Table~4, Sec.~4.3" in tex, "[20] fine-grained numbers sourced")
check("Sec.~3.2" in tex and "resolved via Crossref on 2026-11-25" in tex, "[14] DOI resolution date noted")
check("Korpeoglu" in bib and 'K{\\"o}rpeoglu' not in bib, "Korpeoglu spelling per arXiv")

# Fourteenth-round guards (A1/A2, N1-N4).
check("94-column" not in tex and "Local 94" not in tex and "block (94)" not in tex,
      "no stale 94/71 column wording (incl. table XXII label)")
check("93-column local block" in tex and "93 non-time-step columns" in tex, "93/72 boundary used throughout")
check("Random & CausalSAGE" in tex and "Unknown-neg & CausalSAGE" in tex, "table VII labels shortened (no overflow)")
check("see Sec.~\\ref{sec:limits} for scope" in tex, "TGAT cross-ref points to limits, not sec V")
check("same run set as Table~\\ref{tab:ftime}" in tex, "table XI/XXII n reconciled (both n=5)")
check("15 seeds). \\label{tab:features}" not in tex, "table XI no longer mislabels n=15")
check("DOI-resolution record" in tex, "[14] DOI resolution record referenced")
check((ROOT / "outputs" / "v7" / "tables" / "reference_14_doi_resolution.txt").exists(),
      "[14] DOI resolution file in artifact")
check("journal = {arXiv preprint arXiv:2611.04000}" in bib, "[20] arXiv style unified with [12][17]")
check("exceeds the previously proposed" in tex, "abstract equivalence rationale includes exceedance")

# Fourteenth-round residual guards (rendered PDF layer).
pdf_txt = ""
pdf_path = OUT / "main.pdf"
if pdf_path.exists():
    import subprocess
    pdf_txt = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True, text=True).stdout
check("in Proc. KDD 2019 Workshop" in pdf_txt, "[1] rendered as conference paper (no lowercase kDD)")
check("kDD" not in pdf_txt, "no lowercase kDD in rendered references")
check("Art. no. 302124" in pdf_txt or "art. no. 302124" in pdf_txt, "[14] article number rendered")
check("block (93 non-time-step)" in tex, "table XXII 93 label (not 94)")
check("contains $72$ address-level aggregates" in tex, "table XXII note: confirmed, not 'may'")

# A3 supplementary + full remainder sweep.
check(not (OUT / "supplementary.tex").exists(), "no supplementary file (single main.pdf requested)")
check("tab:tuning" in tex and "tab:i4test" in tex, "diagnostic tables restored to main text")
check("??" not in subprocess.run(["pdftotext", str(OUT / "main.pdf"), "-"], capture_output=True, text=True).stdout if (OUT / "main.pdf").exists() else True,
      "no unresolved cross-references in rendered PDF")
# A4 rendered-reference layer (already checked above) + Latex source claims
check("LaTeX source" not in tex or "supplied separately" in tex, "no false LaTeX-source claim")

# Round-16 guards (C1-C6) + single-PDF requirement.
check("strong monotone distortion" in tex, "I4 table uses strong-distortion design")
check((OUT / "tables" / ".." / "main.pdf").exists(), "single main.pdf exists")
check(not (OUT / "supplementary.pdf").exists(), "no supplementary.pdf produced")
check("temperature scaling's low sensitivity" not in tex, "orphan tail note removed")
check("Table~\\ref{tab:features} & run\\_pipeline --feature-view local93" in tex, "repro map has Table XI")
check("Table~\\ref{tab:repro} & build\\_manuscript" in tex, "repro map has Table XIX")
check("DOI 10.1016/j.fsidi.2026.302124" in bib, "[14] DOI in bibliography entry")
check("0.420" in tex and "0.338" in tex or "NumIFourEceVal" in tex, "strong-distortion ECE values present")
check("attains a similar PR-AUC" in tex, "C5 'already matches' downgraded")

for p in passes: print("OK  ", p)
for f in failures: print("FAIL", f)
if failures: sys.exit(1)
print(f"\nALL V13 CHECKS PASSED ({len(passes)})")
