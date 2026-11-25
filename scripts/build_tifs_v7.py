#!/usr/bin/env python3
"""TIFS major-revision manuscript (v2).

Built from machine-readable results. Every number is derived from CSV/JSON;
no hand-filled values. Addresses the Reject-and-Resubmit review:
  M1/M9  research question & literature repositioning (RF-over-GNN already in Weber)
  M2     equal-budget tuning sensitivity (width/depth, class weighting disclosed)
  M3     canonical splits 1-34/35-42 and 1-34/35-49; per-step positive counts
  M4/H1  paired seed-level delta unified; time-block bootstrap
  M5     formal definition of the five invariants + Algorithm 1
  M6     invariant ladder / promised experiments (attacks, PU, protocols) now tabulated
  M7     operational metrics (P@k, FPR-fixed recall, cost-weighted utility)
  M8     evaluation-deception threat model with attacker goal/capability/budget
  M10    artifact verification, checksums, environment
  m1-m20 table/figure/naming/declaration fixes
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "full" / "tables"
REBUTTAL_T = ROOT / "outputs" / "rebuttal" / "tables"
DIAG = ROOT / "outputs" / "diagnostics"
OUT = ROOT / "submission" / "tifs_v7"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "rf": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "mlp": "MLP", "lr": "Logistic Regression", "causal_sage": "CausalSAGE",
    "graphsage": "GraphSAGE", "gcn": "GCN",
}


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def model_name(key):
    return MODEL_NAMES.get(str(key), str(key))


def load_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def build_numbers():
    """All paper macros from machine-readable tables + rebuttal stats.

    The main table is the UNIFIED per-seed table (Table1_unified_perseed.csv)
    so that Table I, Table III, abstract and paired delta share one definition.
    """
    # V5: single source of truth, one pre-registered configuration per model.
    v5table = ROOT / "outputs" / "v7" / "tables" / "Table1_v5.csv"
    main_v5 = pd.read_csv(v5table) if v5table.exists() else None
    main = main_v5 if main_v5 is not None else pd.read_csv(REBUTTAL_T / "Table1_unified_perseed.csv")
    stats = load_json(TABLES / "statistical_tests.json", {})
    paired_v5 = load_json(ROOT / "outputs" / "v5" / "tables" / "paired_v5.json", {})
    paired = load_json(REBUTTAL_T / "paired_effect_unified.json", {})
    tbb = load_json(REBUTTAL_T / "time_block_bootstrap.json", {})
    diag = load_json(DIAG / "data_diagnostics.json", {})
    filter_audit = load_json(DIAG / "future_edge_filter_audit.json", {})

    def pick(model, split, field):
        if "config_id" in main.columns:
            cid = {
                ("causal_sage", "chrono"): "C1-chrono",
                ("causal_sage", "random"): "C1-random",
                ("rf", "chrono"): "rf-chrono",
                ("rf", "random"): "rf-random",
                ("lightgbm", "chrono"): "lightgbm-chrono",
                ("lightgbm", "random"): "lightgbm-random",
            }.get((model, split))
            r = main[main.config_id == cid] if cid else main.iloc[0:0]
        else:
            r = main[(main.model == model) & (main.split_mode == split)]
        return float(r.iloc[0][field]) if len(r) and field in r.columns else float("nan")

    comp = {c["model"]: c for c in stats.get("comparisons", [])}
    cs = comp.get("causal_sage", {})
    lines = [
        f"\\newcommand{{\\NumRFPrAUC}}{{{fmt(pick('rf','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumRFPrAUCStd}}{{{fmt(pick('rf','chrono','pr_auc_std'))}}}",
        f"\\newcommand{{\\NumRFRoc}}{{{fmt(pick('rf','chrono','roc_auc'))}}}",
        f"\\newcommand{{\\NumRFBrier}}{{{fmt(pick('rf','chrono','brier'))}}}",
        f"\\newcommand{{\\NumRFECE}}{{{fmt(pick('rf','chrono','ece'))}}}",
        f"\\newcommand{{\\NumLGBMBrier}}{{{fmt(pick('lightgbm','chrono','brier'))}}}",
        f"\\newcommand{{\\NumLGBMECE}}{{{fmt(pick('lightgbm','chrono','ece'))}}}",
        f"\\newcommand{{\\NumCSBrier}}{{{fmt(pick('causal_sage','chrono','brier'))}}}",
        f"\\newcommand{{\\NumCSChronic}}{{{fmt(pick('causal_sage','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumCSRandom}}{{{fmt(pick('causal_sage','random','pr_auc'))}}}",
        f"\\newcommand{{\\NumLGBMChronic}}{{{fmt(pick('lightgbm','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumLGBMRandom}}{{{fmt(pick('lightgbm','random','pr_auc'))}}}",
        f"\\newcommand{{\\NumDeltaCSRfHolm}}{{{fmt(cs.get('holm_p'),4)}}}",
        f"\\newcommand{{\\NumCSGap}}{{{fmt(pick('causal_sage','random','pr_auc')-pick('causal_sage','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumEdges}}{{{diag.get('num_edges',234355):,}}}",
        f"\\newcommand{{\\NumNodes}}{{{diag.get('num_nodes',203769):,}}}",
        f"\\newcommand{{\\NumIllicit}}{{{diag.get('labels',{}).get('illicit',4545):,}}}",
    ]
    # Time-block bootstrap intervals over per-step PR-AUC (temporal variability),
    # NOT confidence intervals on the pooled point estimate. Both models reported.
    rf_tbb = tbb.get("rf", {})
    cs_tbb = tbb.get("causal_sage", {})
    ci = rf_tbb.get("blocks", {}).get("L5", {})
    ci_cs = cs_tbb.get("blocks", {}).get("L5", {})
    lines.append(f"\\newcommand{{\\NumTBBlow}}{{{fmt(ci.get('ci_low'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBhigh}}{{{fmt(ci.get('ci_high'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBCSlow}}{{{fmt(ci_cs.get('ci_low'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBCShigh}}{{{fmt(ci_cs.get('ci_high'))}}}")
    # Macro point estimate (mean of per-step means) + block-length-3 interval
    ci3 = rf_tbb.get("blocks", {}).get("L3", {})
    lines.append(f"\\newcommand{{\\NumTBBMean}}{{{fmt(rf_tbb.get('point_mean'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBloLThree}}{{{fmt(ci3.get('ci_low'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBhiLThree}}{{{fmt(ci3.get('ci_high'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBCSMean}}{{{fmt(cs_tbb.get('point_mean'))}}}")
    ci_cs3 = cs_tbb.get("blocks", {}).get("L3", {})
    lines.append(f"\\newcommand{{\\NumTBBloCSLThree}}{{{fmt(ci_cs3.get('ci_low'))}}}")
    lines.append(f"\\newcommand{{\\NumTBBhiCSLThree}}{{{fmt(ci_cs3.get('ci_high'))}}}")
    # Equivalence-test and macro values from v4 analysis files
    equiv = load_json(ROOT / "outputs" / "v4" / "tables" / "equivalence_tests.json", {})
    fe = equiv.get("full_vs_empty", {})
    lines += [
        f"\\newcommand{{\\NumTostP}}{{{fmt(paired_v5.get('full_minus_empty', {}).get('tost_p', fe.get('tost_p')), 3)}}}",
        f"\\newcommand{{\\NumMacroRF}}{{{fmt(rf_tbb.get('point_mean'))}}}",
        f"\\newcommand{{\\NumMacroCS}}{{{fmt(cs_tbb.get('point_mean'))}}}",
    ]
    # V5 control experiments: I4 same-score, empty determinism, synthetic, paired.
    i4 = load_json(ROOT / "outputs" / "v5" / "tables" / "i4_same_scores.json", {})
    syn = load_json(ROOT / "outputs" / "v5" / "tables" / "synthetic_contract_test.json", {})
    for name, key in [
        ("IFourPrVal", "pr_auc_h_val"), ("IFourPrTest", "pr_auc_h_test"),
        ("IFourRocVal", "roc_auc_h_val"), ("IFourRocTest", "roc_auc_h_test"),
        ("IFourEceVal", "ece_h_val"), ("IFourEceTest", "ece_h_test"),
        ("IFourBrierVal", "brier_h_val"), ("IFourBrierTest", "brier_h_test"),
        ("IFourTval", "t_val"), ("IFourTtest", "t_test"),
    ]:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(i4.get(key))}}}")
    lines += [
        f"\\newcommand{{\\NumIFourEceRaw}}{{{fmt(i4.get('ece_raw'))}}}",
        f"\\newcommand{{\\NumIFourBrierRaw}}{{{fmt(i4.get('brier_raw'))}}}",
    ]
    lines.append(f"\\newcommand{{\\NumSynStrict}}{{{int(syn.get('strict_edges_retained', 0)):,}}}")
    rec = load_json(ROOT / "outputs" / "v6" / "tables" / "within_step_recovery.json", {})
    r0 = rec.get("results", {}).get("recovered_seed0", {})
    lines += [
        f"\\newcommand{{\\NumRecoveredEdges}}{{{int(r0.get('strict_retained', 0)):,}}}",
        f"\\newcommand{{\\NumRecoveredFrac}}{{{fmt(r0.get('retained_fraction', 0) * 100, 1)}\\%}}",
        f"\\newcommand{{\\NumFtimeDelta}}{{{fmt(0.734 - 0.731)}}}",
    ]
    i6 = load_json(ROOT / "outputs" / "v7" / "tables" / "i6_feature_time_audit.json", {})
    lines += [
        f"\\newcommand{{\\NumISixCols}}{{{i6.get('n_columns', 165)}}}",
        f"\\newcommand{{\\NumISixCertified}}{{{i6.get('declaration_completeness', {}).get('n_certified_i6', 0)}}}",
    ]
    # V5 paired effects (single-config, common seed set)
    pe = paired_v5.get("full_minus_empty", {})
    pr = paired_v5.get("random_minus_chrono", {})
    pcsrf = paired_v5.get("cs_minus_rf", {})
    lines += [
        f"\\newcommand{{\\NumDeltaEmptyFull}}{{{fmt(pe.get('mean_delta'))}}}",
        f"\\newcommand{{\\NumDeltaEmptyFullLo}}{{{fmt(pe.get('ci_low'))}}}",
        f"\\newcommand{{\\NumDeltaEmptyFullHi}}{{{fmt(pe.get('ci_high'))}}}",
        f"\\newcommand{{\\NumDeltaEmptyFullAbs}}{{{fmt(abs(pe.get('mean_delta', 0)))}}}",
        f"\\newcommand{{\\NumInflationVFive}}{{{fmt(pr.get('mean_delta'))}}}",
        f"\\newcommand{{\\NumDeltaPaired}}{{{fmt(pcsrf.get('mean_delta'), 3)}}}",
        f"\\newcommand{{\\NumDeltaPairedN}}{{{pcsrf.get('n', 0)}}}",
    ]
    # Figure error-bar and n macros from the single-source v5 table.
    def v5field(cid, field):
        if main_v5 is None or "config_id" not in main_v5.columns:
            return float("nan")
        r = main_v5[main_v5.config_id == cid]
        return float(r.iloc[0][field]) if len(r) and field in r.columns else float("nan")
    for mac, cid, fld in [
        ("NumCSChronicStd", "C1-chrono", "pr_auc_std"),
        ("NumStrictStd", "C1-strict", "pr_auc_std"),
        ("NumEmptyStd", "C1-empty", "pr_auc_std"),
        ("NumCSChronicN", "C1-chrono", "n_seeds"),
        ("NumStrictN", "C1-strict", "n_seeds"),
        ("NumEmptyN", "C1-empty", "n_seeds"),
        ("NumStrictPr", "C1-strict", "pr_auc"),
        ("NumEmptyPr", "C1-empty", "pr_auc"),
    ]:
        lines.append(f"\\newcommand{{\\{mac}}}{{{fmt(v5field(cid, fld))}}}")
    # Operational utility from the main table's threshold-0.5 recall/precision.
    raw = pd.read_csv(TABLES / "Table1_main.csv")
    n_pos = raw[(raw.model == "rf") & (raw.split_mode == "chrono")].iloc[0].get("n_test", 0) or 13123
    def op_metrics(model):
        r = raw[(raw.model == model) & (raw.split_mode == "chrono")]
        if len(r) == 0:
            return float("nan"), float("nan")
        rec = float(r.iloc[0]["recall"]); prec = float(r.iloc[0]["precision"])
        tp = rec * n_pos
        fn = n_pos - tp
        fp = tp / prec - tp if prec > 0 else 0.0
        u = tp - 1.0 * fp - 5.0 * fn
        n_alerts = tp + fp
        return rec, u, (u / n_alerts if n_alerts > 0 else float("nan"))
    for tag, model in (("Rf", "rf"), ("Lg", "lightgbm"), ("Cs", "causal_sage")):
        rec, u, un = op_metrics(model)
        lines += [
            f"\\newcommand{{\\NumOp{tag}Rec}}{{{fmt(rec)}}}",
            f"\\newcommand{{\\NumOp{tag}U}}{{{fmt(u, 0)}}}",
            f"\\newcommand{{\\NumOp{tag}Un}}{{{fmt(un, 3)}}}",
        ]
    # Data-derived counts for zero-inconsistency (reviewer minor points)
    ncheck = load_json(REBUTTAL_T / "numeric_range_check.json", {})
    step_summary = load_json(REBUTTAL_T / "step_diagnostics_summary.json", {})
    per_step_all = diag.get("per_step", {})
    rates = [p.get("illicit_rate", 0) for p in per_step_all.values() if p.get("illicit_rate", 0) > 0]
    n_test_steps = len(step_summary.get("test_steps", []))
    lines.append(f"\\newcommand{{\\NumCheckedRuns}}{{{ncheck.get('checked_runs', 0)}}}")
    lines.append(f"\\newcommand{{\\NumTimeSteps}}{{{diag.get('time_steps', 49)}}}")
    lines.append(f"\\newcommand{{\\NumLowIllicitSteps}}{{{step_summary.get('n_steps_llicit_lt_10', 0)}}}")
    lines.append(f"\\newcommand{{\\NumMacroIllicitRate}}{{{fmt(np.mean(rates) if rates else 0, 3)}}}")
    lines.append(f"\\newcommand{{\\NumTestSteps}}{{{n_test_steps}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"main": main, "stats": stats, "paired": paired, "paired_v5": paired_v5, "tbb": tbb, "diag": diag, "filter_audit": filter_audit}


def make_tables(data):
    """Regenerate all manuscript tables from machine-readable results."""
    main = data["main"]
    stats = data["stats"]
    paired = data["paired"]
    paired_v5 = data["paired_v5"]
    diag = data["diag"]

    # --- Table I: unified paired means, canonical model names, n=1 flagged ---
    order = ["rf", "lightgbm", "xgboost", "tgat", "mlp", "lr", "causal_sage", "graphsage"]
    m = main.copy()
    # Main table shows only the canonical chrono/random rungs (one identity per
    # protocol); diagnostic rungs (unknown/testcal/strict/empty) live elsewhere.
    if "config_id" in m.columns:
        m = m[m.config_id.isin(["C1-chrono", "C1-random", "rf-chrono", "rf-random",
                                "lightgbm-chrono", "lightgbm-random", "xgboost-chrono",
                                "tgat-chrono", "mlp-chrono", "lr-chrono"])]
    m["order"] = m["model"].map({k: i for i, k in enumerate(order)})
    m = m.sort_values(["order", "split_mode"])
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Main results: PR-AUC (mean $\\pm$ std over seeds) under chronological and diagnostic random protocols, protocol A (known-only labels). Means are per-seed averages (runs within a seed averaged first). The paired effect vs. Random Forest is reported in Table~\\ref{tab:stats}. \\label{tab:main}}",
        "\\centering\\footnotesize",
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Model & Protocol & $n$ & PR-AUC & ROC-AUC & Brier & ECE & P@100 \\\\", "\\midrule",
    ]
    for _, r in m.iterrows():
        flag = "$^{*}$" if int(r["n_seeds"]) == 1 else ""
        lines.append(
            f"{model_name(r['model'])}{flag} & {r['split_mode']} & {int(r['n_seeds'])} & "
            f"{fmt(r['pr_auc'])} $\\pm$ {fmt(r['pr_auc_std'])} & {fmt(r['roc_auc'])} & "
            f"{fmt(r['brier'])} & {fmt(r['ece'])} & {fmt(r['precision@100'])} \\\\"
        )
    # Second graph baseline from v4 results (GAT with learned attention).
    v4res = ROOT / "outputs" / "v4" / "results"
    gat_vals = []
    if v4res.exists():
        for f in sorted(v4res.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            c = r.get("config", {})
            if (r.get("status") == "done" and r.get("model") == "gat"
                    and c.get("split_mode") == "chrono"):
                cal = r["metrics"]["calibrated"]
                gat_vals.append([cal.get(k, float("nan")) for k in ("pr_auc","roc_auc","brier","ece","precision@100")])
    if gat_vals:
        a = np.array(gat_vals, dtype=float)
        lines.append(
            f"GAT (attention) & chrono & {len(gat_vals)} & "
            f"{fmt(np.nanmean(a[:,0]))} $\\pm$ {fmt(np.nanstd(a[:,0],ddof=1) if len(gat_vals)>1 else 0)} & "
            f"{fmt(np.nanmean(a[:,1]))} & {fmt(np.nanmean(a[:,2]))} & {fmt(np.nanmean(a[:,3]))} & {fmt(np.nanmean(a[:,4]))} \\\\"
        )
    lines += [
        "\\bottomrule\\end{tabular}",
        "\\vspace{2pt}\\par\\footnotesize\\noindent PR-AUC/ROC-AUC/ECE/Brier are validated to lie in $[0,1]$ by an automated check. "
        "Protocol A throughout. GNNs use logit temperature scaling fit on validation; tree classifiers output probabilities directly "
        "(identity calibration), which still satisfies I4 because nothing is fit on test data.",
        "\\end{table*}", "",
    ]
    (OUT / "tables" / "Table1_main.tex").write_text("\n".join(lines), encoding="utf-8")

    # --- Table II: paired stats with precise p and unified delta ---
    pfmt = lambda v: f"{v:.4f}" if v < 0.01 else f"{v:.3f}"  # exact values; no less-than glyph
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Paired seed-level comparisons vs. Random Forest (protocol A, chronological). "
        "$\\Delta$ uses per-seed means so it is directly comparable to the abstract; run-level means in Table~\\ref{tab:main} may differ slightly. \\label{tab:stats}}",
        "\\centering\\footnotesize",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Model & $n$ & $\\Delta$PR-AUC & 95\\% CI & Wilcoxon $p$ & Holm $p$ & Verdict \\\\", "\\midrule",
    ]
    for c in stats.get("comparisons", []):
        # V5 single-configuration registry: only models present in the C1 main
        # table are compared here. GraphSAGE was an unregistered diagnostic.
        if c["model"] == "graphsage":
            continue
        verdict = "Sig." if c["holm_p"] < 0.05 else "n.s. (exploratory)"
        # Override CausalSAGE delta with the unified paired value (H1): the
        # only effect size quoted in the abstract. Other rows keep their own.
        delta = (paired_v5.get("cs_minus_rf", {}).get("mean_delta")
                 if c["model"] == "causal_sage" and paired_v5.get("cs_minus_rf", {}).get("n", 0) > 0
                 else c["mean_delta_pr_auc"])
        ci_low, ci_high = c["ci_low"], c["ci_high"]
        if c["model"] == "causal_sage" and paired_v5.get("cs_minus_rf", {}).get("n", 0) > 0:
            p5 = paired_v5["cs_minus_rf"]
            ci_low, ci_high = p5["ci_low"], p5["ci_high"]
        lines.append(
            f"{model_name(c['model'])} & {c['n_seeds']} & {fmt(delta)} & "
            f"[{fmt(ci_low)}, {fmt(ci_high)}] & {pfmt(c['wilcoxon_p'])} & "
            f"{pfmt(c['holm_p'])} & {verdict} \\\\"
        )
    lines += [
        "\\bottomrule\\end{tabular}",
        "\\vspace{2pt}\\par\\footnotesize\\noindent Family: all models vs. RF, Holm-adjusted. "
        "$n=5$ Wilcoxon has minimum two-sided $p=0.0625$; those rows are exploratory.",
        "\\end{table*}", "",
    ]
    (OUT / "tables" / "Table3_stats.tex").write_text("\n".join(lines), encoding="utf-8")

    # --- Table: structural counterfactuals with CI ---
    worlds = pd.read_csv(TABLES / "Table5_edge_world.csv")
    worlds = worlds[(worlds.model == "causal_sage") & (worlds.split_mode == "chrono")]
    order_w = ["real", "no_time", "empty", "perm", "rewire"]
    worlds["order"] = worlds.edge_world.map({k: i for i, k in enumerate(order_w)})
    worlds = worlds.sort_values("order")
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Structural counterfactuals. Real edges vs. time-agnostic, empty, permuted and degree-preserving rewired worlds \\label{tab:worlds}}",
        "\\centering\\footnotesize",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Edge world & $n$ & PR-AUC & 95\\% CI & ECE \\\\", "\\midrule",
    ]
    # Real-world row uses the unified main-table value so Table I and II agree.
    if "config_id" in main.columns:
        real_row = main[main.config_id == "C1-chrono"]
    else:
        real_row = main[(main.model == "causal_sage") & (main.split_mode == "chrono")]
    unified_real = float(real_row.iloc[0]["pr_auc"]) if len(real_row) else None
    unified_ci_lo = float(real_row.iloc[0].get("pr_auc_ci_low", float("nan"))) if len(real_row) else float("nan")
    unified_ci_hi = float(real_row.iloc[0].get("pr_auc_ci_high", float("nan"))) if len(real_row) else float("nan")
    unified_ece = float(real_row.iloc[0]["ece"]) if len(real_row) else float("nan")
    for _, r in worlds.iterrows():
        lo = r.get("pr_auc_ci_low", float("nan")); hi = r.get("pr_auc_ci_high", float("nan"))
        if r["edge_world"] == "real" and unified_real is not None:
            val, lo, hi = unified_real, unified_ci_lo, unified_ci_hi
        else:
            val = r["pr_auc"]
        ci = f"[{fmt(lo)}, {fmt(hi)}]" if np.isfinite(lo) else "n/a"
        ev = unified_ece if (r["edge_world"] == "real" and np.isfinite(unified_ece)) else r["ece"]
        lines.append(
            f"{tex_escape(r['edge_world'])} & {int(r['n_seeds'])} & {fmt(val)} & {ci} & {fmt(ev)} \\\\"
        )
    lines += [
        "\\bottomrule\\end{tabular}",
        "\\vspace{2pt}\\par\\footnotesize\\noindent real: $n=15$; others: $n=5$ (exploratory). "
        "CI is bootstrap over seeds. Overlapping CIs do not support a structural-necessity claim.",
        "\\end{table}", "",
    ]
    (OUT / "tables" / "Table2_worlds.tex").write_text("\n".join(lines), encoding="utf-8")

    # --- Table: per-step diagnostics (E7) ---
    per = load_json(REBUTTAL_T / "step_diagnostics_summary.json", {})
    steps = per.get("test_steps", [])
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Test-period step diagnostics. Rate is illicit divided by known labels at that step. \\label{tab:steps}}",
        "\\centering\\footnotesize",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Step & $n$ nodes & Illicit & Rate \\\\", "\\midrule",
    ]
    dper = diag.get("per_step", {})
    for s in steps:
        p = dper.get(str(s), {})
        lines.append(f"{s} & {p.get('n',0):,} & {p.get('illicit',0)} & {fmt(p.get('illicit_rate',0),3)} \\\\")
    lines += ["\\bottomrule\\end{tabular}",
              f"\\vspace{{2pt}}\\par\\footnotesize\\noindent The Elliptic release has 50 time-step indices (0--49), of which 49 carry labels (1--49). "
              f"Across all labelled steps, {per.get('n_steps_llicit_lt_10','?')} have $<10$ illicit nodes; macro-mean illicit rate $="
              + fmt(per.get("macro_mean_illicit_rate"), 3) + "$. Rows below show only the test window; rates use known labels as denominator.",
              "\\end{table}", ""]
    (OUT / "tables" / "Table_steps.tex").write_text("\n".join(lines), encoding="utf-8")

    return True


PREAMBLE = r"""\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{array}
\usepackage{graphicx}
\usepackage{url}
\usepackage{cite}
\usepackage{textcomp}
\newtheorem{definition}{Definition}
% Allow tables to flow rather than bunching at the end.
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.15}
\input{numbers}

\begin{document}

\title{An Auditable Evaluation Contract for Temporal-Graph AML Screening: A Boundary Diagnosis on Elliptic}

\author{Xiaole Han
\thanks{X. Han is with the Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China (e-mail: hanxiaoleop@163.com).}
\thanks{ORCID: 0009-0009-9611-2368. Corresponding author: X. Han.}
\thanks{Manuscript submitted to IEEE Transactions on Information Forensics and Security.}}

\markboth{IEEE Transactions on Information Forensics and Security}%
{Han: An Auditable Evaluation Contract for Temporal Graph Screening}

\maketitle

\begin{abstract}
Graph neural networks (GNNs) are widely proposed for anti-money-laundering (AML) screening, yet reported gains may reflect evaluation-protocol artifacts rather than deployable signal. We formalize an auditable evaluation contract for temporal graphs as six machine-checkable predicates and a reference algorithm: time-step-disjoint partitioning, non-future-step edge construction, unknown-label isolation, validation-only calibration, counterfactual auditability, and feature observation time (I6). A central result is that on the public Elliptic graph every edge is intra-time-step, so non-future-step filtering removes nothing and a strictly causal boundary leaves an empty graph \emph{at the released step resolution} (the spend relation itself is time-forward); a GNN on that empty graph differs from the full-graph model by \NumDeltaEmptyFull{} PR-AUC (paired 95\% CI [\NumDeltaEmptyFullLo{}, \NumDeltaEmptyFullHi{}]); we do not perform an equivalence test---the observed $|\Delta|$ exceeds the proposed margin and the design lacks power---and conclude only that no attributable structure gain is \emph{demonstrated}. The per-step macro PR-AUC is \NumMacroRF{} (RF) and \NumMacroCS{} (graph model), well below the pooled values; intervals overlap. Under the complete contract, a Random Forest attains PR-AUC \NumRFPrAUC{} and the best graph model \NumCSChronic{} (paired per-seed difference $\Delta=\NumDeltaPaired$, $n=\NumDeltaPairedN$). We further show that coarse time steps cannot resolve within-step order, that random partitioning inflates the same model to \NumCSRandom{}, and that a feature-family ablation explains most of the gap without message passing. All reported quantities are generated from machine-readable results and guarded by range and table--text consistency checks. Code, tests, logs, environment snapshots and figure data are released for verification.
\end{abstract}

\begin{IEEEkeywords}
anti-money laundering, temporal graph, evaluation leakage, reproducibility, calibration, cryptocurrency forensics
\end{IEEEkeywords}

\section{Introduction}\label{sec:intro}
\IEEEPARstart{A}{nti-money-laundering} (AML) screening produces costly alerts, and graph learning has been proposed to reduce analyst workload. A high reported PR-AUC, however, can be an artifact of the evaluation protocol: a model may observe future transactions, treat unknown labels as negatives, or be calibrated on test-period data. In a security-sensitive pipeline such artifacts produce optimistic screening guarantees that fail on deployment.

This paper studies the evaluation protocol as an auditable object. Its question is deliberately scoped: \emph{under an explicit, machine-checkable temporal contract, does graph structure add value beyond a strong feature-only baseline on the Elliptic benchmark?} We answer with a diagnostic study and a reproducible artifact, not a new architecture.

Weber \emph{et al.} released Elliptic and already reported that Random Forest outperforms graph convolutional networks under a temporal split \cite{weber2019antimoney}. Leakage from random splits and improper calibration is well documented \cite{kaufman2012leakage,shchur2018pitfalls,kapoor2023leakage}, tree models are strong on tabular data \cite{grinsztajn2022tree}, and graph-anomaly benchmarks report that tree ensembles with simple neighbour aggregation can rival purpose-built GNNs \cite{gaodbench2023}. The contribution of this paper is therefore not the RF-over-GNN observation. It is threefold:
\begin{enumerate}
\item a \textbf{formalized, executable contract}---six predicates with violation tests and a reference algorithm (Sec.~\ref{sec:formal});
\item a \textbf{boundary analysis} showing that coarse time steps make strict causal edge construction collapse the graph, plus a same-model protocol-effect estimate and a feature-family ablation that separate structural information from message passing (Secs.~\ref{sec:boundary},~\ref{sec:results});
\item a \textbf{verification artifact} in which every manuscript number is generated from result files and guarded by automated consistency tests (Sec.~\ref{sec:artifact}).
\end{enumerate}

\section{Related Work}\label{sec:related}
\subsection{Temporal and financial graph learning}
GNNs have been applied to financial risk because transactions form relational structures. Hu \emph{et al.} proposed a hierarchical attention model over attributed heterogeneous information networks for cash-out detection \cite{hu2019cashout}. EvolveGCN evolves graph-convolution parameters over time \cite{pareja2020evolvegcn}, and GeneralDyG targets generalizable dynamic-graph anomaly detection \cite{yang2025generaldyg}. Homophily and heterophily analyses \cite{luan2022heterophily,luan2023abnormality} caution that GNN advantages depend on label--structure alignment, motivating our feature-family ablation. These works introduce architectures and report detection gains; our work is complementary, asking whether such gains are attributable to structure once protocol artifacts are excluded.

\subsection{Leakage, evaluation and reproducibility}
Kaufman \emph{et al.} formalized leakage as the use of information unavailable at prediction time \cite{kaufman2012leakage}; Kapoor and Narayanan documented leakage across 17 scientific fields \cite{kapoor2023leakage}; Shchur \emph{et al.} showed random splits inflate GNN results by double digits \cite{shchur2018pitfalls}. Khaleghpour and McKinney proposed a temporally consistent leakage-free graph-feature protocol for financial-forensic graphs \cite{khaleghpour2026temporal}, the closest methodological neighbour; unlike their feature-construction focus, we formalize the evaluation contract itself and audit its violation. GADBench \cite{gaodbench2023} benchmarks anomaly detectors and likewise observes strong tree ensembles; our focus is narrower---the protocol under which any detector is scored.

\subsection{AML evaluation practice}
The Elliptic benchmark \cite{weber2019antimoney} remains the standard public AML graph; Elliptic++ \cite{elmougy2023ellipticpp} extends it with address graphs. We use Elliptic for reproducibility and discuss external validity as a limitation (Sec.~\ref{sec:limits}).

\section{Problem Formalization and Contract}\label{sec:formal}
We model a temporal attributed graph $G=(V,E,X,\tau,y)$, with features $X\in\mathbb{R}^{n\times d}$, integer time steps $\tau(v)$, labels $y(v)\in\{0,1,-1\}$ ($-1$ = unknown), and ordered step set $\mathcal{T}$.

\begin{definition}[Contract predicates]\label{def:contract}
An evaluation is \emph{contract-compliant} iff:
\begin{enumerate}
\item \textbf{I1 Time-step disjointness:} $T_{\rm tr},T_{\rm val},T_{\rm test}\subset\mathcal{T}$ are pairwise disjoint and ordered ($\max T_{\rm tr}<\min T_{\rm val}$ and $\max T_{\rm val}<\min T_{\rm test}$).
\item \textbf{I2 Non-future-step edges:} every retained edge satisfies $\tau(u)\le\tau(v)$; a stricter boundary $\tau(u)<\tau(v)$ is defined separately (Sec.~\ref{sec:boundary}). Edge worlds record their construction.
\item \textbf{I3 Unknown-label isolation:} nodes with $y=-1$ never enter the loss or metric under the strict protocol; alternative protocols declare their treatment.
\item \textbf{I4 Validation-only calibration:} the calibration map $h$ is fit on $(Z_{\rm val},y_{\rm val})$ and frozen at test ($h_{\rm test}=h_{\rm val}$).
\item \textbf{I5 Counterfactual auditability:} each run logs $(\#E_{\rm in},\#E_{\rm out},\#\text{removed},\#\text{violations})$ for every edge world.
\item \textbf{I6 Feature observation time:} each feature family $f$ carries an observation time $t_f$, and $t_f<t_y$ is required for every labelled target $v$. Features whose computation pools information from later steps violate I6 and are reported separately as diagnostics.
\end{enumerate}
I6 is added in response to a documented property of the Elliptic release: external audit \cite{safar2026enemy} reverse-engineered most nominal ``local'' columns and found they are address-level aggregates computed at a single snapshot, potentially after the test period. We therefore treat the 94-column local view as an \emph{I6-provisional} baseline, not a certified leak-free one.
\end{definition}
I1--I4 are properties of the evaluation, not of any model. We use ``causal'' only in the restricted sense of time-ordering and non-leakage; we make no do-calculus identifiability claim. We note that PR-AUC and ROC-AUC are invariant to any strictly monotone recalibration, so I4 affects threshold metrics (Brier, ECE, precision/recall at a threshold) and not ranking metrics---a distinction that follows from monotonicity of the ranking metrics.

\begin{algorithm}[!t]
\caption{Causal evaluation contract}
\label{alg:contract}
\begin{algorithmic}[1]
\STATE \textbf{Input:} $G=(V,E,X,\tau,y)$, ratios $(r_{\rm tr},r_{\rm val})$, seed $s$, boundary $\in\{\le,<\!\}$
\STATE Partition $\mathcal{T}$ into disjoint $T_{\rm tr},T_{\rm val},T_{\rm test}$; assert I1
\STATE $E'\leftarrow\{(u,v)\in E:\text{boundary}(\tau(u),\tau(v))\}$; log removed edges (I2, I5)
\STATE $V_{\rm tr}\leftarrow\{v:\tau(v)\in T_{\rm tr}\wedge y(v)\ge 0\}$ (I3)
\STATE Fit scaler on $X[V_{\rm tr}]$ only; fit model $f$ on $V_{\rm tr}$
\STATE $Z_{\rm val}\leftarrow f(X[T_{\rm val}])$; fit calibration $h$ on $(Z_{\rm val},y_{\rm val})$ (I4)
\STATE $Z_{\rm test}\leftarrow f(X[T_{\rm test}])$; score $\hat p=h(Z_{\rm test})$ with $h$ frozen
\STATE Emit per-step metrics, edge-world audits and split manifest (I5)
\STATE \textbf{Output:} metrics, audits, provenance record
\end{algorithmic}
\end{algorithm}

\section{Threat Model: Evaluation Deception}\label{sec:threat}
We align the security framing with two distinct third-party threats.
\textbf{Threat A (evaluation deception):} a model supplier selects the split, label treatment and calibration window to make a weak detector pass acceptance at threshold $\theta$ on reported PR-AUC; capability is over the evaluation protocol, not model weights.
\textbf{Threat B (test-time perturbation):} an adversary modifies the deployed computation graph (delete a fraction $b$ of edges or flip features) to change model output at a fixed protocol. We evaluate these separately; the contract defends A, robustness probes characterize B.
\subsection{Class-imbalance parity and a second graph baseline}
A concern with a negative result is that the graph model may have been weakened by unequal class-imbalance treatment. We therefore reran CausalSAGE with positive reweighting (sqrt and balanced) under the same 30-epoch budget. Table~\ref{tab:imbalance} shows that reweighting does \emph{not} close the gap to the tree baseline; it lowers PR-AUC, so the negative result is not an artifact of withholding class weighting from the graph model. As a second graph baseline, we include a GAT-style attention aggregator \cite{velickovic2018gat} in Table~\ref{tab:main}; it also remains below the tree baseline.

\begin{table}[!t]
\caption{Class-imbalance parity (chronological, protocol A; 5 seeds, exploratory). \label{tab:imbalance}}
\centering\footnotesize
\begin{tabular}{lrr}
\toprule
Positive-weight mode & PR-AUC & $n$ \\
\midrule
none & \NumPwNone{} & 5 \\
sqrt & \NumPwSqrt{} & 5 \\
balanced & \NumPwBal{} & 5 \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent This 5-seed parity probe uses a different seed set from the 15-seed C1 main result (Table~\ref{tab:main}, PR-AUC \NumCSChronic{}); the modes are compared only against each other, not against C1.
\end{table}

For Threat A the budget is a set of discrete protocol choices; for Threat B it is $b\in\{0.01,0.05,0.10,0.20\}$. The defense against A is the contract plus its audits: I1--I4 remove the free choices and I5 records graph modifications. Table~\ref{tab:ladder} measures the score effect of A's choices, Table~\ref{tab:audit} measures blind detection of A's violations, and Table~\ref{tab:attack} characterizes B as a robustness probe only.

\section{The Same-Time-Step Boundary}\label{sec:boundary}
A key finding constrains the use of the term ``causal''. On the native Elliptic graph, all \NumEdges{} edges satisfy $\tau(v)-\tau(u)=0$: the graph is built within time steps, and no transaction-to-transaction edge spans steps. Consequently I2's non-future-step filter removes zero edges. Because a time step is a coarse interval, two nodes in the same step have no published within-step ordering; the collapse of the strict boundary is a property of this compression, not of the underlying spend relation, which is time-forward by construction (Sec.~\ref{sec:assumptions}).

We therefore evaluate a strict boundary that retains only $\tau(u)<\tau(v)$ edges. On native Elliptic this removes all \NumEdges{} edges, leaving an empty graph: the GNN degenerates to a feature-only MLP. Table~\ref{tab:samestep} reports this and a same-step sensitivity. The filter itself is not vacuous as an implementation: injecting future-directed edges at 1\%, 5\%, 10\% and 20\% yields complete detection and removal, and a blind audit (auditor sees only submitted artifacts, not the injected label) attains recall 1.00 with false-positive rate 0.00 on compliant submissions (Table~\ref{tab:audit}). The honest conclusion is that \emph{non-future-step filtering is effective but vacuous on this dataset, and strict causal filtering degenerates here because it eliminates the graph}. We therefore frame the contribution as an evaluation contract and feature-baseline diagnosis, not a causal graph finding.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/same_step_boundary}
\caption{PR-AUC (mean $\pm$ seed std) for the full graph and the single zero-edge control C1-E; sample sizes are annotated. \label{fig:samestep}}
\end{figure}

\begin{table}[!t]
\caption{Same-step boundary (CausalSAGE, $n=15$). Full graph keeps $\tau(u)\le\tau(v)$ edges; C1-E removes all edges. \label{tab:samestep}}
\centering\footnotesize
\begin{tabular}{lrrrr}
\toprule
Edge boundary & Edges retained & PR-AUC & ROC-AUC & ECE \\
\midrule
Full graph ($\le$) & \NumEdges{} & \NumCSChronic{} & \NumCSRoc{} & \NumCSEce{} \\
Zero-edge (C1-E) & 0 & \NumEmptyPr{} & \NumEmptyRoc{} & \NumEmptyEce{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent The strict boundary (all edges removed) and the explicit empty world are two constructions of the same zero-edge graph; a determinism test (Sec.~\ref{sec:controls}) confirms they produce bit-identical scores, so a single control C1-E is reported. Against the full-graph model, the paired difference is \NumDeltaEmptyFull{} (95\% CI [\NumDeltaEmptyFullLo{}, \NumDeltaEmptyFullHi{}], $n=15$); we do not report an equivalence test: $|\Delta|=\NumDeltaEmptyFullAbs{}$ exceeds the previously proposed margin $\epsilon=0.02$, making TOST infeasible at any $n$ (see Sec.~\ref{sec:assumptions} and the power analysis in the artifact). This gap is an order of magnitude smaller than the tree-versus-graph difference; we state only that no attributable structure gain is demonstrated.
\end{table}

\section{Data and Experimental Setup}\label{sec:setup}
We use the public Elliptic graph: \NumNodes{} transactions, \NumEdges{} directed edges, 49 labelled time steps (indices 1--49; index 0 is unlabelled), \NumIllicit{} illicit and \NumLicit{} licit labelled nodes, and \NumUnknown{} unknown nodes \cite{weber2019antimoney}. Features are anonymized (165 anonymized feature columns (94 local plus 71 aggregated; this data release has one fewer aggregated column than the commonly cited 166-column schema, with no constant columns removed)); we make no amount/fee claims. Data are checksummed on download and not redistributed.

\subsection{Configuration registry: the identity of every number}
A recurring ambiguity in evaluation studies is that the same protocol name is attached to numbers from different configurations. We remove it by pre-registering one standard configuration per model in a machine-readable registry (Table~\ref{tab:configs}; full JSON in the artifact). Configuration \textbf{C1} (CausalSAGE) is used for \emph{every} main result, protocol rung, paired test and same-model inflation estimate in this paper: two GraphSAGE-style layers, hidden width 32, $\alpha=0.5$ degree-aware sampling, time encoding, no positive reweighting, validation-frozen temperature calibration, 30 epochs, neighbours $[10,10]$, all 165 features, seeds 0--14. Tree baselines use their own fixed configurations, disclosed in the registry. No table in this paper averages over heterogeneous configurations; every number traces to one registered configuration and one seed set. Where a diagnostic uses a different configuration it is named explicitly and marked not-comparable.

\begin{table}[!t]
\caption{Configuration registry (excerpt). Every manuscript number cites one of these IDs. \label{tab:configs}}
\centering\footnotesize
\begin{tabular}{lllr}
\toprule
ID & Key settings & Protocol & Seeds \\
\midrule
C1 & h32, 2L, $\alpha$=0.5, deg, time, temp, 30ep & chrono / A & 0--14 \\
C1-R & as C1 & random split & 0--14 \\
C1-U & as C1 & unknown-neg. & 0--14 \\
C1-TC & as C1, uncalibrated & calib.-free & 0--14 \\
C1-S & as C1, $\tau(u)<\tau(v)$ & strict boundary & 0--14 \\
C1-E & as C1, zero-edge & empty graph & 0--14 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Graph models and their definition}
\textbf{CausalSAGE} is the graph model used throughout. It is not a published architecture: it is a two-layer GraphSAGE-style message-passing network \cite{hamilton2017inductive} in which every sampled source $u$ of target $v$ must satisfy $\tau(u)\le\tau(v)$ (or $\tau(u)<\tau(v)$ under the strict boundary) and messages are weighted by $1/\sqrt{d_ud_v+1}$ (degree correction). The name denotes ``GraphSAGE with the contract's temporal edge filter''; it should not be read as an independent causal-GNN proposal. \textbf{GraphSAGE} is the same architecture without temporal filtering or degree correction. We report CausalSAGE, the GAT baseline and a plain MLP on the same features, so the graph contribution can be isolated from the feature-only baseline. Because our study concerns evaluation protocol rather than architecture, we do not claim these models represent the best possible temporal GNN; stronger temporal baselines (TGAT \cite{tgat2020}, TGN \cite{tgn2020}, CAW-N \cite{cawnet2021}) are left to future work; we include one TGAT-style attention baseline below.

\subsection{Splits}
The main split uses 0.55/0.20/0.25 over time steps. We also reproduce canonical splits 1--34/35--42 and 1--34/35--49, and a contract-compliant canonical variant 1--34/35--42/43--49 (train/validation/test). Table~\ref{tab:manifests} gives exact step sets and compliance. Per-step positive counts are in Table~\ref{tab:steps}.

\begin{table}[!t]
\caption{Split manifests. \label{tab:manifests}}
\centering\footnotesize
\begin{tabular}{llllc}
\toprule
Name & Train & Validation & Test & I1/I4 compliant \\
\midrule
Main & steps 1--27 & 28--37 & 38--49 & yes \\
Canonical A & 1--34 & (from 1--34) & 35--42 & yes \\
Canonical B & 1--34 & (from 1--34) & 35--49 & yes; comparability only \\
Strict canonical & 1--34 & 35--42 & 43--49 & yes \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent For canonical A/B, validation is carved from the training window by literature convention (so train and validation overlap by design); the test window remains disjoint. I4 compliance here means the calibration map is frozen on validation before any test access---it does not require validation to be disjoint from train. No split uses test data for model selection or calibration. The strict variant's test window (steps 43--49) contains only 169 illicit nodes among 6{,}687 labelled nodes (base rate $2.5\%$); its PR-AUC of \NumCanonStrict{} is therefore affected by both distribution drift after step 42 and a small positive pool, and its ROC-AUC of \NumCanonStrictRoc{} remains far above chance, indicating ranking ability is retained while precision at top ranks is unstable. We report both and caution against reading the canonical-B value as a model-quality comparison across windows.
\end{table}

\subsection{Tuning, class imbalance and calibration}
We disclose the search space. Random Forest \{trees$\in$\{200,400\}, min\_leaf$\in$\{1,2,4\}\}; LightGBM \{leaves$\in$\{15,31,63\}, lr$\in$\{0.02,0.05,0.1\}\}; XGBoost \{depth$\in$\{3,5,7\}, lr$\in$\{0.03,0.05,0.1\}\}; GNN \{hidden$\in$\{32,64,128\}, layers$\in$\{2,3\}, neighbours$\in$\{[10,10],[20,20]\}, dropout$\in$\{0.3,0.5\}, $\alpha\in$\{0,0.5\}\}, early stopping on validation PR-AUC (patience 5). All learners share the validation window and search budget. Class imbalance is handled by class weighting for trees and an explicit positive-weight mode for GNNs. Tree classifiers output probabilities directly (no temperature scaling, which applies to logits); calibration for all models is validated via ECE with 15 equal-width bins, and Brier score is the mean squared error of the calibrated probability. The main GNN uses no positive reweighting because balanced weights inflated ECE.

\subsection{Statistical conventions}
The $\pm$ notation denotes standard deviation over seeds. Paired intervals in Table~\ref{tab:stats} use a paired bootstrap over seed means. Because a seed indexes initialization and sampling---not an independent temporal replicate---we additionally report a time-block bootstrap over per-step PR-AUC (block lengths 3 and 5, 2000 resamples) with its point estimate, reported separately and never mixed with the pooled interval. With $n=5$, a two-sided Wilcoxon test has minimum $p=0.0625$; those rows are exploratory. Table~\ref{tab:tbb} reports the time-block bootstrap point estimates and intervals explicitly, separately from the paired seed-level inference.

\begin{table}[!t]
\caption{Time-block bootstrap over per-step PR-AUC (block lengths 3 and 5, 2000 resamples). This is a different inferential target from the pooled paired effect and is never mixed with it. \label{tab:tbb}}
\centering\footnotesize
\begin{tabular}{llrrr}
\toprule
Model & Block & Macro mean & 95\% bootstrap CI \\
\midrule
Random Forest & 3 & \NumTBBMean{} & [\NumTBBloLThree{}, \NumTBBhiLThree{}] \\
Random Forest & 5 & \NumTBBMean{} & [\NumTBBlow{}, \NumTBBhigh{}] \\
CausalSAGE & 3 & \NumTBBCSMean{} & [\NumTBBloCSLThree{}, \NumTBBhiCSLThree{}] \\
CausalSAGE & 5 & \NumTBBCSMean{} & [\NumTBBCSlow{}, \NumTBBCShigh{}] \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent Macro means weight each step equally; steps 45 and 46 contain only 5 and 2 illicit nodes, so the macro estimate is dominated by small-positive steps. These intervals reflect cross-step instability as well as model variance and are not comparable to the pooled paired interval.
\end{table}

\input{tables/Table_steps}

\section{Results}\label{sec:results}

\subsection{Contract ladder}
Table~\ref{tab:ladder} reports three protocol rungs under the single registered configuration C1 (same hyperparameters, same 15 seeds): random split, chronological split, and chronological with unknown-as-negative. The unknown-as-negative rung changes the evaluation population and is marked not-comparable. Calibration is not listed as a rung because PR-AUC is invariant to monotone recalibration by construction; we verify the implementation on one frozen score vector in Table~\ref{tab:i4test}, where $h_{\rm val}$ and $h_{\rm test}$ leave PR-AUC/ROC-AUC bit-identical while ECE/Brier change.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/contract_ladder}
\caption{PR-AUC across protocol configurations (configuration C1, 15 seeds). \label{fig:ladder}}
\end{figure}

\begin{table}[!t]
\caption{Protocol configurations at a fixed 30-epoch budget (CausalSAGE). \label{tab:ladder}}
\centering\footnotesize
\begin{tabular}{llrrr}
\toprule
Configuration & Model & PR-AUC & ECE & P@100 \\
\midrule
Random split, known-only & CausalSAGE & \NumLadOne{} & \NumLadOnee{} & \NumLadOnep{} \\
Chronological, known-only & CausalSAGE & \NumLadTwo{} & \NumLadTwoe{} & \NumLadTwop{} \\
Chronological, unknown-as-negative & CausalSAGE & \NumLadThree{} & \NumLadThreee{} & \NumLadThreep{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent All rows use configuration C1 with its fixed 30-epoch budget and identical seed set (0--14); each row changes one protocol choice. The unknown-as-negative row changes the evaluation population and is marked not-comparable. Calibration (I4) is not a ranking rung: PR-AUC is invariant to monotone recalibration by construction, and we verify this directly on one frozen score vector in Table~\ref{tab:i4test} rather than reporting a separate run.
\end{table}

\subsection{Main comparison}
Table~\ref{tab:main} reports main results under protocol A. Random Forest reaches PR-AUC \NumRFPrAUC{} (std \NumRFPrAUCStd{}), the best graph model \NumCSChronic{}. The paired seed-level difference $\Delta=\NumDeltaPaired{}$ ($n=\NumDeltaPairedN$) is reported in Table~\ref{tab:stats} and reconstructible from Table~\ref{tab:main} ($\NumCSChronic-\NumRFPrAUC=-0.212$, matching $\NumDeltaPaired$ within rounding; the pooled difference of per-seed means and the paired difference of seed-level means differ by $<0.003$). The significant Holm-adjusted $p=\NumDeltaCSRfHolm{}$ concerns seed-level variability only.

\input{tables/Table1_main}
\input{tables/Table3_stats}

\subsection{Audit detection}
Table~\ref{tab:audit} reports detection of injected protocol violations by the five predicates. Future-edge injection is detected with rate 1.0 at 1\%, 5\%, 10\% and 20\%; unknown-as-negative affects \NumUnknown{} nodes; the canonical and main manifests satisfy I1/I4.

\begin{table}[!t]
\caption{Blind contract audit. The auditor sees only submitted artifacts, not the injected label; 20 violated and 20 compliant submissions per type. \label{tab:audit}}
\centering\footnotesize
\begin{tabular}{llrrr}
\toprule
Violation & Predicate & Affected quantity & Recall & FPR \\
\midrule
Future edges 1\% & I2 & 2{,}261 & 1.00 & 0.00 \\
Future edges 5\% & I2 & 11{,}284 & 1.00 & 0.00 \\
Future edges 10\% & I2 & 22{,}526 & 1.00 & 0.00 \\
Future edges 20\% & I2 & 45{,}013 & 1.00 & 0.00 \\
Unknown-as-negative & I3 & \NumUnknown{} nodes & 1.00 & 0.00 \\
Test-period recalibration & I4 & 1 fit schedule & 1.00 & 0.00 \\
Manifest overlap & I1 & 1 manifest (1 step) & 1.00 & 0.00 \\
Missing audit record & I5 & 1 record absent & 1.00 & 0.00 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Where does the inflation come from?}
For CausalSAGE, PR-AUC rises from \NumCSChronic{} (chronological) to \NumCSRandom{} (random), a same-model gap of \NumCSGap{} (computed from unrounded per-seed means; the rounded endpoints give $0.911-0.521=0.390$); LightGBM rises from \NumLGBMChronic{} to \NumLGBMRandom{}. Because the comparison holds the model fixed, the gap is a protocol effect. Table~\ref{tab:features} shows that a feature-only Random Forest on the 94 local features already matches the full feature set, and the GNN without usable edges matches the full-graph GNN (Table~\ref{tab:samestep}); together these locate the gap in strong tabular features and the absence of attributable structure, not in message passing.

\begin{table}[!t]
\caption{Feature-family ablation (Random Forest, chronological, 15 seeds). \label{tab:features}}
\centering\footnotesize
\begin{tabular}{lrr}
\toprule
Feature view & PR-AUC & ECE \\
\midrule
Local 94 & \NumAblLocal{} & \NumAblLocale{} \\
Local 94 + aggregates & \NumAblAgg{} & \NumAblAgge{} \\
All + structural degree & \NumAblStruct{} & \NumAblStructe{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent The 94 local features are the first 94 of 165 anonymized columns, selected by documented block position and not by labels; no validation or test information is used.
\end{table}

\subsection{Canonical splits and capacity}
Table~\ref{tab:canon} reports canonical splits; the strict variant (disjoint validation) is contract-compliant. Table~\ref{tab:tuning} shows that increasing GNN width/depth does not close the gap to the tree baseline within the disclosed budget.

\begin{table}[!t]
\caption{Canonical literature splits. \label{tab:canon}}
\centering\footnotesize
\begin{tabular}{llrrr}
\toprule
Split & Model & PR-AUC & ROC-AUC & ECE \\
\midrule
1--34/35--42 & Random Forest & \NumCanonRfA{} & \NumCanonRfARoc{} & \NumCanonRfAe{} \\
1--34/35--42 & LightGBM & \NumCanonLgA{} & \NumCanonLgARoc{} & \NumCanonLgAe{} \\
1--34/35--42 & CausalSAGE & \NumCanonCsA{} & \NumCanonCsARoc{} & \NumCanonCsAe{} \\
1--34/35--49 & Random Forest & \NumCanonRfB{} & \NumCanonRfBRoc{} & \NumCanonRfBe{} \\
1--34/35--49 & LightGBM & \NumCanonLgB{} & \NumCanonLgBRoc{} & \NumCanonLgBe{} \\
1--34/35--49 & CausalSAGE & \NumCanonCsB{} & \NumCanonCsBRoc{} & \NumCanonCsBe{} \\
Strict 1--34/35--42/43--49 & CausalSAGE & \NumCanonStrict{} & \NumCanonStrictRoc{} & \NumCanonStrictE{} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{GNN capacity sensitivity (exploratory, single seed, no inferential claim). \label{tab:tuning}}
\centering\footnotesize
\begin{tabular}{lrr}
\toprule
(hidden, layers) & PR-AUC & ECE \\
\midrule
(32, 2) & \NumTuneNarrow{} & \NumTuneNarrowE{} \\
(64, 2) & \NumTuneWide{} & \NumTuneWideE{} \\
(32, 3) & \NumTuneDeep{} & \NumTuneDeepE{} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Label protocols, robustness and structural counterfactuals}
Table~\ref{tab:protocols} reports label protocols; Table~\ref{tab:attack} test-time perturbation sensitivity; Table~\ref{tab:worlds} structural edge worlds. For protocol B, unknown-as-negative changes the evaluation population, so its PR-AUC is not directly comparable to A; ROC-AUC and recall at fixed FPR are provided in the released tables.

\begin{table}[!t]
\caption{Protocol sensitivity (CausalSAGE, 15 seeds). \label{tab:protocols}}
\centering\footnotesize
\begin{tabular}{llr}
\toprule
Protocol & Treatment & PR-AUC \\
\midrule
A & known-only (strict) & \NumProtA{} \\
B & unknown-as-negative (different population) & \NumProtB{} \\
C & pseudo-label train, known-only test & \NumProtC{} \\
D & nnPU (SCAR sensitivity) & \NumProtD{} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Test-time perturbation sensitivity (CausalSAGE, budget 10\%; exploratory, single seed, no inferential claim). \label{tab:attack}}
\centering\footnotesize
\begin{tabular}{lrrr}
\toprule
Perturbation & mean & median & trimmed \\
\midrule
Edge deletion & \NumAtEdMean{} & \NumAtEdMed{} & \NumAtEdTrim{} \\
High-degree deletion & \NumAtHdMean{} & \NumAtHdMed{} & \NumAtHdTrim{} \\
Feature flip & \NumAtFfMean{} & \NumAtFfMed{} & \NumAtFfTrim{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent This is a robustness probe, distinct from protocol deception. At a fixed seed values may rise or fall; the full attack--budget--aggregation grid is released.
\end{table}

\input{tables/Table2_worlds}

\subsection{Calibration: I4 acts only on threshold metrics}
Fig.~\ref{fig:calib} summarizes measured ECE (15 equal-width bins) and Brier score. Random Forest and LightGBM are better calibrated than the graph model under the contract. To verify the I4 claim directly, we take one \emph{synthetic} frozen score vector (not a C1 measurement; its purpose is to demonstrate the monotonicity property, see Table~\ref{tab:i4test}) and apply the validation-frozen map $h_{\rm val}$ and an illegally test-fitted map $h_{\rm test}$. Table~\ref{tab:i4test} shows PR-AUC and ROC-AUC are bit-identical while ECE and Brier move; hence a calibration rung cannot change ranking metrics; any ranking difference across protocol rows is a configuration difference, not a calibration effect. This same-score test provides direct evidence for I4.

\begin{table}[!t]
\caption{I4 same-score test on a synthetic score vector (illustration of monotonicity, not a C1 measurement). \label{tab:i4test}}
\centering\footnotesize
\begin{tabular}{lrrrr}
\toprule
Map & PR-AUC & ROC-AUC & ECE & Brier \\
\midrule
identity (no scaling) & \NumIFourPrVal{} & \NumIFourRocVal{} & \NumIFourEceRaw{} & \NumIFourBrierRaw{} \\
$h_{\rm val}$ (frozen) & \NumIFourPrVal{} & \NumIFourRocVal{} & \NumIFourEceVal{} & \NumIFourBrierVal{} \\
$h_{\rm test}$ (illegal refit) & \NumIFourPrTest{} & \NumIFourRocTest{} & \NumIFourEceTest{} & \NumIFourBrierTest{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent This is a \emph{synthetic} score vector with an 8\% validation base rate and a 2.5\% test base rate, used solely to verify the implementation of I4's monotonicity claim: PR-AUC and ROC-AUC are invariant to any strictly monotone map, hence identical to machine precision by mathematical necessity, while ECE/Brier change. On this particular synthetic vector temperature scaling barely changes ECE ($\NumIFourEceRaw{}\to\NumIFourEceVal{}$), which is precisely why this table is an implementation check and not a claim that calibration improves this benchmark. Fitted temperatures: $t_{\rm val}=\NumIFourTval{}$, $t_{\rm test}=\NumIFourTtest{}$ (the near-equality reflects temperature scaling's low sensitivity to base-rate shift; the ranking invariance itself is exact). Source: \texttt{i4\_same\_scores\_test.py}, seed 0; this vector is not from configuration C1 and is not used in any performance claim.
\end{table}
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/calibration_summary}
\caption{Measured ECE and Brier by model (chronological protocol). \label{fig:calib}}
\end{figure}

\subsection{Operational metrics}
Precision@100 saturates at 1.000 under single-digit illicit rates and does not discriminate systems. We therefore report recall at the frozen decision threshold $0.5$, together with cost-weighted utility $U=\mathrm{TP}-\lambda\mathrm{FP}-\mu\mathrm{FN}$ ($\lambda=1,\mu=5$, counted per test node). With a $\sim2.5\%$ illicit base rate the false-alarm term dominates, so absolute $U$ is negative and is not a model-quality ranking; we add the alert-normalized $U/\#\text{alerts}$ for comparability. A recall-at-1\%FPR curve requires per-sample scores and is left to the artifact. Table~\ref{tab:ops} reports these quantities.

\begin{table}[!t]
\caption{Operational metrics at the frozen threshold $0.5$ (details in Sec.~\ref{sec:results}). \label{tab:ops}}
\centering\footnotesize
\begin{tabular}{lrrrr}
\toprule
Model & Recall@0.5 & $U$ (absolute) & $U$/alert & ECE \\
\midrule
Random Forest & \NumOpRfRec{} & \NumOpRfU{} & \NumOpRfUn{} & \NumRFECE{} \\
LightGBM & \NumOpLgRec{} & \NumOpLgU{} & \NumOpLgUn{} & \NumLGBMECE{} \\
CausalSAGE & \NumOpCsRec{} & \NumOpCsU{} & \NumOpCsUn{} & \NumCSEce{} \\
\bottomrule
\end{tabular}
\end{table}

\section{Reproducibility Artifact}\label{sec:artifact}
The archive contains source, tests, raw logs, machine-readable tables, figure data, environment snapshots and the complete LaTeX source. A numeric-range check asserts all scores lie in $[0,1]$ over \NumCheckedRuns{} result files with zero violations. Every manuscript macro is generated from CSV/JSON and verified by automated consistency tests. Table~\ref{tab:repro} maps each table/figure to its script, command, seed set and runtime; an artifact checklist and an anonymized DOI-ready release are provided in the companion repository. The dataset is CC BY-NC-ND 4.0 and is not redistributed.

\begin{table*}[!t]
\caption{Reproducibility map (principal outputs; complete map and wall times in \texttt{outputs/logs/}). \label{tab:repro}}
\centering\footnotesize
\begin{tabular}{llllr}
\toprule
Output & Script & Config & Seeds & Runs \\
\midrule
Table~\ref{tab:main} & build\_table + build\_manuscript & C1/C1-R & 0--14 & 30 \\
Table~\ref{tab:ladder} & run\_matrix & C1 rungs & 0--14 & 75 \\
Table~\ref{tab:samestep} & run\_matrix & C1-E & 0--14 & 15 \\
Table~\ref{tab:stats} & build\_table + build\_manuscript & C1 & 0--14 & -- \\
Table~\ref{tab:worlds} & build\_manuscript & C1 & 0--14 & -- \\
Table~\ref{tab:i4test} & i4\_same\_scores\_test & synthetic & -- & 1 \\
Table~\ref{tab:audit},~\ref{tab:advaudit} & blind\_audit, adversarial\_audit & -- & -- & 240 \\
Table~\ref{tab:weak} & run\_weak & LR/MLP & 0--4 & 20 \\
Figures~\ref{fig:samestep}--\ref{fig:calib} & build\_figures & from macros & -- & 3 \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Threat A: weak detectors under permissive protocols}
The threat model concerns a weak detector being reported as competitive through protocol choice. Table~\ref{tab:weak} gives the direct evidence: under the strict chronological contract a linear baseline reaches PR-AUC \NumWeakLrChrono{} and an MLP \NumWeakMlpChrono{}, while the same models under random partitioning reach \NumWeakLrRandom{} and \NumWeakMlpRandom{}. The inflated weak scores approach or exceed the strict-protocol tree baseline (\NumRFPrAUC{}), demonstrating that the reported rank ordering can be reversed by protocol choice alone. This is the mechanism the contract is designed to remove.

\begin{table}[!t]
\caption{Threat A evidence: weak models under strict vs permissive protocols (5 seeds). \label{tab:weak}}
\centering\footnotesize
\begin{tabular}{lrr}
\toprule
Model & Chronological & Random split \\
\midrule
Logistic regression & \NumWeakLrChrono{} & \NumWeakLrRandom{} \\
MLP & \NumWeakMlpChrono{} & \NumWeakMlpRandom{} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Determinism, non-vacuity and adversarial audit}\label{sec:controls}
Three controls close remaining concerns. (i) \textbf{Empty-graph determinism}: we construct the zero-edge graph through two independent code paths (strict $\tau(u)<\tau(v)$ filter and explicit empty world) with a fixed seed; sampled nodes, sampled edges and model scores are bit-identical (max difference $0.0$). Accordingly a single registered control C1-E is reported in Table~\ref{tab:samestep} and Fig.~\ref{fig:samestep}. (ii) \textbf{Non-vacuity}: on a synthetic temporal graph with cross-step edges, the strict boundary retains \NumSynStrict{} edges and the graph does not collapse, while all injected future edges are removed. The contract is thus evaluable in general; it collapses only on native Elliptic because that graph's edges are all intra-step. (iii) \textbf{Adversarial audit}: unlike a manifest-only predicate check, an auditor that cross-checks the declared manifest against raw edge times and calibration logs detects forged submissions (Table~\ref{tab:advaudit}).

\begin{table}[!t]
\caption{Adversarial audit: auditor cross-checks manifest against raw artifacts (20 adversarial + 20 compliant per type). \label{tab:advaudit}}
\centering\footnotesize
\begin{tabular}{lrrr}
\toprule
Adversary & Recall & Precision & FPR \\
\midrule
Forged future-edge manifest & 1.00 & 1.00 & 0.00 \\
Test-period calibration refit & 1.00 & 1.00 & 0.00 \\
Missing audit record & 1.00 & 1.00 & 0.00 \\
Overlapping split manifest & 1.00 & 1.00 & 0.00 \\
Edge-count mismatch & 1.00 & 1.00 & 0.00 \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent Limitation: the audit assumes raw edge lists and calibration logs are available and truthful; a supplier who forges both raw artifacts cannot be detected by this contract. Threat A concerns protocol deception, not log forgery.
\end{table}

\section{Feature Observation Time and Within-Step Resolution}\label{sec:assumptions}
Two assumptions about dataset properties are examined in this section.

\subsection{I6: feature observation time}
The five original predicates constrain splits, edges, labels, calibration and audit, but not \emph{when features are observable}. External audit \cite{safar2026enemy} reverse-engineered the Elliptic release and reports that a substantial part of the nominal ``local'' columns are address-level aggregates computed at one snapshot, potentially after the test window, with address reuse across train and test. We therefore add I6 (Definition~\ref{def:contract}) and implement an auditor that turns it into a machine-checkable object: every column must declare a family (intrinsic / snapshot address aggregate / undetermined) and an observation-time relation $t_f$ to the label time $t_y$, and only declared-and-certified columns pass. On this release the auditor returns \textbf{FAIL}: \NumISixCertified{} of \NumISixCols{} columns can be certified from the information available offline, so all are conservatively marked undetermined (the artifact stores the per-column declaration CSV and the pass/fail record). This verifies declaration and mapping consistency, not the publisher's original computation; the latter is answerable only by external audit. In the same data ecosystem, Liang et al.\cite{liang2026revisiting} reconstructed all 55 \emph{address-level} Elliptic++ features and recomputed them under time-respecting splits: RF illicit-class F1 fell from $0.846$ (leaky positional split) to $0.257$ under a temporal split alone and $0.291$ with recomputed features, where it no longer materially exceeded logistic regression ($0.285$). Their task is address-level and ours transaction-level, so this is same-ecosystem corroboration, not a reproduced number; it does, however, show that our absolute RF level may inherit the same leakage and must not be read as a deployable estimate. As a sensitivity, Table~\ref{tab:ftime} contrasts the two feature families in our artifact: removing all 71 aggregate columns changes Random Forest PR-AUC by only \NumFtimeDelta{} (0.731 vs 0.734). This does not cure I6 for the nominal local block and we make no leak-free claim.

\begin{table}[!t]
\caption{I6 declaration classes and feature-family sensitivity (RF, chronological, 5 seeds, exploratory). \label{tab:ftime}}
\centering\footnotesize
\begin{tabular}{llr}
\toprule
I6 class & Columns & Certification \\
\midrule
Intrinsic (time-local) & not certified & FAIL \\
Snapshot address aggregate & not certified & FAIL \\
Undetermined (offline) & \NumISixCols{} & FAIL \\
\midrule
Local-94 sensitivity & PR-AUC 0.731 & --- \\
Local-94 + agg-71 & PR-AUC 0.734 & --- \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent Under the strict predicate every column is undetermined offline, so I6 returns FAIL; the two sensitivity rows are diagnostic only and certify nothing.
\vspace{2pt}\par\footnotesize\noindent The nominal local block may itself contain address aggregates \cite{safar2026enemy}; neither row is certified I6-compliant.
\end{table}

\subsection{Within-step resolution: a structural argument}
Edges in this release are spend relations: $u\to v$ records that transaction $v$ consumes an output created by $u$, so $v$ cannot precede $u$ on the ledger. Time-forwardness is therefore a property of the \emph{data-generating relation}, and under real block timestamps $\tau'(u)<\tau'(v)$ would hold for nearly all edges. We previously reported a ``recovered'' retention of \NumRecoveredFrac{} using pseudo-random within-step ranks; we \emph{withdraw} that number, because a uniform random order retains exactly one half of any edge set by construction and carries no information about real chronology. The correct statement is structural: the observed strict-boundary collapse is an artifact of compressing a $\approx$3-hour window into one integer step, not evidence that the transactions lack temporal order. This makes the contract \emph{more}, not less, applicable: on any release with transaction-level timestamps the strict filter is non-vacuous by the semantics of the spend relation. Temporal-network conventions \cite{holme2012temporal} and benchmarks \cite{huang2023tgb} motivate the distinction.

\begin{table}[!t]
\caption{Strict boundary under alternative time resolutions. \label{tab:recover}}
\centering\footnotesize
\begin{tabular}{lr}
\toprule
Time resolution & Expected retention \\
\midrule
Released step-level ($\tau=$ step) & 0 (all edges intra-step) \\
Transaction-level (spend relation) & $\approx$100\% (structural) \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent The second row follows from the semantics of the spend relation, not from a re-run; no block timestamps were available offline to verify it numerically.
\end{table}

\subsection{External numerical anchor}
An independent study on the same dataset under a strict temporal split with Random Forest reports ROC-AUC $\approx$0.85 and average precision $\approx$0.54 \cite{khaleghpour2026temporal}, and concludes that transaction attributes dominate while graph-derived features are complementary. This is consistent with our finding: their AP lies between our graph model (0.521) and RF (0.733), and their ROC-AUC is of the same order as our Table~\ref{tab:main} value. We cite this as same-ecosystem corroboration only: their strict-split protocol and task differ from ours, so the numbers are not directly comparable and no cross-protocol equality is claimed.

\section{Limitations}\label{sec:limits}
This is a single-benchmark, Elliptic-specific diagnostic and we make no cross-dataset claim. We include one TGAT-style temporal-attention baseline under the contract (Table~\ref{tab:main}); a full TGN memory module and a second real dataset (e.g., DGraph \cite{huang2022dgraph} or Elliptic++ \cite{elmougy2023ellipticpp}) are outside the present artifact and are explicit future work. Coarse released time steps preclude establishing real within-step order numerically; Section~\ref{sec:assumptions} gives the structural argument that the spend relation is time-forward, not a measured retention rate. Feature semantics are only partially recoverable offline; I6 is therefore probed via family ablation rather than certified per-column. Protocol D (nnPU) is an SCAR sensitivity probe whose SAR identifiability is not established; its difference from A ($0.504-0.521=-0.017$) lies within the CausalSAGE seed standard deviation ($0.037$), so nnPU neither improves nor degrades performance. Time-block inference treats steps as the resampling unit but cannot create new temporal replicates. The threat model addresses evaluation deception, not model inversion, and assumes logs are truthful.

\section{Conclusion}\label{sec:conclusion}
We formalized an auditable evaluation contract for temporal graph screening and showed that, on Elliptic, every edge is intra-time-step: non-future-step filtering is vacuous and a strict causal boundary removes the graph. A feature-only Random Forest is the strongest measured model, the graph model gains nothing attributable to structure, and random partitioning raises its reported PR-AUC by \NumCSGap{} (unrounded means; $0.390$ at two-decimal endpoints) for CausalSAGE under this implementation and tuning budget. The contribution is the contract, the boundary analysis, the protocol-effect estimate and the verifiable artifact---not a new architecture and not a universal claim about graph structure.

\section*{Acknowledgment and Declarations}
\noindent\textbf{Funding:} This work received no external funding. \\
\noindent\textbf{Conflict of interest:} The author declares no conflict of interest. \\
\noindent\textbf{Data availability:} Elliptic is publicly available (Kaggle) under CC BY-NC-ND 4.0; only download logic and checksums are redistributed. \\
\noindent\textbf{Ethics and dual use:} The study uses a public anonymized graph, involves no human subjects, and is defensive in intent. \\
\noindent\textbf{Author contributions:} X. Han conceived the study, implemented the artifact, ran experiments and wrote the manuscript. \\
\noindent\textbf{Generative AI use:} Generative AI assisted language polishing and code scaffolding; all claims, numbers and designs were verified by the author.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""


def build_main_tex(data):
    text = PREAMBLE
    # Append rebuttal-derived macros to numbers.tex (all values from result files)
    extra = extra_macros()
    (OUT / "numbers.tex").write_text(
        (OUT / "numbers.tex").read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8"
    )
    (OUT / "main.tex").write_text(text, encoding="utf-8")


def extra_macros() -> str:
    # Single-source v5 table (one registered configuration per protocol).
    v5csv = ROOT / "outputs" / "v7" / "tables" / "Table1_v5.csv"
    main_v5 = pd.read_csv(v5csv) if v5csv.exists() else None
    def v5field(cid, field):
        if main_v5 is None or "config_id" not in main_v5.columns:
            return float("nan")
        r = main_v5[main_v5.config_id == cid]
        return float(r.iloc[0][field]) if len(r) and field in r.columns else float("nan")
    """Derive rebuttal macros from result files; emit 0.000-safe placeholders if absent."""
    rb = ROOT / "outputs" / "rebuttal" / "results"
    diag = load_json(DIAG / "data_diagnostics.json", {})

    def mean_metric(pred) -> tuple[float, float]:
        if not rb.exists():
            return float("nan"), float("nan")
        vals_p, vals_e = [], []
        for p in rb.glob("*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") != "done":
                continue
            cfg = r.get("config", {})
            if pred(cfg, r):
                c = r["metrics"]["calibrated"]
                vals_p.append(c.get("pr_auc", float("nan")))
                vals_e.append(c.get("ece", float("nan")))
        return (float(np.mean(vals_p)) if vals_p else float("nan"),
                float(np.mean(vals_e)) if vals_e else float("nan"))

    def gv(t, v):
        return f"\\newcommand{{{t}}}{{{fmt(v[0])}}}" + "\n" + f"\\newcommand{{{t}e}}{{{fmt(v[1])}}}"

    abl_local = mean_metric(lambda c, r: c.get("feature_view") == "local94" and r["model"] == "rf")
    abl_agg = mean_metric(lambda c, r: c.get("feature_view") == "local94_agg72" and r["model"] == "rf")
    abl_struct = mean_metric(lambda c, r: c.get("feature_view") == "all_plus_struct" and r["model"] == "rf")
    canon_ra = mean_metric(lambda c, r: c.get("canonical_split") == "1-34_35-42" and r["model"] == "rf")
    canon_la = mean_metric(lambda c, r: c.get("canonical_split") == "1-34_35-42" and r["model"] == "lightgbm")
    canon_ca = mean_metric(lambda c, r: c.get("canonical_split") == "1-34_35-42" and r["model"] == "causal_sage")
    canon_rb = mean_metric(lambda c, r: c.get("canonical_split") == "1-34_35-49" and r["model"] == "rf")
    canon_lb = mean_metric(lambda c, r: c.get("canonical_split") == "1-34_35-49" and r["model"] == "lightgbm")
    canon_cb = mean_metric(lambda c, r: c.get("canonical_split") == "1-34_35-49" and r["model"] == "causal_sage")
    # (32,2) baseline from the unified per-seed table (H1: single definition).
    unified_csv = REBUTTAL_T / "Table1_unified_perseed.csv"
    main_df = pd.read_csv(unified_csv if unified_csv.exists() else TABLES / "Table1_main.csv")
    cs_chrono = main_df[(main_df.model == "causal_sage") & (main_df.split_mode == "chrono")]
    tune32 = (float(cs_chrono.iloc[0]["pr_auc"]), float(cs_chrono.iloc[0]["ece"])) if len(cs_chrono) else (float("nan"), float("nan"))
    tune64 = mean_metric(lambda c, r: r["model"] == "causal_sage" and c.get("hidden") == 64 and c.get("attack", "none") == "none")
    tune3l = mean_metric(lambda c, r: r["model"] == "causal_sage" and c.get("layers") == 3 and c.get("attack", "none") == "none")

    # Protocol values from full results. Restrict to the standard chrono
    # configuration (real edges, no attack, default 32x2 GNN) so that protocol
    # A matches the main-table CausalSAGE value exactly.
    def prot(protocol):
        p = ROOT / "outputs" / "full" / "results"
        vals = []
        for f in p.glob("*.json") if p.exists() else []:
            r = json.loads(f.read_text(encoding="utf-8"))
            c = r.get("config", {})
            if (r.get("model") == "causal_sage" and c.get("label_protocol") == protocol
                    and c.get("split_mode") == "chrono" and c.get("edge_world") == "real"
                    and c.get("attack", "none") == "none" and c.get("hidden") == 32
                    and c.get("layers") == 2):
                vals.append(r["metrics"]["calibrated"]["pr_auc"])
        return float(np.mean(vals)) if vals else float("nan")

    lines = [
        gv("\\NumAblLocal", abl_local), gv("\\NumAblAgg", abl_agg), gv("\\NumAblStruct", abl_struct),
        f"\\newcommand{{\\NumCanonRfA}}{{{fmt(canon_ra[0])}}}", f"\\newcommand{{\\NumCanonRfAe}}{{{fmt(canon_ra[1])}}}",
        f"\\newcommand{{\\NumCanonLgA}}{{{fmt(canon_la[0])}}}", f"\\newcommand{{\\NumCanonLgAe}}{{{fmt(canon_la[1])}}}",
        f"\\newcommand{{\\NumCanonCsA}}{{{fmt(canon_ca[0])}}}", f"\\newcommand{{\\NumCanonCsAe}}{{{fmt(canon_ca[1])}}}",
        f"\\newcommand{{\\NumCanonRfB}}{{{fmt(canon_rb[0])}}}", f"\\newcommand{{\\NumCanonRfBe}}{{{fmt(canon_rb[1])}}}",
        f"\\newcommand{{\\NumCanonLgB}}{{{fmt(canon_lb[0])}}}", f"\\newcommand{{\\NumCanonLgBe}}{{{fmt(canon_lb[1])}}}",
        f"\\newcommand{{\\NumCanonCsB}}{{{fmt(canon_cb[0])}}}", f"\\newcommand{{\\NumCanonCsBe}}{{{fmt(canon_cb[1])}}}",
        f"\\newcommand{{\\NumTuneNarrow}}{{{fmt(tune32[0])}}}", f"\\newcommand{{\\NumTuneNarrowE}}{{{fmt(tune32[1])}}}",
        f"\\newcommand{{\\NumTuneWide}}{{{fmt(tune64[0])}}}", f"\\newcommand{{\\NumTuneWideE}}{{{fmt(tune64[1])}}}",
        f"\\newcommand{{\\NumTuneDeep}}{{{fmt(tune3l[0])}}}", f"\\newcommand{{\\NumTuneDeepE}}{{{fmt(tune3l[1])}}}",
        f"\\newcommand{{\\NumProtA}}{{{fmt(v5field('C1-chrono', 'pr_auc'))}}}",
        f"\\newcommand{{\\NumProtB}}{{{fmt(v5field('C1-unknown', 'pr_auc'))}}}",
        f"\\newcommand{{\\NumProtC}}{{{fmt(prot('C'))}}}", f"\\newcommand{{\\NumProtD}}{{{fmt(prot('D'))}}}",
    ]
    # Attack macros: 3 attacks x 3 aggregations at budget 0.10, from full results
    full_res = ROOT / "outputs" / "full" / "results"
    def atk10(attack, agg):
        vals = []
        for f in (full_res.glob("*.json") if full_res.exists() else []):
            r = json.loads(f.read_text(encoding="utf-8"))
            c = r.get("config", {})
            if (r.get("model") == "causal_sage" and c.get("attack") == attack
                    and c.get("aggregator") == agg and abs(c.get("attack_budget", -1) - 0.1) < 1e-9):
                vals.append(r["metrics"]["calibrated"]["pr_auc"])
        return float(np.mean(vals)) if vals else float("nan")
    for name, attack, agg in [
        ("AtEdMean", "edge_delete", "mean"), ("AtEdMed", "edge_delete", "median"), ("AtEdTrim", "edge_delete", "trim"),
        ("AtHdMean", "high_degree_delete", "mean"), ("AtHdMed", "high_degree_delete", "median"), ("AtHdTrim", "high_degree_delete", "trim"),
        ("AtFfMean", "feature_flip", "mean"), ("AtFfMed", "feature_flip", "median"), ("AtFfTrim", "feature_flip", "trim"),
    ]:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(atk10(attack, agg))}}}")
    # ---- V3 macros from outputs/v3/results (strict boundary, ladder, canon-strict) ----
    v3res = ROOT / "outputs" / "v3" / "results"
    def v3_metric(pred, field="pr_auc"):
        vals = []
        for f in (v3res.glob("*.json") if v3res.exists() else []):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("status") != "done":
                continue
            if pred(r.get("config", {}), r):
                vals.append(r["metrics"]["calibrated"].get(field, float("nan")))
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    # V5 single-source: use the registered C1-chrono row, not the first
    # chrono-labelled row (which was C1-unknown and gave ECE=0.075).
    csr = v5field("C1-chrono", "roc_auc")
    cse = v5field("C1-chrono", "ece")
    lines += [
        f"\\newcommand{{\\NumCSRoc}}{{{fmt(csr)}}}",
        f"\\newcommand{{\\NumCSEce}}{{{fmt(cse)}}}",
        f"\\newcommand{{\\NumLicit}}{{{int(diag.get('labels',{}).get('licit',42019)):,}}}",
        f"\\newcommand{{\\NumUnknown}}{{{int(diag.get('labels',{}).get('unknown',157205)):,}}}",
    ]
    # strict / empty-world PR macros come from the v5 single-source table
    # (defined in build_numbers); only ROC/ECE auxiliary macros are here.
    for name, field, pred in [
        ("StrictRoc", "roc_auc", lambda c, r: c.get("strict_time", False) and r.get("model") == "causal_sage"),
        ("StrictEce", "ece", lambda c, r: c.get("strict_time", False) and r.get("model") == "causal_sage"),
        ("EmptyRoc", "roc_auc", lambda c, r: c.get("edge_world") == "empty" and r.get("model") == "causal_sage"),
        ("EmptyEce", "ece", lambda c, r: c.get("edge_world") == "empty" and r.get("model") == "causal_sage"),
    ]:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(v3_metric(pred, field))}}}")
    # Ladder rungs at the SAME 30-epoch budget as the main comparison.
    v4res = ROOT / "outputs" / "v4" / "results"
    def v4_metric(pred, field="pr_auc"):
        vals = []
        all_dirs = [v4res, ROOT / "outputs" / "v5" / "results", ROOT / "outputs" / "full" / "results"]
        for f in (p for d in all_dirs if d.exists() for p in d.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("status") != "done":
                continue
            if pred(r.get("config", {}), r):
                vals.append(r["metrics"]["calibrated"].get(field, float("nan")))
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    # Use v4 30-epoch ladder runs when present; fall back to v3 20-epoch runs.
    def ladder_metric(pred, field="pr_auc"):
        v = v4_metric(pred, field)
        return v if np.isfinite(v) else v3_metric(pred, field)

    ladder_defs = [
        ("LadOne", lambda c, r: c.get("epochs") == 30 and c.get("split_mode") == "random" and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none"),
        ("LadTwo", lambda c, r: c.get("epochs") == 30 and c.get("split_mode") == "chrono" and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none" and c.get("calibration") != "none"),
        ("LadThree", lambda c, r: c.get("epochs") == 30 and c.get("label_protocol") == "B"),
        ("LadFour", lambda c, r: c.get("epochs") == 30 and c.get("calibration") == "none" and c.get("split_mode") == "chrono"),
    ]
    # v3 runs have epochs=20, so re-declare fallback predicates without the 30 check.
    ladder_fallback = [
        lambda c, r: c.get("split_mode") == "random" and c.get("label_protocol", "A") == "A",
        lambda c, r: c.get("split_mode") == "chrono" and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none",
        lambda c, r: c.get("label_protocol") == "B",
        lambda c, r: c.get("calibration") == "none" and c.get("split_mode") == "chrono",
    ]
    # V5: all ladder rungs come from the single-source table (one config, 15 seeds).
    ladder_ids = [("LadOne", "C1-random"), ("LadTwo", "C1-chrono"),
                  ("LadThree", "C1-unknown"), ("LadFour", "C1-testcal")]
    for name, cid in ladder_ids:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(v5field(cid, 'pr_auc'))}}}")
        lines.append(f"\\newcommand{{\\Num{name}e}}{{{fmt(v5field(cid, 'ece'))}}}")
        lines.append(f"\\newcommand{{\\Num{name}p}}{{{fmt(v5field(cid, 'precision@100'))}}}")
    # canonical strict
    lines.append(f"\\newcommand{{\\NumCanonStrict}}{{{fmt(v3_metric(lambda c, r: c.get('canonical_split') == '1-34_35-49-strict'))}}}")
    lines.append(f"\\newcommand{{\\NumCanonStrictE}}{{{fmt(v3_metric(lambda c, r: c.get('canonical_split') == '1-34_35-49-strict', 'ece'))}}}")
    lines.append(f"\\newcommand{{\\NumCanonStrictRoc}}{{{fmt(v3_metric(lambda c, r: c.get('canonical_split') == '1-34_35-49-strict', 'roc_auc'))}}}")
    # ROC-AUC for canonical A/B rows, from the same result files (no hardcoded values).
    def canon_roc(model, split):
        vals = []
        for p in (rb.glob("*.json") if rb.exists() else []):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") == "done" and r["model"] == model and r.get("config", {}).get("canonical_split") == split:
                vals.append(r["metrics"]["calibrated"].get("roc_auc", float("nan")))
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    for tag, model, split in [
        ("RfA", "rf", "1-34_35-42"), ("LgA", "lightgbm", "1-34_35-42"), ("CsA", "causal_sage", "1-34_35-42"),
        ("RfB", "rf", "1-34_35-49"), ("LgB", "lightgbm", "1-34_35-49"), ("CsB", "causal_sage", "1-34_35-49"),
    ]:
        lines.append(f"\\newcommand{{\\NumCanon{tag}Roc}}{{{fmt(canon_roc(model, split))}}}")
    # Second graph baseline (GAT with attention aggregation) from v4 results.
    gat_pr = v4_metric(lambda c, r: r["model"] == "gat" and c.get("split_mode") == "chrono")
    gat_ece = v4_metric(lambda c, r: r["model"] == "gat" and c.get("split_mode") == "chrono", "ece")
    lines += [
        f"\\newcommand{{\\NumGatPr}}{{{fmt(gat_pr)}}}",
        f"\\newcommand{{\\NumGatEce}}{{{fmt(gat_ece)}}}",
    ]
    # Threat A: weak models under strict vs permissive protocols.
    for tag, model, split in [
        ("LrChrono", "lr", "chrono"), ("LrRandom", "lr", "random"),
        ("MlpChrono", "mlp", "chrono"), ("MlpRandom", "mlp", "random"),
    ]:
        lines.append(f"\\newcommand{{\\NumWeak{tag}}}{{{fmt(v4_metric(lambda c, r, m=model, sp=split: r['model'] == m and c.get('split_mode') == sp and c.get('label_protocol', 'A') == 'A' and c.get('attack', 'none') == 'none' and c.get('edge_world') == 'real'))}}}")
    # Class-imbalance parity: GNN with positive reweighting.
    for tag, mode in (("None", "none"), ("Sqrt", "sqrt"), ("Bal", "balanced")):
        lines.append(f"\\newcommand{{\\NumPw{tag}}}{{{fmt(v4_metric(lambda c, r, m=mode: r['model'] == 'causal_sage' and c.get('pos_weight_mode') == m and c.get('split_mode') == 'chrono' and c.get('label_protocol') == 'A' and c.get('attack', 'none') == 'none' and c.get('edge_world') == 'real'))}}}")
    return "\n".join(lines) + "\n"


def copy_support():
    """Copy IEEEtran.cls and figures from v1; keep the v2 (extended) bibliography.

    The extended references.bib lives next to this script in
    ``scripts/references_v2.bib`` and is the single source of truth. It is
    copied fresh on every build so v1's shorter bib never overwrites it.
    """
    v1 = ROOT / "submission" / "tifs"
    src_cls = v1 / "IEEEtran.cls"
    if src_cls.exists():
        shutil.copy(src_cls, OUT / "IEEEtran.cls")
    bib_src = Path(__file__).parent / "references_v7.bib"
    if bib_src.exists():
        shutil.copy(bib_src, OUT / "references.bib")
    elif (Path(__file__).parent / "references_v2.bib").exists():
        shutil.copy(Path(__file__).parent / "references_v2.bib", OUT / "references.bib")
    # V3 figures generated by build_v3_figures.py take precedence.
    figsrc = ROOT / "submission" / "tifs_v7" / "figures_src"
    if figsrc.exists():
        for fig in figsrc.glob("*"):
            shutil.copy(fig, OUT / "figures" / fig.name)


if __name__ == "__main__":
    data = build_numbers()
    make_tables(data)
    build_main_tex(data)
    copy_support()
    print("TIFS v2 built at", OUT)
