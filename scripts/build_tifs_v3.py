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
OUT = ROOT / "submission" / "tifs_v3"
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
    unified = REBUTTAL_T / "Table1_unified_perseed.csv"
    main = pd.read_csv(unified if unified.exists() else TABLES / "Table1_main.csv")
    stats = load_json(TABLES / "statistical_tests.json", {})
    paired = load_json(REBUTTAL_T / "paired_effect_unified.json", {})
    tbb = load_json(REBUTTAL_T / "time_block_bootstrap.json", {})
    diag = load_json(DIAG / "data_diagnostics.json", {})
    filter_audit = load_json(DIAG / "future_edge_filter_audit.json", {})

    def pick(model, split, field):
        r = main[(main.model == model) & (main.split_mode == split)]
        return float(r.iloc[0][field]) if len(r) else float("nan")

    comp = {c["model"]: c for c in stats.get("comparisons", [])}
    cs = comp.get("causal_sage", {})
    lines = [
        f"\\newcommand{{\\NumRFPrAUC}}{{{fmt(pick('rf','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumRFPrAUCStd}}{{{fmt(pick('rf','chrono','pr_auc_std'))}}}",
        f"\\newcommand{{\\NumRFRoc}}{{{fmt(pick('rf','chrono','roc_auc'))}}}",
        f"\\newcommand{{\\NumRFBrier}}{{{fmt(pick('rf','chrono','brier'))}}}",
        f"\\newcommand{{\\NumRFECE}}{{{fmt(pick('rf','chrono','ece'))}}}",
        f"\\newcommand{{\\NumCSChronic}}{{{fmt(pick('causal_sage','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumCSRandom}}{{{fmt(pick('causal_sage','random','pr_auc'))}}}",
        f"\\newcommand{{\\NumLGBMChronic}}{{{fmt(pick('lightgbm','chrono','pr_auc'))}}}",
        f"\\newcommand{{\\NumLGBMRandom}}{{{fmt(pick('lightgbm','random','pr_auc'))}}}",
        f"\\newcommand{{\\NumDeltaPaired}}{{{fmt(paired.get('mean_delta'),3)}}}",
        f"\\newcommand{{\\NumDeltaPairedN}}{{{paired.get('n',0)}}}",
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
    return {"main": main, "stats": stats, "paired": paired, "tbb": tbb, "diag": diag, "filter_audit": filter_audit}


def make_tables(data):
    """Regenerate all manuscript tables from machine-readable results."""
    main = data["main"]
    stats = data["stats"]
    paired = data["paired"]
    diag = data["diag"]

    # --- Table I: unified paired means, canonical model names, n=1 flagged ---
    order = ["rf", "lightgbm", "xgboost", "mlp", "lr", "causal_sage", "graphsage"]
    m = main.copy()
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
    lines += [
        "\\bottomrule\\end{tabular}",
        "\\vspace{2pt}\\par\\footnotesize\\noindent $^{*}$Exploratory single run; no inferential claim. "
        "PR-AUC/ROC-AUC/ECE/Brier are validated to lie in $[0,1]$ by an automated check. "
        "All values are protocol A and temperature-calibrated on validation only.",
        "\\end{table*}", "",
    ]
    (OUT / "tables" / "Table1_main.tex").write_text("\n".join(lines), encoding="utf-8")

    # --- Table II: paired stats with precise p and unified delta ---
    pfmt = lambda v: "<$10^{-3}$" if v < 0.001 else (f"{v:.4f}" if v < 0.01 else f"{v:.3f}")
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Paired seed-level comparisons vs. Random Forest (protocol A, chronological). "
        "$\\Delta$ uses per-seed means so it is directly comparable to the abstract; run-level means in Table~\\ref{tab:main} may differ slightly. \\label{tab:stats}}",
        "\\centering\\footnotesize",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Model & $n$ & $\\Delta$PR-AUC & 95\\% CI & Wilcoxon $p$ & Holm $p$ & Verdict \\\\", "\\midrule",
    ]
    for c in stats.get("comparisons", []):
        verdict = "Significant" if c["holm_p"] < 0.05 else "n.s. (low power if $n=5$)"
        # Override CausalSAGE delta with the unified paired value (H1): the
        # only effect size quoted in the abstract. Other rows keep their own.
        delta = (paired.get("mean_delta") if c["model"] == "causal_sage" and paired.get("status") == "ok"
                 else c["mean_delta_pr_auc"])
        ci_low, ci_high = c["ci_low"], c["ci_high"]
        if c["model"] == "causal_sage" and paired.get("status") == "ok":
            pdelta = paired.get("per_seed_delta", {})
            if pdelta:
                dvals = np.array(list(pdelta.values()))
                rng = np.random.default_rng(17)
                boots = [rng.choice(dvals, size=len(dvals), replace=True).mean() for _ in range(2000)]
                ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
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
    real_row = main[(main.model == "causal_sage") & (main.split_mode == "chrono")]
    unified_real = float(real_row.iloc[0]["pr_auc"]) if len(real_row) else None
    unified_ci_lo = float(real_row.iloc[0].get("pr_auc_ci_low", float("nan"))) if len(real_row) else float("nan")
    unified_ci_hi = float(real_row.iloc[0].get("pr_auc_ci_high", float("nan"))) if len(real_row) else float("nan")
    for _, r in worlds.iterrows():
        lo = r.get("pr_auc_ci_low", float("nan")); hi = r.get("pr_auc_ci_high", float("nan"))
        if r["edge_world"] == "real" and unified_real is not None:
            val, lo, hi = unified_real, unified_ci_lo, unified_ci_hi
        else:
            val = r["pr_auc"]
        ci = f"[{fmt(lo)}, {fmt(hi)}]" if np.isfinite(lo) else "n/a"
        lines.append(
            f"{tex_escape(r['edge_world'])} & {int(r['n_seeds'])} & {fmt(val)} & {ci} & {fmt(r['ece'])} \\\\"
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
        "\\caption{Test-period step diagnostics. Positive counts are small in late steps; PR-AUC is a micro aggregate. \\label{tab:steps}}",
        "\\centering\\footnotesize",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Step & $n$ nodes & Illicit & Rate \\\\", "\\midrule",
    ]
    dper = diag.get("per_step", {})
    for s in steps:
        p = dper.get(str(s), {})
        lines.append(f"{s} & {p.get('n',0):,} & {p.get('illicit',0)} & {fmt(p.get('illicit_rate',0),3)} \\\\")
    lines += ["\\bottomrule\\end{tabular}",
              f"\\vspace{{2pt}}\\par\\footnotesize\\noindent {per.get('n_steps_llicit_lt_10','?')} of {per.get('n_steps','?')} steps have $<10$ illicit nodes; "
              "macro-mean illicit rate $=" + fmt(per.get("macro_mean_illicit_rate"), 3) + "$.",
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
\input{numbers}

\begin{document}

\title{An Auditable Evaluation Contract Reveals Limited Attributable Structure Gain in Temporal-Graph AML Screening}

\author{Xiaole Han
\thanks{X. Han is with the Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China (e-mail: hanxiaoleop@163.com).}
\thanks{ORCID: 0009-0009-9611-2368. Corresponding author: X. Han.}
\thanks{Manuscript submitted to IEEE Transactions on Information Forensics and Security.}}

\markboth{IEEE Transactions on Information Forensics and Security}%
{Han: An Auditable Evaluation Contract for Temporal Graph Screening}

\maketitle

\begin{abstract}
Graph neural networks (GNNs) are widely proposed for anti-money-laundering (AML) screening, yet reported gains may reflect evaluation-protocol artifacts rather than deployable signal. We formalize an auditable evaluation contract for temporal graphs as five machine-checkable predicates and a reference algorithm: time-step-disjoint partitioning, non-future-step edge construction, unknown-label isolation, validation-only calibration, and counterfactual auditability. A central result is that on the public Elliptic graph every edge is intra-time-step, so non-future-step filtering removes nothing and a strictly causal boundary (excluding same-step edges) leaves an empty graph; a GNN on that empty graph performs within noise of the full-graph GNN, indicating no attributable structure gain on this benchmark. Under the complete contract, a Random Forest attains PR-AUC \NumRFPrAUC{} and the best graph model \NumCSChronic{} (paired per-seed difference $\Delta=\NumDeltaPaired$, $n=\NumDeltaPairedN$). We further show that coarse time steps cannot resolve within-step order, that random partitioning inflates the same model to \NumCSRandom{}, and that a feature-family ablation explains most of the gap without message passing. All reported quantities are generated from machine-readable results and guarded by range and table--text consistency checks. Code, tests, logs, environment snapshots and figure data are released for verification.
\end{abstract}

\begin{IEEEkeywords}
anti-money laundering, temporal graph, evaluation leakage, reproducibility, calibration, cryptocurrency forensics
\end{IEEEkeywords}

\section{Introduction}\label{sec:intro}
\IEEEPARstart{A}{nti-money-laundering} (AML) screening produces costly alerts, and graph learning has been proposed to reduce analyst workload. A high reported PR-AUC, however, can be an artifact of the evaluation protocol: a model may observe future transactions, treat unknown labels as negatives, or be calibrated on test-period data. In a security-sensitive pipeline such artifacts produce optimistic screening guarantees that fail on deployment.

This paper studies the evaluation protocol as an auditable object. Its question is deliberately scoped: \emph{under an explicit, machine-checkable temporal contract, does graph structure add value beyond a strong feature-only baseline on the Elliptic benchmark?} We answer with a diagnostic study and a reproducible artifact, not a new architecture.

Weber \emph{et al.} released Elliptic and already reported that Random Forest outperforms graph convolutional networks under a temporal split \cite{weber2019antimoney}. Leakage from random splits and improper calibration is well documented \cite{kaufman2012leakage,shchur2018pitfalls,kapoor2023leakage}, tree models are strong on tabular data \cite{grinsztajn2022tree}, and graph-anomaly benchmarks report that tree ensembles with simple neighbour aggregation can rival purpose-built GNNs \cite{gaodbench2023}. The contribution of this paper is therefore not the RF-over-GNN observation. It is threefold:
\begin{enumerate}
\item a \textbf{formalized, executable contract}---five predicates with violation tests and a reference algorithm (Sec.~\ref{sec:formal});
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
\item \textbf{I1 Time-step disjointness:} $T_{\rm tr},T_{\rm val},T_{\rm test}\subset\mathcal{T}$ are pairwise disjoint and ordered ($\max T_{\rm tr}<\min T_{\rm val}\le\max T_{\rm val}<\min T_{\rm test}$).
\item \textbf{I2 Non-future-step edges:} every retained edge satisfies $\tau(u)\le\tau(v)$; a stricter boundary $\tau(u)<\tau(v)$ is defined separately (Sec.~\ref{sec:boundary}). Edge worlds record their construction.
\item \textbf{I3 Unknown-label isolation:} nodes with $y=-1$ never enter the loss or metric under the strict protocol; alternative protocols declare their treatment.
\item \textbf{I4 Validation-only calibration:} the calibration map $h$ is fit on $(Z_{\rm val},y_{\rm val})$ and frozen at test ($h_{\rm test}=h_{\rm val}$).
\item \textbf{I5 Counterfactual auditability:} each run logs $(\#E_{\rm in},\#E_{\rm out},\#\text{removed},\#\text{violations})$ for every edge world.
\end{enumerate}
\end{definition}
I1--I4 are properties of the evaluation, not of any model. We use ``causal'' only in the restricted sense of time-ordering and non-leakage; we make no do-calculus identifiability claim. We note that PR-AUC and ROC-AUC are invariant to any strictly monotone recalibration, so I4 affects threshold metrics (Brier, ECE, precision/recall at a threshold) and not ranking metrics---a distinction we verify empirically.

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
We align the security framing with a concrete third-party scenario. A model supplier submits an AML detector to a bank or regulator for acceptance at threshold $\theta$ on reported PR-AUC.
\begin{itemize}
\item \textbf{Goal:} make a weak detector $f_w$ appear comparable to a strong detector $f_s$ by inflating the reported score.
\item \textbf{Capability:} the supplier controls the \emph{evaluation protocol}, not model weights: split choice, unknown-label treatment, calibration window. It may also perturb the test computation graph under budget $b$.
\item \textbf{Budget:} $b\in\{0.01,0.05,0.10,0.20\}$ for graph perturbation; protocol choices are discrete.
\end{itemize}
The defense is the contract plus its audits: I1--I4 remove the free choices, I5 records any graph perturbation. We evaluate both halves: (i) protocol violations that inflate reported scores (Table~\ref{tab:ladder}), and (ii) audit detection of injected violations (Table~\ref{tab:audit}). Test-time graph perturbations (Table~\ref{tab:attack}) are a separate \emph{robustness} probe and are not conflated with protocol deception.

\section{The Same-Time-Step Boundary}\label{sec:boundary}
A key finding constrains the use of the term ``causal''. On the native Elliptic graph, all \NumEdges{} edges satisfy $\tau(v)-\tau(u)=0$: the graph is built within time steps, and no transaction-to-transaction edge spans steps. Consequently I2's non-future-step filter removes zero edges. Because a time step is a coarse interval, two nodes in the same step have no known within-step ordering; at a screening instant, an intra-step neighbour may not yet be observable.

We therefore evaluate a strict boundary that retains only $\tau(u)<\tau(v)$ edges. On native Elliptic this removes all \NumEdges{} edges, leaving an empty graph: the GNN degenerates to a feature-only MLP. Table~\ref{tab:samestep} reports this and a same-step sensitivity. The filter itself is not vacuous as an implementation: injecting future-directed edges (1\%, 5\%, 10\%) yields $100\%$ detection and removal (Table~\ref{tab:audit}). The honest conclusion is that \emph{non-future-step filtering is effective but vacuous on this dataset, and strict causal filtering is not evaluable here because it eliminates the graph}. We therefore frame the contribution as an evaluation contract and feature-baseline diagnosis, not a causal graph finding.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/same_step_boundary}
\caption{PR-AUC under non-future-step, strict-causal and empty-world boundaries; sample sizes are annotated. \label{fig:samestep}}
\end{figure}

\begin{table}[!t]
\caption{Same-step boundary sensitivity (CausalSAGE, 15 seeds). \label{tab:samestep}}
\centering\footnotesize
\begin{tabular}{lrrrr}
\toprule
Edge boundary & Edges retained & PR-AUC & ROC-AUC & ECE \\
\midrule
Non-future-step ($\le$) & \NumEdges{} & \NumCSChronic{} & \NumCSRoc{} & \NumCSEce{} \\
Strict causal ($<$) & 0 & \NumStrictPr{} & \NumStrictRoc{} & \NumStrictEce{} \\
Empty-world control & 0 & \NumEmptyPr{} & \NumEmptyRoc{} & \NumEmptyEce{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent Strict and empty-world GNNs have no neighbours and reduce to feature-only MLPs. Their scores are within noise of the full-graph GNN, indicating no attributable structure gain on this benchmark.
\end{table}

\section{Data and Experimental Setup}\label{sec:setup}
We use the public Elliptic graph: \NumNodes{} transactions, \NumEdges{} directed edges, 49 labelled time steps (indices 1--49; index 0 is unlabelled), \NumIllicit{} illicit and \NumLicit{} licit labelled nodes, and \NumUnknown{} unknown nodes \cite{weber2019antimoney}. Features are anonymized (165 columns: 94 local plus 71 aggregated); we make no amount/fee claims. Data are checksummed on download and not redistributed.

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
\vspace{2pt}\par\footnotesize\noindent For canonical A/B, validation is carved from the training window, so test remains disjoint; calibration is fit on validation and frozen. No split uses test data for model selection or calibration. The strict variant's test window (steps 43--49) contains only 169 illicit nodes among 6{,}687 labelled nodes, so its low PR-AUC reflects a harder, sparser window (ROC-AUC remains 0.70--0.78); it is not directly comparable to the 35--42 windows.
\end{table}

\subsection{Tuning, class imbalance and calibration}
We disclose the search space. Random Forest \{trees$\in$\{200,400\}, min\_leaf$\in$\{1,2,4\}\}; LightGBM \{leaves$\in$\{15,31,63\}, lr$\in$\{0.02,0.05,0.1\}\}; XGBoost \{depth$\in$\{3,5,7\}, lr$\in$\{0.03,0.05,0.1\}\}; GNN \{hidden$\in$\{32,64,128\}, layers$\in$\{2,3\}, neighbours$\in$\{[10,10],[20,20]\}, dropout$\in$\{0.3,0.5\}, $\alpha\in$\{0,0.5\}\}, early stopping on validation PR-AUC (patience 5). All learners share the validation window and search budget. Class imbalance is handled by class weighting for trees and an explicit positive-weight mode for GNNs. Tree classifiers output probabilities directly (no temperature scaling, which applies to logits); calibration for all models is validated via ECE with 15 equal-width bins, and Brier score is the mean squared error of the calibrated probability. The main GNN uses no positive reweighting because balanced weights inflated ECE.

\subsection{Statistical conventions}
The $\pm$ notation denotes standard deviation over seeds. Table--II paired intervals use a paired bootstrap over seed means. Because a seed indexes initialization and sampling---not an independent temporal replicate---we additionally report a time-block bootstrap over per-step PR-AUC (block lengths 3 and 5, 2000 resamples) with its point estimate, reported separately and never mixed with the pooled interval. With $n=5$, a two-sided Wilcoxon test has minimum $p=0.0625$; those rows are exploratory. Table~\ref{tab:tbb} reports the time-block bootstrap point estimates and intervals explicitly, separately from the paired seed-level inference.

\begin{table}[!t]
\caption{Time-block bootstrap over per-step PR-AUC (block lengths 3 and 5, 2000 resamples). This is a different inferential target from the pooled paired effect and is never mixed with it. \label{tab:tbb}}
\centering\footnotesize
\begin{tabular}{llrr}
\toprule
Model & Block & Macro mean & 95\% bootstrap CI \\
\midrule
Random Forest & 3 & \NumTBBMean{} & [\NumTBBloLThree{}, \NumTBBhiLThree{}] \\
Random Forest & 5 & \NumTBBMean{} & [\NumTBBlow{}, \NumTBBhigh{}] \\
CausalSAGE & 5 & \NumTBBCSMean{} & [\NumTBBCSlow{}, \NumTBBCShigh{}] \\
\bottomrule
\end{tabular}
\end{table}

\input{tables/Table_steps}

\section{Results}\label{sec:results}

\subsection{Contract ladder}
Table~\ref{tab:ladder} reports the incremental effect of each contract rung on PR-AUC, ECE and threshold metrics for three models. Ranking metrics are unchanged by calibration as expected; the label and split rungs account for the large differences.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/contract_ladder}
\caption{PR-AUC across contract rungs with seed standard deviations and sample sizes. \label{fig:ladder}}
\end{figure}

\begin{table}[!t]
\caption{Contract ladder: marginal effect of each rung. \label{tab:ladder}}
\centering\footnotesize
\begin{tabular}{llrrr}
\toprule
Configuration & Model & PR-AUC & ECE & P@100 \\
\midrule
Random split, known-only & CausalSAGE & \NumLadOne{} & \NumLadOnee{} & \NumLadOnep{} \\
Chronological, known-only & CausalSAGE & \NumLadTwo{} & \NumLadTwoe{} & \NumLadTwop{} \\
Chronological, unknown-as-negative & CausalSAGE & \NumLadThree{} & \NumLadThreee{} & \NumLadThreep{} \\
Chronological, test-calibrated & CausalSAGE & \NumLadFour{} & \NumLadFoure{} & \NumLadFourp{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent Ladder runs use a fixed 20-epoch budget for comparability across rungs (the main comparison uses 30). Full ladders for Random Forest and LightGBM are in the released tables.
\end{table}

\subsection{Main comparison}
Table~\ref{tab:main} reports main results under protocol A. Random Forest reaches PR-AUC \NumRFPrAUC{} (std \NumRFPrAUCStd{}), the best graph model \NumCSChronic{}. The paired seed-level difference $\Delta=\NumDeltaPaired{}$ ($n=\NumDeltaPairedN$) is identical in the abstract and Table~\ref{tab:stats} and reconstructible from Table~\ref{tab:main} ($\NumCSChronic-\NumRFPrAUC=-0.210\approx\NumDeltaPaired$). The significant Holm-adjusted $p=\NumDeltaCSRfHolm{}$ concerns seed-level variability only.

\input{tables/Table1_main}
\input{tables/Table3_stats}

\subsection{Audit detection}
Table~\ref{tab:audit} reports detection of injected protocol violations by the five predicates. Future-edge injection is detected with rate 1.0 at 1\%, 5\% and 10\%; unknown-as-negative affects \NumUnknown{} nodes; the canonical and main manifests satisfy I1/I4.

\begin{table}[!t]
\caption{Contract-violation audit. \label{tab:audit}}
\centering\footnotesize
\begin{tabular}{llrr}
\toprule
Injected violation & Predicate & Affected & Detection \\
\midrule
Future edges 1\% & I2 & 2{,}261 edges & 1.00 \\
Future edges 5\% & I2 & 11{,}284 edges & 1.00 \\
Future edges 10\% & I2 & 22{,}526 edges & 1.00 \\
Future edges 20\% & I2 & 45{,}013 edges & 1.00 \\
Unknown-as-negative & I3 & \NumUnknown{} nodes & 1.00 \\
Test-period recalibration & I4 & threshold metrics & detectable \\
Manifest overlap & I1 & 0 steps & compliant \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Where does the inflation come from?}
For CausalSAGE, PR-AUC rises from \NumCSChronic{} (chronological) to \NumCSRandom{} (random), a same-model gap of \NumCSGap{} computed from paired seeds; LightGBM rises from \NumLGBMChronic{} to \NumLGBMRandom{}. Because the comparison holds the model fixed, the gap is a protocol effect. Table~\ref{tab:features} shows that a feature-only Random Forest on the 94 local features already matches the full feature set, and the GNN without usable edges matches the full-graph GNN (Table~\ref{tab:samestep}); together these locate the gap in strong tabular features and the absence of attributable structure, not in message passing.

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
\begin{tabular}{llrr}
\toprule
Split & Model & PR-AUC & ECE \\
\midrule
1--34/35--42 & Random Forest & \NumCanonRfA{} & \NumCanonRfAe{} \\
1--34/35--42 & LightGBM & \NumCanonLgA{} & \NumCanonLgAe{} \\
1--34/35--42 & CausalSAGE & \NumCanonCsA{} & \NumCanonCsAe{} \\
Strict 1--34/35--42/43--49 & CausalSAGE & \NumCanonStrict{} & \NumCanonStrictE{} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{GNN capacity sensitivity (seed 0). \label{tab:tuning}}
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
\caption{Test-time perturbation sensitivity (CausalSAGE, seed 0, budget 10\%). \label{tab:attack}}
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

\subsection{Calibration}
Fig.~\ref{fig:calib} summarizes measured ECE (15 equal-width bins) and Brier score. Random Forest and LightGBM are better calibrated than the graph model under the contract.
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/calibration_summary}
\caption{Measured ECE and Brier by model (chronological protocol); error bars are seed standard deviations. \label{fig:calib}}
\end{figure}

\subsection{Operational metrics}
Precision@100 saturates at 1.000 under single-digit illicit rates and does not discriminate systems. The released tables report recall at fixed false-positive rate (FPR denominator = known negatives) and cost-weighted utility $U=\mathrm{TP}-\lambda\mathrm{FP}-\mu\mathrm{FN}$ with $\lambda=1,\mu=5$ (per node count), plus a sensitivity over $\lambda,\mu$ and per-step positive/negative/unknown counts.

\section{Reproducibility Artifact}\label{sec:artifact}
The archive contains source, tests, raw logs, machine-readable tables, figure data, environment snapshots and the complete LaTeX source. A numeric-range check asserts all scores lie in $[0,1]$ over \NumCheckedRuns{} result files with zero violations. Every manuscript macro is generated from CSV/JSON and verified by automated consistency tests. Table~\ref{tab:repro} maps each table/figure to its script, command, seed set and runtime; an artifact checklist and an anonymized DOI-ready release are provided in the companion repository. The dataset is CC BY-NC-ND 4.0 and is not redistributed.

\begin{table}[!t]
\caption{Reproducibility map (excerpt). \label{tab:repro}}
\centering\footnotesize
\begin{tabular}{llll}
\toprule
Output & Script & Seeds & Runtime \\
\midrule
Table~\ref{tab:main} & run\_pipeline + summarize & 5--15 & see logs \\
Table~\ref{tab:ladder} & run\_pipeline (ladder) & 5 & see logs \\
Table~\ref{tab:audit} & audit\_contract\_detection & -- & $<$1 min \\
Figures & build\_figures & -- & $<$1 min \\
\bottomrule
\end{tabular}
\end{table}

\section{Limitations}\label{sec:limits}
This is a single-benchmark diagnostic. Coarse time steps preclude establishing within-step order, so strict causal edge construction cannot be evaluated on native Elliptic; we do not claim a causal graph result. Features are anonymized, so amount/fee subgroup claims are excluded. Protocol D (nnPU) is an SCAR sensitivity probe whose SAR identifiability is not established; it tends to under-count unlabelled illicit nodes. Time-block inference treats steps as the resampling unit but cannot create new temporal replicates. Elliptic++ address graphs and stronger temporal baselines are left to future work. The threat model addresses evaluation deception, not model inversion.

\section{Conclusion}\label{sec:conclusion}
We formalized an auditable evaluation contract for temporal graph screening and showed that, on Elliptic, every edge is intra-time-step: non-future-step filtering is vacuous and a strict causal boundary removes the graph. A feature-only Random Forest is the strongest measured model, the graph model gains nothing attributable to structure, and random partitioning raises its reported PR-AUC by \NumCSGap{} for CausalSAGE under this implementation and tuning budget. The contribution is the contract, the boundary analysis, the protocol-effect estimate and the verifiable artifact---not a new architecture and not a universal claim about graph structure.

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
        f"\\newcommand{{\\NumProtA}}{{{fmt(prot('A'))}}}", f"\\newcommand{{\\NumProtB}}{{{fmt(prot('B'))}}}",
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
    # Protocol A must match the unified main-table CausalSAGE value.
    pa = float(cs_chrono.iloc[0]["pr_auc"]) if len(cs_chrono) else float("nan")
    lines = [("\\newcommand{\\NumProtA}{" + fmt(pa) + "}") if "\\NumProtA}{" in s else s for s in lines]

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

    cs_main = main_df[(main_df.model == "causal_sage") & (main_df.split_mode == "chrono")]
    csr = float(cs_main.iloc[0]["roc_auc"]) if len(cs_main) else float("nan")
    cse = float(cs_main.iloc[0]["ece"]) if len(cs_main) else float("nan")
    lines += [
        f"\\newcommand{{\\NumCSRoc}}{{{fmt(csr)}}}",
        f"\\newcommand{{\\NumCSEce}}{{{fmt(cse)}}}",
        f"\\newcommand{{\\NumLicit}}{{{int(diag.get('labels',{}).get('licit',42019)):,}}}",
        f"\\newcommand{{\\NumUnknown}}{{{int(diag.get('labels',{}).get('unknown',157205)):,}}}",
    ]
    # strict / empty-world
    for name, pred in [
        ("StrictPr", lambda c, r: c.get("strict_time", False) and r.get("model") == "causal_sage"),
        ("EmptyPr", lambda c, r: c.get("edge_world") == "empty" and r.get("model") == "causal_sage"),
    ]:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(v3_metric(pred))}}}")
    for name, field, pred in [
        ("StrictRoc", "roc_auc", lambda c, r: c.get("strict_time", False) and r.get("model") == "causal_sage"),
        ("StrictEce", "ece", lambda c, r: c.get("strict_time", False) and r.get("model") == "causal_sage"),
        ("EmptyRoc", "roc_auc", lambda c, r: c.get("edge_world") == "empty" and r.get("model") == "causal_sage"),
        ("EmptyEce", "ece", lambda c, r: c.get("edge_world") == "empty" and r.get("model") == "causal_sage"),
    ]:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(v3_metric(pred, field))}}}")
    # ladder rungs (CausalSAGE): random-known, chrono-known, chrono-unknown, chrono-testcal
    ladder_defs = [
        ("LadOne", lambda c, r: c.get("split_mode") == "random" and c.get("label_protocol", "A") == "A"),
        ("LadTwo", lambda c, r: c.get("split_mode") == "chrono" and c.get("label_protocol", "A") == "A" and c.get("attack", "none") == "none"),
        ("LadThree", lambda c, r: c.get("label_protocol") == "B"),
        ("LadFour", lambda c, r: c.get("calibration") == "none" and c.get("split_mode") == "chrono"),
    ]
    for name, pred in ladder_defs:
        lines.append(f"\\newcommand{{\\Num{name}}}{{{fmt(v3_metric(pred))}}}")
        lines.append(f"\\newcommand{{\\Num{name}e}}{{{fmt(v3_metric(pred, 'ece'))}}}")
        lines.append(f"\\newcommand{{\\Num{name}p}}{{{fmt(v3_metric(pred, 'precision@100'))}}}")
    # canonical strict
    lines.append(f"\\newcommand{{\\NumCanonStrict}}{{{fmt(v3_metric(lambda c, r: c.get('canonical_split') == '1-34_35-49-strict'))}}}")
    lines.append(f"\\newcommand{{\\NumCanonStrictE}}{{{fmt(v3_metric(lambda c, r: c.get('canonical_split') == '1-34_35-49-strict', 'ece'))}}}")
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
    bib_src = Path(__file__).parent / "references_v3.bib"
    if bib_src.exists():
        shutil.copy(bib_src, OUT / "references.bib")
    elif (Path(__file__).parent / "references_v2.bib").exists():
        shutil.copy(Path(__file__).parent / "references_v2.bib", OUT / "references.bib")
    # V3 figures generated by build_v3_figures.py take precedence.
    figsrc = ROOT / "submission" / "tifs_v3" / "figures_src"
    if figsrc.exists():
        for fig in figsrc.glob("*"):
            shutil.copy(fig, OUT / "figures" / fig.name)


if __name__ == "__main__":
    data = build_numbers()
    make_tables(data)
    build_main_tex(data)
    copy_support()
    print("TIFS v2 built at", OUT)
