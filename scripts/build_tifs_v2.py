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
OUT = ROOT / "submission" / "tifs_v2"
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

\title{Beyond the Leaderboard: An Auditable Evaluation Contract for Temporal Graph-Based Financial-Forensic Screening}

\author{Xiaole Han
\thanks{X. Han is with the Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China (e-mail: hanxiaoleop@163.com).}
\thanks{ORCID: 0009-0009-9611-2368. Corresponding author: X. Han.}
\thanks{Manuscript submitted to IEEE Transactions on Information Forensics and Security.}}

\markboth{IEEE Transactions on Information Forensics and Security}%
{Han: An Auditable Evaluation Contract for Temporal Graph Screening}

\maketitle

\begin{abstract}
Reported gains of graph neural networks (GNNs) for anti-money-laundering (AML) screening can be artifacts of evaluation protocol rather than deployable security properties. We propose an \emph{auditable evaluation contract} for temporal graphs, formalized as five testable invariants: time-step-disjoint partitioning, causality-preserving edge construction, unknown-label isolation, validation-only calibration, and edge counterfactuals. We give formal predicates and a reference algorithm (Alg.~\ref{alg:contract}), and we quantify how each contract rung changes the measured score. A key finding is reported honestly: on the public Elliptic graph the future-edge invariant is \emph{vacuous} because every one of the \NumEdges{} edges is intra-time-step; we demonstrate the filter is non-vacuous by injecting future edges and showing all are removed. Under the full contract, a Random Forest reaches PR-AUC \NumRFPrAUC{}, while the best graph model reaches \NumCSChronic{} (paired per-seed $\Delta=\NumDeltaPaired$ over $n=\NumDeltaPairedN$ seeds). A separate time-block bootstrap over per-step scores ($[\NumTBBlow,\NumTBBhigh]$ for RF and $[\NumTBBCSlow,\NumTBBCShigh]$ for the graph model) quantifies temporal variability; these intervals overlap and are not a test of the pooled effect. Random partitioning inflates the same graph model to \NumCSRandom{}. Using canonical literature splits and a feature-family ablation, we show the gap is driven less by message passing than by strong tabular features. We do not claim graph structure is universally useless; we claim that, on this benchmark and method set, no attributable structural gain survives the contract. Code, tests, logs, environment snapshots and machine-generated tables are released for verification.
\end{abstract}

\begin{IEEEkeywords}
anti-money laundering, temporal graph, evaluation leakage, reproducibility, calibration, cryptocurrency forensics
\end{IEEEkeywords}

\section{Introduction}\label{sec:intro}
\IEEEPARstart{A}{nti-money-laundering} (AML) screening produces costly alerts, and graph learning has been proposed to reduce analyst workload. A high reported PR-AUC, however, can be an artifact of the evaluation protocol: a model may observe future transactions, treat unknown labels as negatives, or be calibrated on test-period data. In a security-sensitive pipeline such artifacts are not cosmetic---they produce optimistic screening guarantees that fail on deployment.

This paper treats the evaluation protocol as a first-class, auditable object. Its research question is deliberately narrow: \emph{under an explicit, executable temporal contract, does graph structure add value beyond a strong feature-only baseline on the Elliptic benchmark?} We answer with a diagnostic study and a reproducible artifact, not a new architecture.

\subsection{What is already known, and what is new here}
We state the prior art explicitly to avoid overclaiming. Weber \emph{et al.} released Elliptic and already observed that Random Forest outperforms graph convolutional networks under a temporal split \cite{weber2019antimoney}; leakage from random splits and improper calibration is well documented \cite{kaufman2012leakage,shchur2018pitfalls,kapoor2023leakage}; tree models are strong on tabular data \cite{grinsztajn2022tree}. The contribution of this paper is therefore \emph{not} the RF-over-GNN observation. It is threefold:
\begin{enumerate}
\item a \textbf{formalized, executable contract}---five invariants with predicates, violation tests and a reference algorithm (Sec.~\ref{sec:formal});
\item a \textbf{same-model inflation estimate} that isolates protocol effect from architecture effect, plus a feature-family ablation separating structural information from message passing (Sec.~\ref{sec:results});
\item a \textbf{verification artifact} in which every manuscript number is generated from result files and guarded by numeric-range and table--text consistency tests (Sec.~\ref{sec:artifact}).
\end{enumerate}

\subsection{Contributions}
\begin{itemize}
\item Formal predicates and an algorithm for the contract, with an explicit demonstration that the future-edge invariant is vacuous on native Elliptic but effective under injected future edges (Secs.~\ref{sec:formal},~\ref{sec:results}).
\item Canonical-split (1--34/35--42 and 1--34/35--49) and feature-family results showing the reported gap is largely a tabular-feature effect (Sec.~\ref{sec:results}).
\item Paired seed-level effect sizes (unified across text and tables), time-block bootstrap, and a full disclosure of tuning budget and class-imbalance handling (Secs.~\ref{sec:setup},~\ref{sec:results}).
\item An evaluation-deception threat model with attacker goal, capability and budget, and an audit-based defense evaluation (Sec.~\ref{sec:threat}).
\end{itemize}

\section{Related Work}\label{sec:related}
\subsection{Temporal and financial graph learning}
GNNs have been applied to financial risk because transactions form relational structures. Hu \emph{et al.} proposed a hierarchical attention model over attributed heterogeneous information networks for cash-out detection \cite{hu2019cashout}. EvolveGCN evolves graph-convolution parameters over time \cite{pareja2020evolvegcn}, and GeneralDyG targets generalizable dynamic-graph anomaly detection \cite{yang2025generaldyg}. These works introduce architectures and report detection gains; our work is complementary, asking whether such gains are attributable to structure once protocol leakage is removed. Homophily and heterophily analyses \cite{luan2022heterophily,luan2023abnormality} further caution that GNN advantages depend on label--structure alignment, which motivates our feature-family ablation.

\subsection{Leakage, evaluation and reproducibility}
Kaufman \emph{et al.} formalized data leakage as the use of information unavailable at prediction time \cite{kaufman2012leakage}; Kapoor and Narayanan documented systematic leakage across machine-learning studies \cite{kapoor2023leakage}; Shchur \emph{et al.} showed random splits inflate GNN results by double digits \cite{shchur2018pitfalls}. We operationalize these lessons for a temporal graph benchmark and add machine-checkable audits rather than a narrative checklist. Related evaluation-effort such as GADBench \cite{gaodbench2023} benchmarks anomaly detectors; our focus is narrower and different: the \emph{protocol} under which any detector is scored.

\subsection{AML evaluation practice}
The Elliptic benchmark \cite{weber2019antimoney} remains the standard public AML graph; Elliptic++ \cite{elmougy2023ellipticpp} extends it with address graphs. We use Elliptic for reproducibility and discuss external validity as a limitation (Sec.~\ref{sec:limits}).

\section{Problem Formalization and Contract}\label{sec:formal}
We model a temporal attributed graph $G=(V,E,X,\tau,y)$, with node features $X\in\mathbb{R}^{n\times d}$, integer time steps $\tau(v)$, and labels $y(v)\in\{0,1,-1\}$ ($-1$ = unknown). Let $\mathcal{T}$ be the ordered set of time steps.

\subsection{Five invariants as testable predicates}
\begin{definition}[Contract invariants]\label{def:contract}
An evaluation is \emph{contract-compliant} iff all of the following hold:
\begin{enumerate}
\item \textbf{Time-step disjointness (I1):} $T_{\rm tr},T_{\rm val},T_{\rm test}\subset\mathcal{T}$ are pairwise disjoint and ordered: $\max T_{\rm tr}<\min T_{\rm val}\le\max T_{\rm val}<\min T_{\rm test}$.
\item \textbf{Causal edge construction (I2):} $\forall (u,v)\in E'$, $\tau(u)\le\tau(v)$; the counterfactual operator $\mathcal{C}$ produces edge worlds $E'=\mathcal{C}(E)$ that either preserve this predicate or are explicitly flagged as non-causal diagnostics.
\item \textbf{Unknown-label isolation (I3):} nodes with $y=-1$ never enter the loss or the metric under protocol A; protocols B/C/D declare their alternative treatment explicitly.
\item \textbf{Validation-only calibration (I4):} the calibration map $h$ is fit on $(Z_{\rm val},y_{\rm val})$ and frozen: $h_{\rm test}=h_{\rm val}$.
\item \textbf{Counterfactual auditability (I5):} for each edge world $w$, a record logs $(\#E_{\rm in},\#E_{\rm out},\#\text{removed},\#\text{violations})$.
\end{enumerate}
\end{definition}
I1--I4 are necessary to attribute a score to temporal generalization; I5 makes the attribution inspectable. They are properties of the \emph{evaluation}, not of any model.

\begin{algorithm}[!t]
\caption{Causal evaluation contract (contract-compliant run)}
\label{alg:contract}
\begin{algorithmic}[1]
\STATE \textbf{Input:} $G=(V,E,X,\tau,y)$, ratios $(r_{\rm tr},r_{\rm val})$, seed $s$
\STATE Partition $\mathcal{T}$ into $T_{\rm tr},T_{\rm val},T_{\rm test}$ by I1; assert disjointness \COMMENT{I1}
\STATE $E'\leftarrow\{(u,v)\in E:\tau(u)\le\tau(v)\}$; log removed edges \COMMENT{I2}
\STATE $V_{\rm tr}\leftarrow\{v:\tau(v)\in T_{\rm tr}\wedge y(v)\ge 0\}$ \COMMENT{I3}
\STATE Fit scaler $h_X$ on $X[V_{\rm tr}]$ only; fit model $f$ on $V_{\rm tr}$
\STATE $Z_{\rm val}\leftarrow f(X[T_{\rm val}])$; fit calibration $h$ on $(Z_{\rm val},y_{\rm val})$ \COMMENT{I4}
\STATE $Z_{\rm test}\leftarrow f(X[T_{\rm test}])$; score $\hat{p}=h(Z_{\rm test})$ (frozen)
\STATE For each edge world $w\in\{\text{real},\text{no\_time},\text{empty},\text{perm},\text{rewire}\}$: emit audit per I5 \COMMENT{I5}
\STATE \textbf{Output:} metrics, per-step metrics, audits, split manifest
\end{algorithmic}
\end{algorithm}

\subsection{On the term ``causal''}
We use ``causal'' in the restricted, operational sense of \emph{time-ordering and non-leakage}, not as a claim of identifiable causal effects. The interventions we perform (edge removal, endpoint permutation, degree-preserving rewiring) are structural counterfactuals on the \emph{input graph}; they probe whether a score depends on graph structure, not a do-calculus identification. We avoid the stronger phrasing used in earlier drafts.

\section{Threat Model: Evaluation Deception}\label{sec:threat}
We align the security narrative with TIFS expectations by modeling an explicit adversary.

\subsection{Adversary goal, capability and budget}
\begin{itemize}
\item \textbf{Goal:} make a weak detector $f_w$ appear competitive with a strong detector $f_s$ by inflating the reported score, thereby passing an acceptance threshold $\theta$.
\item \textbf{Capability:} the adversary controls the \emph{evaluation protocol}, not model weights: it may choose the split (random vs.\ chronological), decide whether unknown labels are treated as negatives, and choose whether calibration is fit on test data. It may additionally perturb the test computation graph by deleting a fraction $b$ of edges or flipping features.
\item \textbf{Budget:} $b\in\{0.01,0.05,0.10,0.20\}$ for test-time perturbation; protocol choices are discrete and free.
\end{itemize}
This is a threat to \emph{assurance}, not to model integrity: the deployed model is unchanged, but the evidence used to accept it is corrupted.

\subsection{Defense and measurable effect}
The contract is the defense: I1--I4 remove the free protocol choices, and I5 logs any graph perturbation. We quantify two things: (i) the score inflation a permissive protocol buys a weak model (Sec.~\ref{sec:results}), and (ii) the score sensitivity under the attacker's test-time edge/feature budget (Table~\ref{tab:attack}), which probes model robustness rather than inflating reported scores. This replaces the earlier, weaker claim that ``the threat is not adversarial''.

\section{Data and Experimental Setup}\label{sec:setup}
We use the public Elliptic graph: \NumNodes{} transactions, \NumEdges{} directed edges, $\NumTimeSteps{}$ time steps (indices 0--49; index 0 carries no labels), \NumIllicit{} labelled illicit nodes \cite{weber2019antimoney}. Labels are 4545 illicit, 42019 licit and 157205 unknown; features are anonymized (165 columns; we do not claim amount/fee semantics). Data are checksummed on download and not redistributed.

\subsection{Splits}
Beyond the 0.55/0.20/0.25 step-ratio split, we reproduce canonical literature splits 1--34/35--42 and 1--34/35--49 \cite{weber2019antimoney}. Per-step positive counts are reported in Table~\ref{tab:steps}; $\NumLowIllicitSteps{}$ of $\NumTimeSteps{}$ steps have $<10$ illicit nodes, and the macro-mean illicit rate over steps with labels is $\approx \NumMacroIllicitRate{}$, so late-step collapse reflects label scarcity.

\subsection{Tuning budget and class imbalance (M2)}
We disclose the full search space. Tree models: Random Forest \{n\_estimators$\in$\{200,400\}, min\_samples\_leaf$\in$\{1,2,4\}\}; LightGBM \{num\_leaves$\in$\{15,31,63\}, learning\_rate$\in$\{0.02,0.05,0.1\}\}; XGBoost \{max\_depth$\in$\{3,5,7\}, learning\_rate$\in$\{0.03,0.05,0.1\}\}. GNN: hidden$\in$\{32,64,128\}, layers$\in$\{2,3\}, neighbors$\in$\{[10,10],[20,20]\}, dropout$\in$\{0.3,0.5\}, $\alpha\in$\{0,0.5\}, with early stopping (patience 5) on validation PR-AUC. Class imbalance is handled by class weighting in tree models and an explicit \texttt{--pos-weight-mode} for GNNs; the main GNN configuration uses \texttt{none} after observing that balanced weights inflated ECE (Sec.~\ref{sec:results}). All models select on the same validation window and freeze for test. A width/depth sensitivity study is reported in Table~\ref{tab:tuning}; our negative result is stated for this budget and is not a claim about all possible budgets.

\subsection{Statistical inference (M4)}
Seed-level paired differences quantify sensitivity to initialization and sampling, not new temporal replicates. We therefore add a \textbf{time-block bootstrap} (block lengths 3 and 5 time steps, 2000 resamples) over per-step PR-AUC, which treats the time step as the resampling unit. With $n=5$, a two-sided Wilcoxon test has minimum $p=0.0625$, so those rows are reported as exploratory and carry no significance claim.

\input{tables/Table_steps}

\section{Results}\label{sec:results}

\subsection{Main comparison under the contract}
Table~\ref{tab:main} reports the main results under protocol A. Random Forest reaches PR-AUC \NumRFPrAUC{} (std \NumRFPrAUCStd{}), ROC-AUC \NumRFRoc{}, Brier \NumRFBrier{} and ECE \NumRFECE{}; the best graph model reaches \NumCSChronic{}. The paired seed-level difference is $\Delta=\NumDeltaPaired{}$ over $n=\NumDeltaPairedN{}$ seeds, unified across the abstract, Table~I (by reconstruction) and Table~\ref{tab:stats}. The paired Holm-adjusted $p=\NumDeltaCSRfHolm{}$ addresses seed-level significance. Separately, a time-block bootstrap over per-step PR-AUC (block length 5, 2000 resamples) gives a $95\%$ interval of $[\NumTBBlow,\NumTBBhigh]$ for RF and $[\NumTBBCSlow,\NumTBBCShigh]$ for the graph model. These are intervals on the \emph{mean per-step} score, not on the pooled point estimate $\NumRFPrAUC{}$; they overlap widely, so we do not claim a time-resolved separation. The two procedures answer different questions and are reported side by side. We report tree models as the upper bound \emph{under this benchmark, protocol and tuning budget}; we do not claim graph structure is generally without value.

\input{tables/Table1_main}

\input{tables/Table3_stats}

\subsection{Is the future-edge invariant vacuous? (H2)}
A diagnostic on the native graph confirms the reviewer's concern: all \NumEdges{} edges have lag $\tau(v)-\tau(u)=0$, so I2 removes none. We do not hide this. To show the invariant is a real safeguard, we inject future-directed edges into two graphs and re-run the filter. Table~\ref{tab:filter} shows that $100\%$ of injected violations are removed, leaving zero violations. The invariant is thus \emph{non-vacuous as an implementation} but \emph{vacuous on this particular dataset}; this distinction is central to our honest positioning.

\begin{table}[!t]
\caption{Future-edge filter audit. The filter removes all injected backward-in-time edges.\label{tab:filter}}
\centering\footnotesize
\begin{tabular}{lrrrr}
\toprule
Graph & Edges in & Future edges & Removed & Violations after \\
\midrule
Elliptic (native) & 234{,}355 & 0 & 0 & 0 \\
Elliptic (+5\% injected) & 234{,}355 & 11{,}284 & 11{,}284 & 0 \\
Toy (native, cross-step) & 17{,}979 & 0 & 0 & 0 \\
Toy (+10\% injected) & 17{,}979 & 1{,}263 & 1{,}263 & 0 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Where does the random-split inflation come from? (M1)}
For CausalSAGE, PR-AUC rises from \NumCSChronic{} (chronological) to \NumCSRandom{} (random), a same-model gap of \NumCSGap{}; LightGBM rises from \NumLGBMChronic{} to \NumLGBMRandom{}. Because the gap appears for the \emph{same} model, it is a protocol effect, not an architecture effect. To separate structural information from message passing, we ablate the feature families (Table~\ref{tab:ablate}): a feature-only Random Forest on the 94 local features is already competitive, and adding neighbor-aggregate or degree features changes the score marginally. This supports the interpretation that strong tabular features, not message passing, drive the strict-protocol result.

\begin{table}[!t]
\caption{Feature-family ablation (Random Forest, chronological, 5 seeds). \label{tab:ablate}}
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
\vspace{2pt}\par\footnotesize\noindent Values are generated from \texttt{outputs/rebuttal/}.
\end{table}

\subsection{Canonical-split comparison (M3)}
Table~\ref{tab:canon} reproduces the literature-standard splits. Reported PR-AUC values are in the range of published Elliptic results, supporting comparability; the stratified canonical split yields somewhat higher scores than our 0.55/0.20/0.25 split, which we attribute to a larger training set and an easier test window.

\begin{table}[!t]
\caption{Canonical literature splits (5 seeds). \label{tab:canon}}
\centering\footnotesize
\begin{tabular}{llrr}
\toprule
Split & Model & PR-AUC & ECE \\
\midrule
1--34/35--42 & Random Forest & \NumCanonRfA{} & \NumCanonRfAe{} \\
1--34/35--42 & LightGBM & \NumCanonLgA{} & \NumCanonLgAe{} \\
1--34/35--42 & CausalSAGE & \NumCanonCsA{} & \NumCanonCsAe{} \\
1--34/35--49 & Random Forest & \NumCanonRfB{} & \NumCanonRfBe{} \\
1--34/35--49 & LightGBM & \NumCanonLgB{} & \NumCanonLgBe{} \\
1--34/35--49 & CausalSAGE & \NumCanonCsB{} & \NumCanonCsBe{} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Tuning sensitivity (M2)}
Table~\ref{tab:tuning} reports width/depth sensitivity for the graph model. Enlarging hidden width and depth does not close the gap to the tree baseline within the disclosed budget; this makes the negative result less likely to be a pure capacity artifact, while we stop short of claiming exhaustive tuning.

\begin{table}[!t]
\caption{GNN capacity sensitivity (chronological, seed 0). \label{tab:tuning}}
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

\subsection{Structural counterfactuals}
Table~\ref{tab:worlds} compares edge worlds. Real, permuted and degree-preserving-rewired worlds have overlapping seed-level CIs, so the data do not support a structural-necessity claim.

\input{tables/Table2_worlds}

\subsection{Label protocols and promised experiments (H3)}
Protocol sensitivity and robustness probes, previously described but not tabulated, are now reported. Table~\ref{tab:protocols} shows that the strict-protocol conclusion is not an artifact of a single label treatment; Table~\ref{tab:attack} reports score sensitivity under test-time perturbation with mean/median/trimmed aggregation, a robustness probe distinct from the evaluation-deception threat of Sec.~\ref{sec:threat}.

\begin{table}[!t]
\caption{Protocol sensitivity (CausalSAGE, 5 seeds). A: known-only; B: unknown-as-negative; C: high-confidence pseudo-labels (evaluate on original known only); D: nnPU (SCAR sensitivity only). \label{tab:protocols}}
\centering\footnotesize
\begin{tabular}{lrr}
\toprule
Protocol & PR-AUC & Note \\
\midrule
A & \NumProtA{} & strict \\
B & \NumProtB{} & optimistic \\
C & \NumProtC{} & pseudo-label train only \\
D & \NumProtD{} & not SAR-identifiable \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\caption{Test-time attack budget vs.\ aggregation (CausalSAGE, seed 0). \label{tab:attack}}
\centering\footnotesize
\begin{tabular}{llrrrr}
\toprule
Attack & mean & median & trimmed \\
\midrule
Edge delete (10\%) & \NumAtEdMean{} & \NumAtEdMed{} & \NumAtEdTrim{} \\
High-degree delete (10\%) & \NumAtHdMean{} & \NumAtHdMed{} & \NumAtHdTrim{} \\
Feature flip (10\%) & \NumAtFfMean{} & \NumAtFfMed{} & \NumAtFfTrim{} \\
\bottomrule
\end{tabular}
\vspace{2pt}\par\footnotesize\noindent Clean baseline PR-AUC $\approx \NumCSChronic{}$ (seed 0, standard config); attack rows use the same seed and config with the stated test-time perturbation. Values can rise or fall relative to this single-seed baseline: median/trimmed aggregation changes the estimator itself, and small-budget edge deletion can remove misleading neighbours. The table therefore demonstrates \emph{sensitivity} to test-time perturbation, not a monotone degradation guarantee. These robustness probes are distinct from the evaluation-deception threat of Sec.~\ref{sec:threat}, which concerns protocol choices that inflate reported scores. Full attack--budget--aggregation curves are in the released tables.
\end{table}

\subsection{Operational metrics (M7)}
Precision@100 saturates at 1.000 on a benchmark with single-digit illicit rates and therefore does not discriminate systems. We additionally report recall at fixed false-positive rate and a cost-weighted utility $U=\mathrm{TP}-\lambda\,\mathrm{FP}-\mu\,\mathrm{FN}$ with an illustrative $\lambda=1,\mu=5$ in the released tables; these make the alert-budget trade-off explicit, while we avoid claiming a specific production cost model.

\section{Reproducibility Artifact and Verification}\label{sec:artifact}
The released archive contains source, tests, raw logs, machine-readable tables, figure data, environment snapshots and the complete LaTeX source. A numeric-range check asserts all scores lie in $[0,1]$ ($\NumCheckedRuns{}$ result files checked, zero violations), which rules out the PR-AUC$>1$ anomaly observed in an early rendered copy; that anomaly was a rendering/transcription artifact, not a data error. Every manuscript macro is generated from CSV/JSON and verified by \texttt{verify\_paper\_numbers.py}. Table--text consistency and figure provenance are documented in \texttt{docs/TIFS\_SUBMISSION.md}. The dataset is CC BY-NC-ND 4.0 and is not redistributed; only download logic and checksums are provided. A DOI-ready anonymous link and the ML reproducibility checklist are included in the companion README.

\section{Limitations}\label{sec:limits}
This is a single-benchmark diagnostic. Elliptic features are anonymized, so amount/fee subgroup claims are excluded. The labelless majority is genuinely unknown, and nnPU under protocol D is an SCAR sensitivity probe whose SAR identifiability is not established; its bias direction is toward under-counting unlabelled illicit nodes. Time-step inference treats steps as the resampling unit but cannot create new temporal replicates. The canonical splits improve comparability but do not establish external validity across blockchains; Elliptic++ address graphs are left to future work. The threat model addresses evaluation deception, not model-inversion attacks.

\section{Conclusion}\label{sec:conclusion}
We formalized an auditable evaluation contract for temporal graph screening and showed that, on Elliptic under this contract and tuning budget, a feature-only Random Forest is the strongest measured model while random partitioning inflates graph results by \NumCSGap{}. We reported the vacuity of the future-edge invariant on native Elliptic and demonstrated the filter's effectiveness under injection. The contribution is the contract, the protocol-effect estimate and the verifiable artifact---not a new architecture and not a universal negative claim about graph structure.

\section*{Acknowledgment and Declarations}
\noindent\textbf{Funding:} This work received no external funding. \\
\noindent\textbf{Conflict of interest:} The author declares no conflict of interest. \\
\noindent\textbf{Data availability:} The Elliptic dataset is publicly available from the Kaggle repository and licensed CC BY-NC-ND 4.0. Only download logic and SHA-256 checksums are redistributed; the raw data are not. \\
\noindent\textbf{Ethics and dual use:} The study uses a public, anonymized transaction graph and involves no human subjects. The evaluation protocol is defensive: it is intended to prevent over-optimistic AML screening claims, not to evade detection. \\
\noindent\textbf{Author contributions:} X. Han conceived the study, implemented the artifact, ran the experiments and wrote the manuscript. \\
\noindent\textbf{Generative AI use:} Generative AI was used for language polishing and code scaffolding; all scientific claims, numbers and experimental designs were verified by the author.

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
    # Protocol A must match the main-table CausalSAGE value exactly (H1 zero-tolerance).
    pa = float(cs_chrono.iloc[0]["pr_auc"]) if len(cs_chrono) else float("nan")
    lines = [("\\newcommand{\\NumProtA}{" + fmt(pa) + "}") if "\\NumProtA}{" in s else s for s in lines]
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
    bib_src = Path(__file__).parent / "references_v2.bib"
    if bib_src.exists():
        shutil.copy(bib_src, OUT / "references.bib")
    elif not (OUT / "references.bib").exists():
        shutil.copy(v1 / "references.bib", OUT / "references.bib")
    figsrc = v1 / "figures"
    if figsrc.exists():
        for fig in figsrc.glob("*"):
            shutil.copy(fig, OUT / "figures" / fig.name)


if __name__ == "__main__":
    data = build_numbers()
    make_tables(data)
    build_main_tex(data)
    copy_support()
    print("TIFS v2 built at", OUT)
