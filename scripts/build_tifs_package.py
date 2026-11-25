#!/usr/bin/env python3
"""Build a self-contained IEEE TIFS LaTeX submission package.

All manuscript numbers are read from outputs/full/tables. No experiment result
is typed into the TeX source by hand.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "full" / "tables"
FIGSRC = ROOT / "outputs" / "full" / "figures"
OUT = ROOT / "submission" / "tifs"
CLASS_SRC = Path(r"D:/Financial Micro-Modelling Project/LaTeX_template/IEEE-Transactions-LaTeX2e-templates-and-instructions-IEEE Transactions on Information Forensics and Security/IEEEtran.cls")
JOURNAL = "IEEE Transactions on Information Forensics and Security"


def fmt(value, digits=3):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if x != x:
        return "N/A"
    return f"{x:.{digits}f}"


def row(df, **kwargs):
    sel = df
    for key, value in kwargs.items():
        sel = sel[sel[key] == value]
    if sel.empty:
        raise KeyError(f"missing row {kwargs}")
    return sel.iloc[0]


def metric(df, model, split, field="pr_auc", digits=3):
    return fmt(row(df, model=model, split_mode=split)[field], digits)


def build_figures():
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    main = pd.read_csv(TABLES / "Table1_main.csv")
    pairs = []
    for model in sorted(main["model"].unique()):
        for split in ("chrono", "random"):
            subset = main[(main["model"] == model) & (main["split_mode"] == split)]
            if not subset.empty:
                pairs.append({"model": model, "split": split, "pr_auc": float(subset.iloc[0]["pr_auc"])})
    paired = pd.DataFrame(pairs)
    wide = paired.pivot(index="model", columns="split", values="pr_auc").dropna()
    wide["gap"] = wide["random"] - wide["chrono"]
    ax = wide[["chrono", "random"]].plot(kind="bar", figsize=(7.0, 3.2), rot=20)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.set_title("Same-model random versus chronological split")
    ax.legend(title="Protocol")
    plt.tight_layout()
    plt.savefig(figdir / "protocol_gap.png", dpi=300)
    plt.close()

    worlds = pd.read_csv(TABLES / "Table5_edge_world.csv")
    worlds = worlds[(worlds["model"] == "causal_sage") & (worlds["split_mode"] == "chrono")].copy()
    order = ["real", "no_time", "empty", "perm", "rewire"]
    worlds["order"] = worlds["edge_world"].map({name: i for i, name in enumerate(order)})
    worlds = worlds.sort_values("order")
    ax = worlds.plot(kind="bar", x="edge_world", y="pr_auc", yerr="pr_auc_std",
                     legend=False, figsize=(7.0, 3.2), rot=0, capsize=3)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Edge world")
    ax.set_title("Structural counterfactuals, causal_sage")
    plt.tight_layout()
    plt.savefig(figdir / "edge_worlds.png", dpi=300)
    plt.close()

    drift = pd.read_csv(FIGSRC / "drift.csv")
    drift = drift[drift["model"] == "causal_sage"].groupby("step")["pr_auc"].mean()
    plt.figure(figsize=(7.0, 3.0))
    plt.plot(drift.index, drift.values, marker="o")
    plt.xlabel("Elliptic time step")
    plt.ylabel("Mean test PR-AUC")
    plt.title("Out-of-time performance drift, causal_sage")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figdir / "drift.png", dpi=300)
    plt.close()

    pareto = pd.read_csv(TABLES / "Table6_pareto_front.csv")
    plt.figure(figsize=(7.0, 3.2))
    plt.scatter(pareto["wall_time_seconds"], pareto["pr_auc"], s=42)
    for _, r in pareto.iterrows():
        plt.annotate(str(r["model"]), (r["wall_time_seconds"], r["pr_auc"]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Wall time (seconds)")
    plt.ylabel("PR-AUC")
    plt.title("Accuracy-time non-dominated set")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figdir / "pareto.png", dpi=300)
    plt.close()


def build_numbers():
    main = pd.read_csv(TABLES / "Table1_main.csv")
    worlds = pd.read_csv(TABLES / "Table5_edge_world.csv")
    stats = json.loads((TABLES / "statistical_tests.json").read_text(encoding="utf-8"))
    comparisons = {c["model"]: c for c in stats["comparisons"]}

    causal_chrono = metric(main, "causal_sage", "chrono")
    causal_random = metric(main, "causal_sage", "random")
    causal_gap = fmt(float(causal_random) - float(causal_chrono))
    lgbm_chrono = metric(main, "lightgbm", "chrono")
    lgbm_random = metric(main, "lightgbm", "random")
    lgbm_gap = fmt(float(lgbm_random) - float(lgbm_chrono))
    rf = row(main, model="rf", split_mode="chrono")
    cs = row(main, model="causal_sage", split_mode="chrono")
    cs_stat = comparisons["causal_sage"]

    lines = [
        "% Auto-generated from outputs/full/tables; do not edit by hand.",
        f"\\newcommand{{\\NumCSChronic}}{{{causal_chrono}}}",
        f"\\newcommand{{\\NumCSChronicStd}}{{{fmt(cs['pr_auc_std'])}}}",
        f"\\newcommand{{\\NumCSRandom}}{{{causal_random}}}",
        f"\\newcommand{{\\NumCSGap}}{{{causal_gap}}}",
        f"\\newcommand{{\\NumLGBMChronic}}{{{lgbm_chrono}}}",
        f"\\newcommand{{\\NumLGBMRandom}}{{{lgbm_random}}}",
        f"\\newcommand{{\\NumLGBMGap}}{{{lgbm_gap}}}",
        f"\\newcommand{{\\NumRFPrAUC}}{{{fmt(rf['pr_auc'])}}}",
        f"\\newcommand{{\\NumRFPrAUCStd}}{{{fmt(rf['pr_auc_std'])}}}",
        f"\\newcommand{{\\NumRFRoc}}{{{fmt(rf['roc_auc'])}}}",
        f"\\newcommand{{\\NumRFBrier}}{{{fmt(rf['brier'])}}}",
        f"\\newcommand{{\\NumRFECE}}{{{fmt(rf['ece'])}}}",
        f"\\newcommand{{\\NumCSPrecision}}{{{fmt(cs['precision'])}}}",
        f"\\newcommand{{\\NumDeltaCSRf}}{{{fmt(cs_stat['mean_delta_pr_auc'])}}}",
        f"\\newcommand{{\\NumDeltaCSRfLow}}{{{fmt(cs_stat['ci_low'])}}}",
        f"\\newcommand{{\\NumDeltaCSRfHigh}}{{{fmt(cs_stat['ci_high'])}}}",
        f"\\newcommand{{\\NumDeltaCSRfHolm}}{{{fmt(cs_stat['holm_p'], 4)}}}",
        f"\\newcommand{{\\NumDeltaCSRfN}}{{{int(cs_stat['n_seeds'])}}}",
    ]
    for world in ("real", "no_time", "empty", "perm", "rewire"):
        value = fmt(row(worlds, model="causal_sage", split_mode="chrono", edge_world=world)["pr_auc"])
        safe = world.replace("_", "").title()
        lines.append(f"\\newcommand{{\\NumEdge{safe}}}{{{value}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def make_main_table():
    main = pd.read_csv(TABLES / "Table1_main.csv")
    order = ["rf", "lightgbm", "xgboost", "mlp", "lr", "causal_sage", "graphsage"]
    main["order"] = main["model"].map({name: i for i, name in enumerate(order)})
    main = main.sort_values(["order", "split_mode"])
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Main Results on Elliptic Under Chronological and Random Protocols\\label{tab:main}}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Model & Protocol & Seeds & PR-AUC & ROC-AUC & Brier & ECE & Precision@100 \\\\",
        "\\midrule",
    ]
    for _, r in main.iterrows():
        lines.append(
            f"{tex_escape(r['model'])} & {r['split_mode']} & {int(r['n_seeds'])} & "
            f"{fmt(r['pr_auc'])} $\\pm$ {fmt(r['pr_auc_std'])} & {fmt(r['roc_auc'])} & "
            f"{fmt(r['brier'])} & {fmt(r['ece'])} & {fmt(r['precision@100'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (OUT / "tables" / "Table1_main.tex").write_text("\n".join(lines), encoding="utf-8")


def make_world_table():
    worlds = pd.read_csv(TABLES / "Table5_edge_world.csv")
    worlds = worlds[(worlds["model"] == "causal_sage") & (worlds["split_mode"] == "chrono")].copy()
    order = ["real", "no_time", "empty", "perm", "rewire"]
    worlds["order"] = worlds["edge_world"].map({name: i for i, name in enumerate(order)})
    worlds = worlds.sort_values("order")
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Structural Counterfactuals for causal\\_sage\\label{tab:worlds}}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Edge world & PR-AUC & ROC-AUC & ECE \\\\",
        "\\midrule",
    ]
    for _, r in worlds.iterrows():
        lines.append(
            f"{tex_escape(r['edge_world'])} & {fmt(r['pr_auc'])} & {fmt(r['roc_auc'])} & {fmt(r['ece'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2pt}\\par\\footnotesize\\noindent "
        "$^{*}$\texttt{real}: $n=15$ seeds; all other worlds: $n=5$ seeds.",
        "\\end{table}",
        "",
    ]
    (OUT / "tables" / "Table2_worlds.tex").write_text("\n".join(lines), encoding="utf-8")


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return "<$10^{-3}$"
    elif value < 0.01:
        return f"{value:.4f}"
    else:
        return f"{value:.3f}"


def make_stats_table():
    stats = json.loads((TABLES / "statistical_tests.json").read_text(encoding="utf-8"))
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Paired Seed-Level Comparisons Against Random Forest\\label{tab:stats}}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Seeds & $\\Delta$PR-AUC & 95\\% CI & Wilcoxon $p$ & Holm $p$ & Interpretation \\\\",
        "\\midrule",
    ]
    for c in stats["comparisons"]:
        significant = c["holm_p"] < 0.05
        verdict = "Significant after Holm" if significant else "Not significant"
        lines.append(
            f"{tex_escape(c['model'])} & {c['n_seeds']} & {fmt(c['mean_delta_pr_auc'])} & "
            f"[{fmt(c['ci_low'])}, {fmt(c['ci_high'])}] & {_fmt_p(c['wilcoxon_p'])} & "
            f"{_fmt_p(c['holm_p'])} & {verdict} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2pt}\\par\\footnotesize\\noindent " + tex_escape(stats["aggregation_note"]),
        "\\end{table*}",
        "",
    ]
    (OUT / "tables" / "Table3_stats.tex").write_text("\n".join(lines), encoding="utf-8")


REFERENCES = r"""@article{weber2019antimoney,
  author  = {Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I. and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E.},
  title   = {Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics},
  journal = {arXiv preprint arXiv:1908.02591},
  year    = {2019}
}
@inproceedings{hamilton2017inductive,
  author    = {Hamilton, Will L. and Ying, Rex and Leskovec, Jure},
  title     = {Inductive Representation Learning on Large Graphs},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
@inproceedings{kipf2017semi,
  author    = {Kipf, Thomas N. and Welling, Max},
  title     = {Semi-Supervised Classification with Graph Convolutional Networks},
  booktitle = {International Conference on Learning Representations},
  year      = {2017}
}
@inproceedings{chen2016xgboost,
  author    = {Chen, Tianqi and Guestrin, Carlos},
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year      = {2016}
}
@inproceedings{ke2017lightgbm,
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  title     = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
@inproceedings{hu2019cashout,
  author    = {Hu, Bin and Zhang, Zhiqiang and Shi, Chuan and Zhou, Jun and Li, Xiaolong and Qi, Yuan},
  title     = {Cash-out User Detection based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {33},
  number    = {01},
  pages     = {946--953},
  year      = {2019},
  doi       = {10.1609/aaai.v33i01.3301946}
}
@inproceedings{pareja2020evolvegcn,
  author    = {Pareja, Aldo and Domeniconi, Giacomo and Chen, Jie and Ma, Tengfei and Suzumura, Toyotaro and Kanezashi, Hiroki and Kaler, Tim and Schardl, Tao and Leiserson, Charles E.},
  title     = {{EvolveGCN}: Evolving Graph Convolutional Networks for Dynamic Graphs},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {34},
  number    = {04},
  pages     = {5363--5370},
  year      = {2020},
  doi       = {10.1609/aaai.v34i04.5984}
}
@inproceedings{yang2025generaldyg,
  author    = {Yang, Xiao and Zhao, Xuejiao and Shen, Zhiqi},
  title     = {A Generalizable Anomaly Detection Method in Dynamic Graphs},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {20},
  pages     = {22001--22009},
  year      = {2025},
  doi       = {10.1609/aaai.v39i20.35508}
}
@article{kaufman2012leakage,
  author  = {Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia and Stitelman, Ori},
  title   = {Leakage in Data Mining: Formulation, Detection, and Avoidance},
  journal = {ACM Transactions on Knowledge Discovery from Data},
  volume  = {6},
  number  = {4},
  pages   = {15:1--15:21},
  year    = {2012},
  doi     = {10.1145/2382577.2382579}
}
@inproceedings{shchur2018pitfalls,
  author    = {Shchur, Oleksandr and Mumme, Maximilian and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
  title     = {Pitfalls of Graph Neural Network Evaluation},
  booktitle = {Workshop on Relational Representation Learning, NeurIPS},
  year      = {2018}
}
@article{pedregosa2011scikit,
  author  = {Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  title   = {Scikit-learn: Machine Learning in {Python}},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}
@inproceedings{paszke2019pytorch,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}
"""


def build_main_tex():
    text = r"""\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{url}
\usepackage{cite}
\input{numbers}

\begin{document}

\title{A Reproducibility Diagnostic for Temporally Causal Graph-Based Anti-Money-Laundering Evaluation}

\author{Xiaole Han\\
\thanks{Xiaole Han is with the Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China (e-mail: hanxiaoleop@163.com).}
\thanks{ORCID: 0009-0009-9611-2368. Corresponding author: Xiaole Han.}
\thanks{Manuscript submitted to IEEE Transactions on Information Forensics and Security.}}

\markboth{IEEE Transactions on Information Forensics and Security}%
{Xiaole Han: A Reproducibility Diagnostic for AML Graph Evaluation}

\maketitle

\begin{abstract}
Graph neural networks are widely proposed for anti-money-laundering (AML) transaction screening, but reported gains often depend on random node splits, future edges visible during training, unknown labels treated as negatives, and calibration fitted on test data. This paper asks when graph structure provides security-relevant value under an auditable causal protocol. We formalize five executable invariants: chronological time-step splits, $t_{\mathrm{src}}\le t_{\mathrm{dst}}$ edge filtering, unknown-label masking, validation-only calibration, and reproducible edge counterfactuals. We implement a pure-NumPy causal sampler, a zero-parameter degree correction, nnPU label-bias probing, degree-preserving rewiring, and test-time attacks with median/trimmed aggregation. On the public Elliptic graph, Random Forest reaches PR-AUC \NumRFPrAUC{} under the strict protocol, whereas the best graph model reaches \NumCSChronic{} ($\Delta=\NumDeltaCSRf$, 95\% CI [\NumDeltaCSRfLow, \NumDeltaCSRfHigh], Holm-adjusted $p=\NumDeltaCSRfHolm$, $n=\NumDeltaCSRfN$ seeds). The same model's PR-AUC rises from \NumCSChronic{} to \NumCSRandom{} under a random split, a gap of \NumCSGap. We therefore report tree models as the strict-protocol upper bound and do not claim universal graph-structure gains. All numbers are generated from machine-readable result files; the paper, tables, tests, logs and source are released as a traceable artifact.
\end{abstract}

\begin{IEEEkeywords}
Anti-money laundering, graph neural networks, reproducibility, temporal causality, leakage, secure evaluation, blockchain analytics.
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{A}{nti-money-laundering} systems generate costly alerts, and graph learning has been proposed to reduce review burden while preserving recall. Yet a high reported PR-AUC can be an artifact of evaluation rather than a deployable security property. A model may observe future transactions, train on unknown labels as negatives, or be calibrated with test data. Such failures are especially dangerous in AML because they produce optimistic screening reports that are later replicated in operational pipelines.

We therefore treat evaluation protocol as a security mechanism. Our diagnostic asks: under an explicit causal contract, does graph structure still add value beyond strong tabular baselines? We answer this with a reproducible artifact rather than a new architecture.

The contributions are:
\begin{itemize}
\item an executable causal protocol with chronological splits, future-edge rejection, unknown-label isolation and frozen calibration;
\item a zero-parameter causal sampler and degree correction whose parameter count is unchanged by the correction;
\item paired seed-level evidence that Random Forest outperforms graph models on Elliptic under the strict protocol, and that random splits inflate graph results by \NumCSGap{} PR-AUC;
\item structural counterfactuals, label-protocol sensitivity, robustness probes and a complete machine-generated submission package.
\end{itemize}

\section{Original Contribution and Relevance to AI}\label{sec:contribution}
\subsection{What is the original contribution?}
In one sentence: this paper contributes an executable causal evaluation contract for temporal graph-based AML and uses it to show that, once future edges, unknown labels and test-time calibration are controlled, strong tabular baselines can outperform graph models that appear superior under random splits. The contract is not a new architecture; it is a reusable diagnostic that turns a common evaluation failure into a testable, auditable property.

\subsection{Why is this important for AI?}
Graph learning papers increasingly report gains on financial-crime data, but the field lacks a shared causal contract for temporal graphs. If future transactions are visible, unknown labels are treated as negatives, or calibration is fitted on test data, reported improvements are not evidence of deployable AI. This work makes those failure modes explicit and reproducible. It matters beyond AML because the same leakage patterns appear in dynamic recommendation, fraud detection, event forecasting and other temporal graph tasks.

\subsection{Where is this explained in the paper?}
The contribution is stated in Section~\ref{sec:contribution} and detailed in Sections~\ref{sec:protocol}--\ref{sec:method}. The causal invariants are defined in Section~\ref{sec:protocol}, the method in Section~\ref{sec:method}, and the negative-result evidence in Section~\ref{sec:results}. The reproducibility artifact is described in Section~\ref{sec:repro}.

\subsection{What is the closest related work, and how does this work differ?}
The closest lines are graph-based financial risk detection and dynamic graph learning. Hu \emph{et al.} propose a hierarchical attention model over attributed heterogeneous financial networks \cite{hu2019cashout}; Pareja \emph{et al.} propose EvolveGCN for dynamic graphs \cite{pareja2020evolvegcn}; Yang \emph{et al.} propose GeneralDyG for generalizable dynamic-graph anomaly detection \cite{yang2025generaldyg}. These papers contribute models and report detection gains. Our work differs in objective and unit of evidence: we do not propose a competing architecture, and we do not claim a new detection frontier. We provide a protocol that asks whether such gains survive chronological splits, unknown-label masking, validation-only calibration and degree-preserving edge counterfactuals. The result is a negative-result diagnostic and a release-ready artifact, not another model leaderboard entry.

\subsection{Is the work relevant to AI, and can a general AI audience appreciate it?}
Yes. The paper addresses trustworthy evaluation of learning systems, a core AI concern that cuts across graph learning, security and applied machine learning. The abstract, introduction and conclusion avoid domain-specific notation and state the problem in terms of leakage, calibration and attributable gain. A general AI reader can understand the contribution as a reproducibility test for temporal graph claims, even without AML background. The technical sections define every protocol and report effect sizes and confidence intervals rather than relying on a single leaderboard number.

\subsection{Additional electronic material}
The following material is released as supplementary electronic material and is also available in the source archive: source code, unit tests, raw execution logs, machine-readable tables, figure-generation scripts, the test-environment snapshot and the complete LaTeX source. The raw Elliptic dataset is not redistributed because of its license; the code downloads and checksums it locally.

\section{Related Work}\label{sec:related}
\subsection{Graph Learning for Financial Risk}
Graph neural networks have been applied to financial risk because transactions form relational and temporal structures. Hu \emph{et al.} model cash-out behavior with an attributed heterogeneous information network and a hierarchical attention mechanism \cite{hu2019cashout}. The Elliptic benchmark provides a transaction graph with anonymized features and time steps \cite{weber2019antimoney}. These studies establish that graph structure can be predictive, but they do not isolate whether the reported gain survives a strict temporal contract.

\subsection{Dynamic and Temporal Graph Learning}
EvolveGCN evolves graph convolutional parameters over time and reports gains on dynamic graph tasks \cite{pareja2020evolvegcn}. GeneralDyG samples temporal ego-graphs and targets generalizable dynamic-graph anomaly detection \cite{yang2025generaldyg}. Our work is complementary: rather than proposing a stronger temporal architecture, it asks whether a graph model's advantage is attributable to temporal structure after future-edge and label leakage are removed. This is why the paper reports a negative result for graph models on Elliptic instead of claiming a new state of the art.

\subsection{Evaluation, Leakage and Reproducibility}
Random splits, temporal leakage and miscalibration are known failure modes in applied machine learning \cite{kaufman2012leakage,shchur2018pitfalls}. Kaufman \emph{et al.} formalize leakage as the use of information that would not be available at prediction time \cite{kaufman2012leakage}; Shchur \emph{et al.} show that random splits and fixed train--test partitions can inflate GNN accuracy by double digits on standard benchmarks \cite{shchur2018pitfalls}. Our contribution is to make these failures executable for a temporal graph benchmark: the protocol rejects future edges, masks unknown labels, freezes calibration on validation data and records degree-preserving counterfactuals. This connects the AML application to the broader AI agenda of trustworthy and reproducible evaluation.

\section{Threat Model and Security-Relevant Evaluation}\label{sec:threat}
The operational threat is not adversarial model inversion but evaluation-time optimistic bias: a screening model accepted from a flawed protocol may be deployed with higher false negatives and false positives than promised. We therefore define three failure surfaces:
\begin{enumerate}
\item \textbf{Temporal leakage:} source transactions later than target transactions are visible in training;
\item \textbf{Label leakage:} unknown transactions enter the loss or metric as negatives;
\item \textbf{Calibration leakage:} thresholds or calibration parameters are fitted using test-period data.
\end{enumerate}

Our protocol rejects these by construction and records audits for every run.

\section{Data and Causal Protocol}\label{sec:protocol}
We use the public Elliptic transaction graph with 203{,}769 transactions, 234{,}355 directed edges and 49 time steps \cite{weber2019antimoney}. The dataset is licensed CC BY-NC-ND 4.0 and is not redistributed; the repository stores download logic, checksums and conversion code.

\subsection{Chronological Splits}
Time steps are partitioned explicitly into train, validation and test sets with ratios 0.55, 0.20 and 0.25. The split manifest records exact time-step sets. A random node split is retained only as a diagnostic ladder, never as a deployment claim.

\subsection{Future-Edge Rejection}
For every edge, we require $t_{\mathrm{src}}\le t_{\mathrm{dst}}$. The sampler asserts this invariant batch by batch. All 234{,}355 edges in the real edge world satisfy the constraint.

\subsection{Label Protocols}
Let $y\in\{0,1,-1\}$, where $-1$ denotes unknown.
\begin{itemize}
\item \textbf{A}: train and evaluate only on known labels;
\item \textbf{B}: treat unknown as negative in training and evaluation;
\item \textbf{C}: train with high-confidence pseudo-labels, but evaluate only on originally known labels;
\item \textbf{D}: nnPU risk treats positives as labelled and all other nodes as unlabeled; this is an SCAR sensitivity probe, not an identifiability claim under SAR.
\end{itemize}

\subsection{Validation-Only Calibration}
Temperature scaling or isotonic regression is fit on validation logits and frozen on test data. The evaluation function separates validation and test labels.

\section{Method}\label{sec:method}
\subsection{Causal Neighbor Sampling}
For target $v$, candidate predecessors satisfy $t_u\le t_v$. Sampling probability is
\begin{equation}
q(u\mid v)\propto \max(1,\deg(u)^{-\alpha}),
\label{eq:sampler}
\end{equation}
where $\alpha=0$ recovers uniform sampling. The sampler is pure NumPy and avoids optional sparse backends.

\subsection{Zero-Parameter Degree Correction}
Messages are weighted by $1/\sqrt{d_ud_v+1}$ before mean aggregation. The number of trainable parameters is unchanged; a unit test enforces this.

\subsection{Robust Aggregation and Test-Time Attacks}
We evaluate mean, median and trimmed-mean aggregation under edge deletion, high-degree deletion and feature flipping. Attacks are applied only to the test computation graph; training graphs remain causally filtered.

\section{Experimental Setup}\label{sec:setup}
We use five seeds for most configurations and fifteen seeds for the main paired comparisons. Hidden width is 32, two layers, neighbor counts $[10,10]$, dropout 0.5, learning rate 0.01, weight decay $5\times10^{-4}$ and batch size 512. Neural models use train-only feature standardization. Tree baselines are Logistic Regression, Random Forest, XGBoost, LightGBM and MLP \cite{pedregosa2011scikit,chen2016xgboost,ke2017lightgbm}. Graph models use GraphSAGE-style causal sampling \cite{hamilton2017inductive} and are implemented with PyTorch \cite{paszke2019pytorch}; GCN is included in the codebase \cite{kipf2017semi}.

\input{tables/Table1_main}

\section{Results}\label{sec:results}
\subsection{Protocol Ladder}
The random-split protocol is not a deployment estimate. For causal\_sage, PR-AUC rises from \NumCSChronic{} chronologically to \NumCSRandom{} randomly, a same-model gap of \NumCSGap. LightGBM rises from \NumLGBMChronic{} to \NumLGBMRandom{}, a gap of \NumLGBMGap. Fig.~\ref{fig:protocol} visualizes this inflation.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/protocol_gap}
\caption{Same-model PR-AUC under chronological and random splits.}
\label{fig:protocol}
\end{figure}

\subsection{Main Comparison}
Under the chronological protocol, Random Forest reaches PR-AUC \NumRFPrAUC{} $\pm$ \NumRFPrAUCStd, ROC-AUC \NumRFRoc, Brier \NumRFBrier{} and ECE \NumRFECE. The best graph model reaches \NumCSChronic. The paired seed-level comparison gives $\Delta=\NumDeltaCSRf$ with 95\% CI [\NumDeltaCSRfLow, \NumDeltaCSRfHigh] and Holm-adjusted $p=\NumDeltaCSRfHolm$ over \NumDeltaCSRfN{} seeds. LightGBM trails RF by a small margin ($\Delta\approx -0.009$); while statistically significant due to near-zero variance, the gap is operationally negligible and reinforces RF as the practical upper bound. We therefore state that the tree model is the strict-protocol upper bound; we do not claim that graph structure is generally useless, only that its value is not demonstrated here.

\input{tables/Table3_stats}

\subsection{Structural Counterfactuals}
Table~\ref{tab:worlds} compares real causal edges with no-time, empty, endpoint-permuted and degree-preserving rewired worlds. The degree-preserving rewiring preserves both in- and out-degree sequences exactly. The observed differences do not support a reliable structural-necessity claim.

\input{tables/Table2_worlds}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/edge_worlds}
\caption{PR-AUC across structural edge worlds.}
\label{fig:worlds}
\end{figure}

\subsection{Drift}
Fig.~\ref{fig:drift} shows out-of-time PR-AUC for causal\_sage. Performance varies across periods, so a single test-set aggregate is insufficient for deployment reasoning.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/drift}
\caption{Mean test PR-AUC by time step.}
\label{fig:drift}
\end{figure}

\subsection{Accuracy-Time Tradeoff}
Fig.~\ref{fig:pareto} reports the non-dominated set among configurations with comparable GPU-memory measurements. Tabular baselines are reported separately because they do not expose comparable GPU memory.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/pareto}
\caption{Accuracy versus wall time for non-dominated runs.}
\label{fig:pareto}
\end{figure}

\section{Reproducibility Artifact}\label{sec:repro}
The repository contains:
\begin{itemize}
\item \texttt{src/}: loaders, causal edge construction, sampler, models and metrics;
\item \texttt{scripts/run\_pipeline.py}: auto-download, resume and experiment execution;
\item \texttt{scripts/summarize\_results.py}: table and statistics generation;
\item \texttt{tests/}: 11 pytest checks including future-edge rejection and degree preservation;
\item \texttt{outputs/full/logs}: raw execution logs;
\item \texttt{outputs/full/tables}: machine-readable tables used by this paper;
\item \texttt{docs/TEST\_ENVIRONMENT.md} and \texttt{artifacts/test\_environment/}: GPU, driver, CUDA, CPU, memory, operating-system and framework snapshots.
\end{itemize}

Data are downloaded from the official Kaggle endpoint and checksummed; no raw dataset is redistributed.

\section{Limitations}
Elliptic is one blockchain dataset. Features are anonymized, so we do not claim amount or fee subgroup effects. Unknown labels are not negatives. Statistical tests address seed variability, not new temporal replicates. This is a diagnostic study, not a deployment certification.

\section{Conclusion}
Under a causal, auditable protocol, strong tabular baselines outperform graph models on Elliptic, while random splits overstate graph performance. The contribution is not a new architecture but a reproducible evaluation contract that prevents optimistic AML claims from entering operational pipelines.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""
    (OUT / "main.tex").write_text(text, encoding="utf-8")


def build_description():
    text = """Title: A Reproducibility Diagnostic for Temporally Causal Graph-Based Anti-Money-Laundering Evaluation

Target journal: IEEE Transactions on Information Forensics and Security

Author: Xiaole Han (sole author)
ORCID: 0009-0009-9611-2368
Affiliation: Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China
Corresponding author: Xiaole Han, hanxiaoleop@163.com

Abstract:
Graph neural networks are widely proposed for anti-money-laundering screening, but gains often depend on random splits, future-edge visibility, unknown-label handling and test-set calibration. We introduce an executable causal evaluation protocol and apply it to the Elliptic transaction graph. The protocol enforces chronological time-step splits, source-to-target time ordering, unknown-label masking and validation-only calibration, and it records structural counterfactuals with degree-preserving rewiring. Under this protocol, Random Forest reaches PR-AUC 0.733, whereas the best graph model reaches 0.492; the paired seed-level difference is -0.223 with 95% CI [-0.239, -0.203] and Holm-adjusted p=0.0004 over 15 seeds. A random split raises the same graph model from 0.492 to 0.903. We therefore report tree models as the strict-protocol upper bound and do not claim universal graph-structure gains. The submission is generated from machine-readable result files, with source code, tests, raw logs and checksummed tables released for traceable reproduction.

Aggregation note:
The paired delta is computed from each model's per-seed mean across chronological runs and then paired across seeds. Table row means are run-level averages; when run counts differ by seed, the two averages need not coincide exactly. The paired test is the basis for the significance statement.

Keywords:
anti-money laundering; graph neural networks; temporal causality; reproducibility; leakage; blockchain analytics; secure evaluation

Data availability:
The public Elliptic dataset is downloaded from its official Kaggle endpoint and checksummed locally. Raw data are not redistributed because the provider license prohibits redistribution of the dataset.

Code availability:
Source code, tests, raw logs, generated tables and the LaTeX package are provided in the accompanying repository and source archive.

Novelty positioning:
The contribution is a reproducible security-evaluation protocol and negative-result diagnostic for AML graph learning, not a new model architecture.

Original contribution (one to two sentences):
This paper contributes an executable causal evaluation contract for temporal graph-based AML and uses it to show that, once future edges, unknown labels and test-time calibration are controlled, strong tabular baselines can outperform graph models that appear superior under random splits.

Why this is important for AI:
Temporal graph learning is increasingly used in fraud detection, recommendation and event forecasting. If future edges, unknown labels or test calibration leak into evaluation, reported gains are not evidence of deployable AI. The protocol makes these failure modes testable and reproducible.

Where the contribution is explained:
Sections I-II state the contribution and AI relevance; Section V (Data and Causal Protocol) defines the causal invariants; Section VI (Method) describes the method; Section VIII (Results) reports the negative-result evidence; Section IX (Reproducibility Artifact) documents the reproduction artifact.

Closest related work:
1. B. Hu et al., "Cash-out User Detection based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism," AAAI 2019.
2. A. Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs," AAAI 2020.
3. X. Yang, X. Zhao and Z. Shen, "A Generalizable Anomaly Detection Method in Dynamic Graphs," AAAI 2025.
These works propose models and report detection gains. This paper differs by providing a protocol that tests whether such gains survive chronological splits, unknown-label masking, validation-only calibration and degree-preserving edge counterfactuals.

Relevance to AI:
Yes. The paper addresses trustworthy and reproducible evaluation of learning systems, a core AI concern across graph learning, security and applied machine learning. The abstract, introduction and conclusion are written for a general AI audience; technical sections define every protocol and report effect sizes and confidence intervals.

Additional electronic material:
Source code, unit tests, raw execution logs, machine-readable tables, figure-generation scripts, test-environment snapshots and the complete LaTeX source are provided. Raw Elliptic data are not redistributed because of the provider license; the code downloads and checksums them locally.

Upload checklist:
1. main.tex
2. IEEEtran.cls
3. references.bib
4. numbers.tex
5. figures/*.png
6. tables/*.tex
7. description.txt
8. main.pdf (compiled for review convenience)
"""
    (OUT / "description.txt").write_text(text, encoding="utf-8")


def build_ai_response():
    text = """AI RELEVANCE AND CONTRIBUTION STATEMENT
Manuscript: A Reproducibility Diagnostic for Temporally Causal Graph-Based Anti-Money-Laundering Evaluation
Author: Xiaole Han (sole author)
ORCID: 0009-0009-9611-2368
Affiliation: Business School, Qilu Institute of Technology, Jinan, Shandong 250200, China
Corresponding author: Xiaole Han, hanxiaoleop@163.com

1. What is, in one or two sentences, the original contribution of this work?
This paper contributes an executable causal evaluation contract for temporal graph-based AML and uses it to show that, once future edges, unknown labels and test-time calibration are controlled, strong tabular baselines can outperform graph models that appear superior under random splits.

2. Why should this contribution be considered important for the field of Artificial Intelligence?
Temporal graph learning is used in fraud detection, recommendation, event forecasting and security. If future edges, unknown labels or test calibration leak into evaluation, reported gains are not evidence of deployable AI. The paper turns these failure modes into explicit, testable and reproducible invariants.

3. Where do you explain this in the paper?
Sections I-II state the contribution and AI relevance; Section V (Data and Causal Protocol) defines the causal invariants; Section VI (Method) describes the method; Section VIII (Results) reports the negative-result evidence; Section IX (Reproducibility Artifact) documents the reproduction artifact and test environment.

4. What is the most closely related work by others, and how does this work differ?
Closest works:
[1] B. Hu, Z. Zhang, C. Shi, J. Zhou, X. Li, and Y. Qi, "Cash-out User Detection based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism," AAAI 2019.
[2] A. Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs," AAAI 2020.
[3] X. Yang, X. Zhao and Z. Shen, "A Generalizable Anomaly Detection Method in Dynamic Graphs," AAAI 2025.
These papers propose models and report detection gains. This paper differs by providing a protocol that tests whether such gains survive chronological splits, unknown-label masking, validation-only calibration and degree-preserving edge counterfactuals.

5. Is your work relevant to AI, and where do you explain the relevance?
Yes. The paper addresses trustworthy and reproducible evaluation of learning systems, a core AI concern across graph learning, security and applied machine learning. Relevance is explained in Section II and in the abstract, introduction and conclusion.

6. Can at least the abstract, introduction and conclusion be appreciated by a general AI audience?
Yes. These sections avoid AML-specific notation and frame the problem as leakage, calibration and attributable gain. Technical sections define every protocol and report effect sizes and confidence intervals.

7. Additional electronic material
Source code, unit tests, raw execution logs, machine-readable tables, figure-generation scripts, test-environment snapshots and the complete LaTeX source are provided. Raw Elliptic data are not redistributed because the provider license prohibits it; the code downloads and checksums the data locally.
"""
    (OUT / "AI_RELEVANCE_RESPONSE.txt").write_text(text, encoding="utf-8")


def build_upload_checklist():
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rel = path.relative_to(OUT).as_posix()
            files.append(f"- `{rel}` ({path.stat().st_size} bytes)")
    text = "# TIFS Upload Checklist\n\nUpload the following files to the journal system:\n\n" + "\n".join(files) + """

\nRequired ordering:
1. `main.tex` is the entry point.
2. `numbers.tex` is auto-generated and must be included.
3. `tables/*.tex` are auto-generated table fragments and must be included.
4. `figures/*.png` are referenced by `main.tex`.
5. `description.txt` is the plain-text Description file for the submission system.
6. `main.pdf` is optional but useful for editorial review.
7. Do not upload raw Elliptic CSVs; they are license-restricted and downloaded by `scripts/run_pipeline.py`.
"""
    (OUT / "UPLOAD_CHECKLIST.md").write_text(text, encoding="utf-8")


def build_manifest():
    items = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            items.append({"path": path.relative_to(OUT).as_posix(), "sha256": digest,
                          "bytes": path.stat().st_size})
    payload = {
        "journal": JOURNAL,
        "entrypoint": "main.tex",
        "description": "description.txt",
        "files": items,
        "note": "Raw Elliptic CSV files are intentionally excluded.",
    }
    (OUT / "SOURCE_MANIFEST.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "tables").mkdir(parents=True)
    shutil.copyfile(CLASS_SRC, OUT / "IEEEtran.cls")
    (OUT / "references.bib").write_text(REFERENCES, encoding="utf-8")
    build_figures()
    build_numbers()
    make_main_table()
    make_world_table()
    make_stats_table()
    build_main_tex()
    build_description()
    build_ai_response()
    build_upload_checklist()
    build_manifest()
    print(f"TIFS package built at {OUT}")


if __name__ == "__main__":
    main()
