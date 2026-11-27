# elliptic-causal-aml

Reproducibility artifact for the manuscript

This repository contains the **programs, configuration registry, machine-readable
result tables, and the test/run logs** behind every number in the manuscript.
The LaTeX source of the paper and the review/response documents are **not** in
this repository; the LaTeX source is supplied separately with the submission.

The central finding is deliberately modest: on the public Elliptic graph, strict
temporal filtering collapses at the *released step resolution*, and no
attributable structure gain is demonstrated. Predicate I6 returns **fail** on
144 of 165 model feature columns (their observation time, block 575,059,
postdates the test window).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Data loaders, causal edge construction, temporal sampler, models, metrics, calibration, statistics |
| `scripts/run_pipeline.py` | Single-config experiment runner (auto-download, resume, checksums) |
| `scripts/run_v5_matrix.sh`, `run_v5_weak.sh` | 75-config and weak-model matrices (resumable, timestamped logs) |
| `scripts/i6_feature_time_audit.py` | I6 auditor: 21 pass / 144 fail, per-column declaration CSV, column-index assertion (21+72+72=165) |
| `scripts/recall_at_fpr.py` | Recall@1/5/10% FPR recomputed from a registered checkpoint (no retraining) |
| `results/results_v11/` | 93-column local-block ablation rebuttal (RF, 5 seeds) |
| `scripts/recall_at_fpr.py` | Recall@1/5/10% FPR recomputed from a registered checkpoint (no retraining) |
| `scripts/tost_power_analysis.py` | Power analysis explaining why the equivalence test was dropped |
| `scripts/verify_v11.py` | 119 consistency checks (numeric ranges, cross-table identity, cell-text integrity, PDF text layer) |
| `tests/` | Unit tests for leakage predicates, metrics, calibration and splits |
| `configs/registry.json` | Configuration registry: one registered configuration per reported number |
| `results/tables_v5|v6|v7/` | Machine-readable single-source tables and audit records |
| `results/manifests_v5/` | Split/edge manifests for the registered configurations |
| `logs/run_v5/`, `logs/run_full/` | Raw experiment run logs (v5 and full matrices) |
| `logs/run_v11/` | 93-column ablation run logs |
| `logs/pytest-test-log.txt` | Output of `python -m pytest tests -q` (11 passed) |

## Reproducing the main numbers

```bash
pip install torch==2.2.1 torch-geometric==2.8.1 scikit-learn pandas numpy scipy
python scripts/run_pipeline.py --data-mode elliptic --model causal_sage \
  --seed 0 --epochs 30 --split-mode chrono --calibration temperature \
  --keep-checkpoint --output-dir outputs/v5
bash scripts/run_v5_matrix.sh           # 15 seeds x 5 protocol rungs, resumable
python scripts/build_v5_table.py        # single-source result table
python scripts/i6_feature_time_audit.py # I6 verdict
python scripts/recall_at_fpr.py         # operational metrics from checkpoint
python scripts/verify_v11.py            # full consistency verification
python -m pytest tests -q               # 11 passed
```

Key numbers (n = 15 unless noted):

| Quantity | Value |
|---|---|
| Random Forest PR-AUC (I1–I5, protocol A) | 0.733 |
| CausalSAGE PR-AUC (same protocol) | 0.521 |
| Paired Δ (CausalSAGE − RF) | −0.212 |
| CausalSAGE PR-AUC, random partition | 0.911 |
| TGAT-style baseline / GAT baseline | 0.387±0.037 / 0.382±0.063 (n=5) |
| I6 audit verdict | **fail** (144/165 = 72 address + 72 neighbourhood columns) |
| Recall @ 1% / 5% / 10% FPR | 0.419 / 0.565 / 0.616 |

## Data

The Elliptic dataset is **not redistributed**. It is downloaded from its official
Kaggle endpoint and verified against SHA256 checksums by `src/data/download.py`.
License: CC BY-NC-ND 4.0. Download logic and checksums are included; data files are not.

## Verification status

- `python scripts/verify_v11.py` — 119/119 checks pass
- `python -m pytest tests -q` — 11 passed (log in `logs/pytest-test-log.txt`)

## Honest scope

Single-benchmark diagnostic on Elliptic. No cross-dataset claim is made. A full
TGN memory module, a second real dataset, and numerical verification of
within-step block times are explicit future work. Two earlier candidate results
(a pseudo-random "recovered" within-step retention rate and an equivalence test)
were withdrawn after being shown uninformative; their withdrawal is documented
in the submission rather than hidden.
