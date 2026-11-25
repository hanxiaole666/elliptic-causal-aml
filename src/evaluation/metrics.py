"""Primary, secondary and operational metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(score, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = indices == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return float(ece)


def subgroup_metrics(group: np.ndarray, y: np.ndarray, score: np.ndarray, q: int = 5, k_list=(50, 100)) -> dict:
    """Report performance by degree/amount quantile with explicit group sizes."""
    group = np.asarray(group, dtype=float)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(group) == 0:
        return {}
    edges = np.quantile(group, np.linspace(0, 1, q + 1))
    edges = np.unique(edges)
    result = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (group >= lo) & (group <= hi if i == len(edges) - 2 else group < hi)
        if not mask.any():
            continue
        result[f"q{i+1}"] = {
            "range": [float(lo), float(hi)],
            "n": int(mask.sum()),
            "positive_n": int(y[mask].sum()),
            **binary_metrics(y[mask], score[mask], k_list, 10, 0.5),
        }
    return result


def subgroup_feature_proxy(
    x: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    columns=(0, 1, 2, 3, 4),
    q: int = 5,
) -> dict:
    """Exploratory quintiles on anonymised local feature indices.

    Elliptic features are anonymised and have no public semantic labels, so
    these columns must NOT be called 'amount' or 'fee'. They are a sensitivity
    probe over local-attribute-like dimensions, not a fairness claim.
    """
    return {
        f"local_feature_{col}": subgroup_metrics(x[:, col], y, score, q, (50, 100))
        for col in columns
        if col < x.shape[1]
    }


def binary_metrics(
    y: np.ndarray,
    score: np.ndarray,
    k_list=(50, 100, 500),
    ece_bins: int = 10,
    threshold: float = 0.5,
) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    y, score = y[valid], score[valid]
    result: dict = {"n": int(len(y)), "positive_n": int(y.sum())}
    if len(np.unique(y)) < 2:
        result.update({key: float("nan") for key in [
            "pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "ece"
        ]})
        return result

    result["pr_auc"] = float(average_precision_score(y, score))
    result["roc_auc"] = float(roc_auc_score(y, score))
    pred = (score >= threshold).astype(int)
    result["f1"] = float(f1_score(y, pred, zero_division=0))
    result["recall"] = float(recall_score(y, pred, zero_division=0))
    result["precision"] = float(precision_score(y, pred, zero_division=0))
    result["brier"] = float(np.mean((score - y) ** 2))
    result["ece"] = expected_calibration_error(y, score, ece_bins)

    order = np.argsort(-score)
    for k in k_list:
        top = order[: min(int(k), len(y))]
        result[f"precision@{k}"] = float(y[top].mean()) if len(top) else float("nan")
    return result
