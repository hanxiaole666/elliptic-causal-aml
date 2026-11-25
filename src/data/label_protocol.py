"""Label protocols A/B/C/D with explicit SCAR/SAR caveats."""
from __future__ import annotations

import numpy as np


def apply_label_protocol(
    y: np.ndarray,
    protocol: str,
    pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (effective_labels, loss_mask). Unknown is -1 in the raw labels."""
    y = np.asarray(y)
    if protocol == "A":
        mask = y >= 0
        return y.copy(), mask
    if protocol == "B":
        filled = np.where(y < 0, 0, y)
        return filled, np.ones_like(y, dtype=bool)
    if protocol == "C":
        if pred is None:
            # The pipeline will supply high-confidence pseudo-labels after a
            # temporary model is fitted; start from the known-label mask.
            return y.copy(), y >= 0
        filled = np.where(y < 0, (np.asarray(pred) >= threshold).astype(np.int64), y)
        mask = y >= 0
        return filled, mask
    if protocol == "D":
        # PU setting: positives are y==1; every other node (including known
        # licit and unknown) is treated as unlabeled for the nnPU risk.
        return y.copy(), np.ones_like(y, dtype=bool)
    raise ValueError(f"Unknown label protocol: {protocol}")


def pu_weights(y: np.ndarray, prior_positive: float, epsilon: float = 1e-6) -> np.ndarray:
    """Heuristic PU weights; valid only under SCAR. Under SAR this is a sensitivity probe."""
    y = np.asarray(y)
    weights = np.zeros(len(y), dtype=np.float32)
    weights[y == 1] = 1.0
    weights[y == -1] = max(epsilon, (1.0 - prior_positive) / max(epsilon, prior_positive))
    return weights


def unknown_distribution_audit(x: np.ndarray, time: np.ndarray, y: np.ndarray, n_cols: int = 5) -> dict:
    """Compare unknown versus known feature means on the first n_cols dimensions."""
    known = y >= 0
    unknown = ~known
    result = {"n_known": int(known.sum()), "n_unknown": int(unknown.sum()), "columns": {}}
    for col in range(min(n_cols, x.shape[1])):
        a, b = x[known, col], x[unknown, col]
        if len(a) == 0 or len(b) == 0:
            continue
        mean_delta = float(b.mean() - a.mean())
        pooled = float(np.sqrt((a.var() + b.var()) / 2 + 1e-12))
        result["columns"][str(col)] = {"mean_delta": mean_delta, "std_diff": mean_delta / pooled}
    return result
