"""Validation-only calibration; the test set is never used to fit parameters."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    z = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(log_t: float) -> float:
        p = _sigmoid(logits / max(1e-6, np.exp(log_t)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(result.x))


def apply_calibration(kind: str, val_logits, val_y, test_logits):
    kind = (kind or "none").lower()
    if kind == "none":
        return _sigmoid(np.asarray(test_logits)), {"kind": "none"}

    val_logits = np.asarray(val_logits, dtype=float)
    test_logits = np.asarray(test_logits, dtype=float)
    val_y = np.asarray(val_y, dtype=int)

    if kind == "temperature":
        temperature = fit_temperature(val_logits, val_y)
        return _sigmoid(test_logits / temperature), {"kind": "temperature", "temperature": temperature}

    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(_sigmoid(val_logits), val_y)
        return np.asarray(model.predict(_sigmoid(test_logits)), dtype=float), {
            "kind": "isotonic",
            "n_thresholds": int(len(model.X_thresholds_)),
        }

    raise ValueError(f"Unknown calibration kind: {kind}")
