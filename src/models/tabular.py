"""Feature-only baselines. They receive exactly the same features as the GNNs."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def build_tabular(name: str, seed: int):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32, 32),
            max_iter=40,
            early_stopping=True,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown tabular model: {name}")


def fit_tabular(name: str, x: np.ndarray, y: np.ndarray, seed: int):
    model = build_tabular(name, seed)
    model.fit(x, y)
    return model


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    positive = classes.index(1)
    return np.asarray(proba[:, positive], dtype=np.float64)
