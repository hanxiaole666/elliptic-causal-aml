"""Explicit chronological split; no floating-point ratio is recomputed at train time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.io import atomic_write_json


@dataclass
class TimeSplit:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    train_times: list[int]
    valid_times: list[int]
    test_times: list[int]


def chronological_split(time: np.ndarray, ratios=(0.55, 0.2, 0.25)) -> TimeSplit:
    unique = np.sort(np.unique(time))
    if len(unique) < 3:
        raise ValueError("Temporal split needs at least 3 distinct time steps")
    n = len(unique)
    cut1 = max(1, int(round(n * ratios[0])))
    cut2 = max(cut1 + 1, int(round(n * (ratios[0] + ratios[1]))))
    cut2 = min(cut2, n - 1)
    train_times = unique[:cut1]
    valid_times = unique[cut1:cut2]
    test_times = unique[cut2:]

    train_idx = np.flatnonzero(np.isin(time, train_times))
    valid_idx = np.flatnonzero(np.isin(time, valid_times))
    test_idx = np.flatnonzero(np.isin(time, test_times))
    split = TimeSplit(
        train_idx, valid_idx, test_idx,
        [int(t) for t in train_times],
        [int(t) for t in valid_times],
        [int(t) for t in test_times],
    )
    validate_split(split)
    return split


def canonical_split(time: np.ndarray, scheme: str = "1-34_35-42") -> TimeSplit:
    """Canonical Elliptic splits used in prior literature.

    ``1-34_35-42``: train on time steps 1..34, validate on an inner tail of
    train, test on 35..42. ``1-34_35-49``: test on 35..49. Time steps in the
    raw data are 1-based; step 0 exists but has no labels and is kept in train.
    """
    unique = np.sort(np.unique(time))
    if scheme == "1-34_35-42":
        train_times = [int(t) for t in unique if 0 <= t <= 34]
        test_times = [int(t) for t in unique if 35 <= t <= 42]
    elif scheme == "1-34_35-49":
        train_times = [int(t) for t in unique if 0 <= t <= 34]
        test_times = [int(t) for t in unique if 35 <= t <= 49]
    elif scheme == "1-34_35-49-strict":
        # Contract-compliant canonical variant: disjoint train/val/test.
        train_times = [int(t) for t in unique if 0 <= t <= 34]
        valid_times = [int(t) for t in unique if 35 <= t <= 42]
        test_times = [int(t) for t in unique if 43 <= t <= 49]
        train_idx = np.flatnonzero(np.isin(time, train_times))
        valid_idx = np.flatnonzero(np.isin(time, valid_times))
        test_idx = np.flatnonzero(np.isin(time, test_times))
        split = TimeSplit(
            train_idx, valid_idx, test_idx,
            [int(t) for t in train_times], [int(t) for t in valid_times], [int(t) for t in test_times],
        )
        validate_split(split)
        return split
    else:
        raise ValueError(f"Unknown canonical scheme: {scheme}")
    train_sorted = sorted(train_times)
    cut = max(1, int(round(len(train_sorted) * 0.8)))
    valid_times = train_sorted[cut:]
    train_times = train_sorted[:cut]
    train_idx = np.flatnonzero(np.isin(time, train_times))
    valid_idx = np.flatnonzero(np.isin(time, valid_times))
    test_idx = np.flatnonzero(np.isin(time, test_times))
    split = TimeSplit(
        train_idx, valid_idx, test_idx,
        [int(t) for t in train_times], [int(t) for t in valid_times], [int(t) for t in test_times],
    )
    validate_split(split)
    return split


def validate_split(split: TimeSplit) -> None:
    sets = [set(split.train_times), set(split.valid_times), set(split.test_times)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("time-step sets overlap")
    if len(split.train_idx) == 0 or len(split.valid_idx) == 0 or len(split.test_idx) == 0:
        raise AssertionError("one temporal partition is empty")


def random_split(time: np.ndarray, ratios=(0.55, 0.2, 0.25), seed: int = 0) -> TimeSplit:
    """Node-level random split. This is the biased protocol used as a ladder baseline."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(time))
    n = len(time)
    n_train = int(round(n * ratios[0]))
    n_valid = int(round(n * ratios[1]))
    train_idx = np.sort(order[:n_train])
    valid_idx = np.sort(order[n_train : n_train + n_valid])
    test_idx = np.sort(order[n_train + n_valid :])
    split = TimeSplit(
        train_idx,
        valid_idx,
        test_idx,
        [int(t) for t in np.unique(time[train_idx])],
        [int(t) for t in np.unique(time[valid_idx])],
        [int(t) for t in np.unique(time[test_idx])],
    )
    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise AssertionError("one random partition is empty")
    if len(set(split.train_idx) | set(split.valid_idx) | set(split.test_idx)) != n:
        raise AssertionError("random split does not cover all nodes")
    return split


def save_split_manifest(path: str, split: TimeSplit) -> None:
    atomic_write_json(path, {
        "train_times": split.train_times,
        "valid_times": split.valid_times,
        "test_times": split.test_times,
        "n_train_nodes": len(split.train_idx),
        "n_valid_nodes": len(split.valid_idx),
        "n_test_nodes": len(split.test_idx),
    })
