#!/usr/bin/env python3
"""Third-round requirement: quantify 'within noise' between graph worlds.

Paired bootstrap CI on Delta(full - empty) and Delta(full - strict) over the
15 matched seeds, plus a TOST equivalence test with a pre-registered margin
epsilon = 0.02 PR-AUC. Also pairs against the plain MLP baseline.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "outputs" / "full" / "results"
V3 = ROOT / "outputs" / "v3" / "results"
OUT = ROOT / "outputs" / "v4" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    rows = []
    for d in (RESULTS, V3):
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("status") == "done" and r.get("dataset_version") == "elliptic":
                rows.append(r)
    return rows


def seed_scores(rows, pred):
    d = defaultdict(list)
    for r in rows:
        if pred(r.get("config", {}), r):
            d[r["seed"]].append(r["metrics"]["calibrated"]["pr_auc"])
    return {s: float(np.mean(v)) for s, v in d.items()}


def paired_analysis(a, b, eps=0.02, n_boot=5000, seed=0):
    common = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in common])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # TOST: H0: |delta| >= eps; reject if both one-sided tests pass at 0.05
    n = len(diff); se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    mean = diff.mean()
    t1 = (mean + eps) / se  # H0: mean <= -eps
    t2 = (mean - eps) / se  # H0: mean >= +eps
    p1 = 1 - stats.t.cdf(t1, n - 1)
    p2 = stats.t.cdf(t2, n - 1)
    p_tost = max(p1, p2)
    return {
        "n_seeds": n, "seeds": common,
        "mean_delta": float(mean), "std_delta": float(diff.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": float(lo), "ci_high": float(hi),
        "epsilon": eps, "tost_p": float(p_tost),
        "equivalent_at_epsilon": bool(p_tost < 0.05),
        "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if n > 1 else float("nan"),
    }


def main():
    rows = load()
    std = lambda c: (c.get("split_mode") == "chrono" and c.get("label_protocol", "A") == "A"
                     and c.get("attack", "none") == "none" and c.get("edge_world") == "real"
                     and c.get("canonical_split", "none") == "none" and c.get("feature_view", "all") == "all"
                     and c.get("hidden", 32) == 32 and c.get("layers", 2) == 2)
    full = seed_scores(rows, lambda c, r: r["model"] == "causal_sage" and std(c) and not c.get("strict_time", False))
    strict = seed_scores(rows, lambda c, r: r["model"] == "causal_sage" and c.get("strict_time", False))
    empty = seed_scores(rows, lambda c, r: r["model"] == "causal_sage" and c.get("edge_world") == "empty")
    mlp = seed_scores(rows, lambda c, r: r["model"] == "mlp" and c.get("split_mode") == "chrono")

    out = {
        "definition": "Delta = PR-AUC(full-graph CausalSAGE) - PR-AUC(comparison), paired on seed",
        "epsilon_pr_auc": 0.02,
        "full_vs_empty": paired_analysis(full, empty),
        "full_vs_strict": paired_analysis(full, strict),
        "full_vs_mlp": paired_analysis(full, mlp),
        "n_full": len(full), "n_empty": len(empty), "n_strict": len(strict), "n_mlp": len(mlp),
        "note": "full/empty/strict are feature-only vs message-passing comparisons; mlp is a separate model family.",
    }
    (OUT / "equivalence_tests.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: ({kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items() if kk in ("n_seeds","mean_delta","ci_low","ci_high","tost_p","equivalent_at_epsilon","wilcoxon_p")} if isinstance(v, dict) else v) for k, v in out.items() if k in ("full_vs_empty","full_vs_strict","full_vs_mlp","n_full","n_empty","n_strict","n_mlp")}, indent=2))
    print("saved", OUT / "equivalence_tests.json")


if __name__ == "__main__":
    main()
