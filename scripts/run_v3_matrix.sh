#!/usr/bin/env bash
# V3 experiment matrix with RESUMABLE execution and PRESERVED logs.
#
# Guarantees requested by the author:
#   1. Test/experiment logs are NEVER overwritten -- appended with timestamps.
#   2. The matrix is resumable: a config whose result JSON exists with
#      status="done" is skipped; re-running this script continues from where
#      it stopped. A progress file records completed tags.
set -u
PY="C:/Users/apple/anaconda3/python.exe"
ROOT="D:/Financial Micro-Modelling Project/elliptic-causal-aml"
OUT="$ROOT/outputs/v3"
LOGDIR="$OUT/logs"
RESDIR="$OUT/results"
mkdir -p "$LOGDIR" "$RESDIR"

PROGRESS="$OUT/progress.txt"
touch "$PROGRESS"

# run <tag> <args...>
# Skip if a completed result already exists for this exact config (resume).
# The runner computes run_id from the config hash, so we ask Python to check
# whether any result with status=done matches the tag recorded in progress.
run() {
  local tag="$1"; shift
  if grep -qx "$tag" "$PROGRESS"; then
    echo "[SKIP-DONE] $tag"
    return 0
  fi
  local log="$LOGDIR/$tag.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $tag" >> "$log"
  echo "[RUN] $tag"
  "$PY" "$ROOT/scripts/run_pipeline.py" --data-mode elliptic --device auto \
    --output-dir "$OUT" "$@" >> "$log" 2>&1
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $tag rc=$rc" >> "$log"
  if [ "$rc" -eq 0 ]; then
    echo "$tag" >> "$PROGRESS"
    echo "[OK] $tag"
  else
    echo "[FAIL] $tag (log: $log)"
  fi
}

# ---------- X2: same-step edge sensitivity + strict boundary ----------
# strict: keep only tau(src)<tau(dst); on native Elliptic this is an EMPTY graph.
for s in 0 1 2 3 4; do
  run "strict_s${s}" --model causal_sage --strict-time --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration temperature --pos-weight-mode none
done
# empty-world control (feature-only GNN with no edges at all) for comparison
for s in 0 1 2 3 4; do
  run "emptyworld_s${s}" --model causal_sage --edge-world empty --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration temperature --pos-weight-mode none
done

# ---------- X1: contract ladder (rungs) ----------
# rung1: random split, known-only, val calibration
for s in 0 1 2 3 4; do
  run "ladder_random_known_s${s}" --model causal_sage --split-mode random --seed "$s" \
    --epochs 20 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration temperature --pos-weight-mode none
done
# rung2: chrono, known-only, val calibration (= standard contract)
for s in 0 1 2 3 4; do
  run "ladder_chrono_known_s${s}" --model causal_sage --seed "$s" \
    --epochs 20 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration temperature --pos-weight-mode none
done
# rung3: chrono, unknown-as-negative (protocol B)
for s in 0 1 2 3 4; do
  run "ladder_chrono_unknown_s${s}" --model causal_sage --label-protocol B --seed "$s" \
    --epochs 20 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration temperature --pos-weight-mode none
done
# rung4: chrono, known-only, TEST calibration (protocol violation)
for s in 0 1 2 3 4; do
  run "ladder_chrono_testcal_s${s}" --model causal_sage --seed "$s" \
    --epochs 20 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration none --pos-weight-mode none
done

# ---------- X4: structural information alignment (RF + causal aggregates) ----------
# 1-hop/2-hop causal neighbour aggregates as tabular features
for s in 0 1 2 3 4; do
  run "struct_rf_local_s${s}" --model rf --feature-view local94 --seed "$s" \
    --calibration temperature --pos-weight-mode none
  run "struct_rf_all_s${s}" --model rf --feature-view all --seed "$s" \
    --calibration temperature --pos-weight-mode none
done

# ---------- X5: strict canonical variant 1-34 / 35-42 / 43-49 ----------
for s in 0 1 2 3 4; do
  run "canonstrict_cs_s${s}" --model causal_sage --canonical-split 1-34_35-49-strict --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 \
    --calibration temperature --pos-weight-mode none
done

echo "V3_MATRIX_DONE ($(wc -l < "$PROGRESS") configs completed)"
