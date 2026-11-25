#!/usr/bin/env bash
# V4 experiments: resumable + preserved logs (same guarantees as v3).
set -u
PY="C:/Users/apple/anaconda3/python.exe"
ROOT="D:/Financial Micro-Modelling Project/elliptic-causal-aml"
OUT="$ROOT/outputs/v4"
LOGDIR="$OUT/logs"; RESDIR="$OUT/results"
mkdir -p "$LOGDIR" "$RESDIR"
PROGRESS="$OUT/progress.txt"; touch "$PROGRESS"

run() {
  local tag="$1"; shift
  if grep -qx "$tag" "$PROGRESS"; then echo "[SKIP-DONE] $tag"; return 0; fi
  local log="$LOGDIR/$tag.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $tag" >> "$log"
  echo "[RUN] $tag"
  "$PY" "$ROOT/scripts/run_pipeline.py" --data-mode elliptic --device auto \
    --output-dir "$OUT" "$@" >> "$log" 2>&1
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $tag rc=$rc" >> "$log"
  if [ "$rc" -eq 0 ]; then echo "$tag" >> "$PROGRESS"; echo "[OK] $tag"; else echo "[FAIL] $tag"; fi
}

# Match the main comparison budget (30 epochs) for a directly comparable ladder.
for s in 0 1 2 3 4; do
  run "ladder30_random_s${s}" --model causal_sage --split-mode random --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight \
    --calibration temperature --pos-weight-mode none
  run "ladder30_chrono_s${s}" --model causal_sage --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight \
    --calibration temperature --pos-weight-mode none
  run "ladder30_unknown_s${s}" --model causal_sage --label-protocol B --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight \
    --calibration temperature --pos-weight-mode none
  run "ladder30_testcal_s${s}" --model causal_sage --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight \
    --calibration none --pos-weight-mode none
done

# Class-imbalance parity: GNN with positive reweighting (sqrt + balanced)
for s in 0 1 2 3 4; do
  run "gnn_sqrt_s${s}" --model causal_sage --pos-weight-mode sqrt --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight \
    --calibration temperature
  run "gnn_balanced_s${s}" --model causal_sage --pos-weight-mode balanced --seed "$s" \
    --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight \
    --calibration temperature
done

echo "V4_MATRIX_DONE ($(wc -l < "$PROGRESS") configs completed)"
