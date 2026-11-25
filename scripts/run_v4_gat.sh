#!/usr/bin/env bash
# V4 second graph baseline (GAT) -- resumable, logs preserved.
set -u
PY="C:/Users/apple/anaconda3/python.exe"
ROOT="D:/Financial Micro-Modelling Project/elliptic-causal-aml"
OUT="$ROOT/outputs/v4"; LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR"
PROGRESS="$OUT/progress_gat.txt"; touch "$PROGRESS"
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
for s in 0 1 2 3 4; do
  run "gat_baseline_s${s}" --model gat --seed "$s" --epochs 30 --batch-size 512 \
    --neighbors 10 10 --alpha 0.5 --calibration temperature --pos-weight-mode none
done
echo "GAT_BASELINE_DONE"
