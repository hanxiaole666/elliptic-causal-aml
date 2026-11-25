#!/usr/bin/env bash
# V5: ONE pre-registered configuration (C1) for every protocol rung, 15 seeds.
# Resumable + preserved timestamped logs.
set -u
PY="C:/Users/apple/anaconda3/python.exe"
ROOT="D:/Financial Micro-Modelling Project/elliptic-causal-aml"
OUT="$ROOT/outputs/v5"
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

# C1 standard: chrono, protocol A, hidden32/2layers/alpha0.5/degree/time-enc/temperature/30ep
COMMON="--model causal_sage --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight --pos-weight-mode none"
# Check the flag names exist; adjust if time-encoding is default-on.
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  run "c1_random_s${s}" --split-mode random $COMMON --calibration temperature --seed "$s"
  run "c1_unknown_s${s}" --label-protocol B $COMMON --calibration temperature --seed "$s"
  run "c1_testcal_s${s}" $COMMON --calibration none --seed "$s"
  run "c1_strict_s${s}" --strict-time $COMMON --calibration temperature --seed "$s"
  run "c1_empty_s${s}" --edge-world empty $COMMON --calibration temperature --seed "$s"
done

echo "V5_MATRIX_DONE ($(wc -l < "$PROGRESS") configs completed)"
