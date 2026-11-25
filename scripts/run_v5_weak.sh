#!/usr/bin/env bash
# V5 Threat-A direct evidence: weak models under permissive protocols.
set -u
PY="C:/Users/apple/anaconda3/python.exe"
ROOT="D:/Financial Micro-Modelling Project/elliptic-causal-aml"
OUT="$ROOT/outputs/v5"; LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR"
PROGRESS="$OUT/progress_weak.txt"; touch "$PROGRESS"
run() {
  local tag="$1"; shift
  if grep -qx "$tag" "$PROGRESS"; then echo "[SKIP-DONE] $tag"; return 0; fi
  local log="$LOGDIR/$tag.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $tag" >> "$log"
  "$PY" "$ROOT/scripts/run_pipeline.py" --data-mode elliptic --device auto --output-dir "$OUT" "$@" >> "$log" 2>&1
  local rc=$?; echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $tag rc=$rc" >> "$log"
  if [ "$rc" -eq 0 ]; then echo "$tag" >> "$PROGRESS"; echo "[OK] $tag"; else echo "[FAIL] $tag"; fi
}
for model in lr mlp; do
  for s in 0 1 2 3 4; do
    run "weak_${model}_random_s${s}" --model "$model" --split-mode random --seed "$s" --calibration temperature --pos-weight-mode none
    run "weak_${model}_unknown_s${s}" --model "$model" --label-protocol B --seed "$s" --calibration temperature --pos-weight-mode none
  done
done
echo "WEAK_DONE"
