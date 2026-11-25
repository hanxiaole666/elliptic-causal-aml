#!/usr/bin/env bash
set -u
PY="C:/Users/apple/anaconda3/python.exe"
ROOT="D:/Financial Micro-Modelling Project/elliptic-causal-aml"
OUT="$ROOT/outputs/full"
mkdir -p "$OUT/logs"

run() {
  local name="$1"; shift
  echo "[START] $name $(date '+%H:%M:%S')"
  "$PY" "$ROOT/scripts/run_pipeline.py" --data-mode elliptic --device auto --output-dir "$OUT" "$@" > "$OUT/logs/${name}.log" 2>&1
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo "[FAILED:$code] $name -- see logs/${name}.log"
  else
    echo "[DONE] $name $(date '+%H:%M:%S')"
  fi
}

echo "=== 1/8 protocol ladder (random split) ==="
for s in 0 1 2 3 4; do
  run "random_sage_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --split-mode random --epochs 10 --batch-size 512 --neighbors 5 5 --calibration none
  run "random_lgbm_s$s" --model lightgbm --seed "$s" --split-mode random --calibration temperature
done

echo "=== 2/8 main model, 5 seeds x 30 epochs ==="
for s in 0 1 2 3 4; do
  run "main_sage_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --epochs 30 --batch-size 512 --neighbors 10 10 --alpha 0.5 --degree-weight --calibration temperature
done

echo "=== 3/8 tabular baselines, 5 seeds ==="
for m in lr rf xgboost lightgbm mlp; do
  for s in 0 1 2 3 4; do
    run "tab_${m}_s$s" --model "$m" --seed "$s" --calibration temperature
  done
done

echo "=== 4/8 algorithm ablations, 5 seeds x 10 epochs ==="
for s in 0 1 2 3 4; do
  run "ablate_A0_s$s" --model graphsage --pos-weight-mode none --seed "$s" --alpha 0 --no-time-encoding --epochs 10 --batch-size 512 --neighbors 10 10 --calibration none
  run "ablate_A1_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --alpha 0 --no-time-encoding --epochs 10 --batch-size 512 --neighbors 10 10 --calibration none
  run "ablate_A2_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --alpha 0 --degree-weight --no-time-encoding --epochs 10 --batch-size 512 --neighbors 10 10 --calibration none
  run "ablate_A3_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --alpha 0.5 --degree-weight --epochs 10 --batch-size 512 --neighbors 10 10 --calibration temperature
  run "ablate_A4_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --alpha 0.5 --degree-weight --time-k 16 --epochs 10 --batch-size 512 --neighbors 10 10 --calibration temperature
done

echo "=== 5/8 edge worlds, 5 seeds x 10 epochs ==="
for w in real no_time empty perm rewire; do
  for s in 0 1 2 3 4; do
    run "world_${w}_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --edge-world "$w" --epochs 10 --batch-size 512 --neighbors 10 10 --calibration none
  done
done

echo "=== 6/8 label protocols, 5 seeds x 10 epochs ==="
for p in A B C D; do
  for s in 0 1 2 3 4; do
    run "label_${p}_s$s" --model causal_sage --pos-weight-mode none --seed "$s" --label-protocol "$p" --epochs 10 --batch-size 512 --neighbors 10 10 --calibration none
  done
done

echo "=== 7/8 robustness, attacks x aggregators, seed 0 x 5 epochs ==="
for a in edge_delete high_degree_delete feature_flip; do
  for g in mean median trim; do
    run "robust_${a}_${g}" --model causal_sage --pos-weight-mode none --seed 0 --attack "$a" --attack-budget 0.1 --aggregator "$g" --epochs 5 --batch-size 512 --neighbors 10 10 --calibration none
  done
done
run "robust_clean_median" --model causal_sage --pos-weight-mode none --seed 0 --aggregator median --epochs 5 --batch-size 512 --neighbors 10 10 --calibration none
run "robust_clean_trim" --model causal_sage --pos-weight-mode none --seed 0 --aggregator trim --epochs 5 --batch-size 512 --neighbors 10 10 --calibration none

echo "=== 8/8 Pareto configurations ==="
for nb in 5 20; do
  for h in 16 32; do
    run "pareto_nb${nb}_h${h}" --model causal_sage --pos-weight-mode none --seed 0 --hidden "$h" --neighbors "$nb" "$nb" --epochs 10 --batch-size 512 --alpha 0.5 --degree-weight --calibration temperature
  done
done

echo "=== summarize ==="
"$PY" "$ROOT/scripts/summarize_results.py" --output-dir "$OUT" > "$OUT/logs/summarize.log" 2>&1
echo "[ALL DONE] $(date '+%H:%M:%S')"
