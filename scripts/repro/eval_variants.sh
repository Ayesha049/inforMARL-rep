#!/bin/bash
# Diagnose the Table 2 gap: re-evaluate every completed policy under protocol variants.
#   A: 3 obstacles (as in training), deterministic
#   B: obstacles ~ U{0..10} (Table 2 protocol), stochastic
#   C: 3 obstacles, stochastic (= training conditions)
# Test sizes: m = n and m = 7. Output: results_repro/eval_variants/<exp>_<variant>.json
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONNOUSERSITE=1
ROOT=onpolicy/results/GraphMPE/navigation_graph/rmappo
OUT=results_repro/eval_variants; mkdir -p $OUT
for d in ${ROOT}/informarl_n*_s*/run*/models; do
  exp=$(basename "$(dirname "$(dirname "$d")")")
  n=$(echo "$exp" | sed -E 's/informarl_n([0-9]+)_s.*/\1/')
  LOG=logs/repro/train_${exp}.log
  if [ ! -f "$LOG" ] || grep -q Traceback "$LOG" || \
     ! grep "Percentage complete" "$LOG" | tail -1 | grep -qE "complete (99|100)"; then
    echo "== $exp: SKIPPED (training incomplete or failed)"; continue
  fi
  ms=$(printf "%s\n" "$n" 7 | sort -nu | tr '\n' ' ')
  for v in A B C; do
    case $v in
      A) extra="--fixed_obstacles 3" ;;
      B) extra="--stochastic" ;;
      C) extra="--fixed_obstacles 3 --stochastic" ;;
    esac
    echo "== $exp variant $v (m = $ms)"
    python onpolicy/scripts/eval_transfer.py --model_dir "$d" --test_num_agents $ms \
      --eval_episodes 100 $extra --out_json $OUT/${exp}_${v}.json 2>&1 | grep -E "^m="
  done
done
echo "ALL VARIANTS DONE"
