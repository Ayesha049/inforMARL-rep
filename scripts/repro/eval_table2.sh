#!/bin/bash
# Evaluate every trained Table 2 policy at m in {3,7,10,15}, 100 episodes each,
# obstacles ~ U{0..10} per episode (paper Section 4.3), then build the table.
# usage: bash scripts/repro/eval_table2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONNOUSERSITE=1
ROOT=onpolicy/results/GraphMPE/navigation_graph/rmappo
mkdir -p results_repro/eval
for d in ${ROOT}/informarl_n*_s*/run*/models; do
  exp=$(basename "$(dirname "$(dirname "$d")")")
  [ -f "$d/actor.pt" ] || continue
  # only evaluate runs that trained to completion (a crashed run still leaves a checkpoint)
  LOG=logs/repro/train_${exp}.log
  if [ ! -f "$LOG" ] || grep -q Traceback "$LOG" || \
     ! grep "Percentage complete" "$LOG" | tail -1 | grep -qE "complete (99|100)"; then
    echo "== $exp: SKIPPED (training incomplete or failed)"; continue
  fi
  echo "== $exp"
  python onpolicy/scripts/eval_transfer.py --model_dir "$d" \
    --test_num_agents 3 7 10 15 --eval_episodes 100 \
    --out_json results_repro/eval/${exp}.json 2>&1 | grep -E "^m=|saved"
done
python scripts/repro/make_table2.py results_repro/eval > results_repro/table2.md
cat results_repro/table2.md
