#!/bin/bash
# Launch all Table 2 training runs in the background (one process per run).
# Budgets: n=3 -> 2M (README, compare3.sh), n=7 -> 5M (informarl.sh),
#          n=10 -> STEPS_N10 (default 20M as in compare10.sh; informarl_10n.sh uses 5M).
# usage: SEEDS="1 2 3" STEPS_N10=20000000 bash scripts/repro/launch_table2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SEEDS=${SEEDS:-"1 2 3"}
STEPS_N10=${STEPS_N10:-20000000}
declare -A STEPS=( [3]=2000000 [7]=5000000 [10]=${STEPS_N10} )
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}; [ "$NGPU" -ge 1 ] || NGPU=1
i=0
mkdir -p logs/repro
for N in 10 7 3; do
  for S in ${SEEDS}; do
    GPU=$(( i % NGPU ))
    nohup bash scripts/repro/train_one.sh ${N} ${S} ${STEPS[$N]} ${GPU} \
      > logs/repro/nohup_n${N}_s${S}.out 2>&1 &
    echo "started n=${N} seed=${S} steps=${STEPS[$N]} gpu=${GPU} pid=$!"
    i=$(( i + 1 ))
  done
done
