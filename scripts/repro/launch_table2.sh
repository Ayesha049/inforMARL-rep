#!/bin/bash
# Launch all Table 2 training runs, in waves (runs inside a wave run in parallel,
# the next wave starts when the previous one has finished).
# Budgets: n=3 -> 2M (README, compare3.sh), n=7 -> 5M (informarl.sh),
#          n=10 -> STEPS_N10 (default 5M as in informarl_10n.sh; compare10.sh uses 20M).
#
# usage (detached):
#   SEEDS="1 2 3" WAVES="10,7 3" nohup setsid bash scripts/repro/launch_table2.sh \
#       > logs/repro/launcher.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
SEEDS=${SEEDS:-"1 2 3"}
WAVES=${WAVES:-"10,7 3"}
STEPS_N10=${STEPS_N10:-5000000}
declare -A STEPS=( [3]=2000000 [7]=5000000 [10]=${STEPS_N10} )
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -ge 1 ] || NGPU=1
mkdir -p logs/repro
i=0
for WAVE in ${WAVES}; do
  pids=()
  echo "$(date '+%F %T') wave [${WAVE}] starting"
  for N in ${WAVE//,/ }; do
    for S in ${SEEDS}; do
      GPU=$(( i % NGPU )); i=$(( i + 1 ))
      bash scripts/repro/train_one.sh ${N} ${S} ${STEPS[$N]} ${GPU} \
        > logs/repro/nohup_n${N}_s${S}.out 2>&1 &
      pids+=($!)
      echo "$(date '+%F %T') started n=${N} seed=${S} steps=${STEPS[$N]} gpu=${GPU} pid=$!"
    done
  done
  for p in "${pids[@]}"; do
    wait "$p"; echo "$(date '+%F %T') pid=$p exited with $?"
  done
  echo "$(date '+%F %T') wave [${WAVE}] finished"
done
echo "$(date '+%F %T') all waves finished"
