#!/bin/bash
# Train one InforMARL policy (Target / navigation_graph task), reproduction of
# Nayak et al. (ICML 2023). Hyper-parameters are the README / scripts/informarl*.sh
# ones; only --num_agents, --seed and --num_env_steps vary.
#
# usage: bash scripts/repro/train_one.sh <num_agents> <seed> <num_env_steps> [gpu_id]
set -euo pipefail
N=$1; SEED=$2; STEPS=$3; GPU=${4:-0}
cd "$(dirname "$0")/../.."
mkdir -p logs/repro
EXP="informarl_n${N}_s${SEED}"
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONNOUSERSITE=1
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
  --project_name "informarl_repro" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed ${SEED} \
  --experiment_name "${EXP}" \
  --scenario_name "navigation_graph" \
  --num_agents ${N} \
  --collision_rew 5 \
  --n_training_threads 1 --n_rollout_threads 128 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps ${STEPS} \
  --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
  --user_name "marl" \
  --use_cent_obs "False" \
  --graph_feat_type "relative" \
  --auto_mini_batch_size --target_mini_batch_size 128 \
  --use_wandb \
  > logs/repro/train_${EXP}.log 2>&1
