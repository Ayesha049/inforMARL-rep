# Reproduction of InforMARL, Table 2 (team-size transfer, Target task)

Fork of [nsidn98/InforMARL](https://github.com/nsidn98/InforMARL) (upstream commit `7d3c42a`), used to reproduce
Nayak et al., *Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation*, ICML 2023,
Table 2: InforMARL trained with n ∈ {3, 7, 10} agents and tested with m ∈ {3, 7, 10, 15} agents.

## What was changed

| File | Change | Why |
|---|---|---|
| `requirements_modern.txt`, `scripts/repro/setup_env.sh` | Python 3.10, torch 2.7.1 (CUDA 12.8), torch_geometric 2.6.1, numpy 1.26, gym 0.10.5, setuptools < 70 | Original pins (torch 1.8.1, PyG 2.0.4) do not install on current Python and cannot run on RTX 5090 / RTX 6000 Ada |
| `onpolicy/utils/graph_buffer.py` | `np.int` → `int` | `np.int` was removed in numpy 1.24 |
| `onpolicy/scripts/train_mpe.py` | Writes `config.json` (all training args) into the run directory | Lets evaluation rebuild the policy without wandb |
| `onpolicy/scripts/eval_transfer.py` (new) | Batched train-n / test-m evaluation with the paper's protocol | The repo only has a render script (one env, wandb config) |
| `scripts/repro/*.sh`, `scripts/repro/make_table2.py` (new) | Launch, evaluate and tabulate the runs | Reproducibility |

No change to the environment, model, loss or hyper-parameters.

## Protocol

**Training** (hyper-parameters from the README and `scripts/informarl*.sh`): RMAPPO + InforMARL GNN, `navigation_graph`
scenario, 3 obstacles, `graph_feat_type=relative`, local observations (`use_cent_obs=False`), 128 rollout threads,
episode length 25, PPO epochs 10, lr 7e-4, ValueNorm + PopArt, auto mini-batch with target size 128.

| n | Env steps | Source of the budget |
|---|---|---|
| 3 | 2M | README, `scripts/compare3.sh` |
| 7 | 5M | `scripts/informarl.sh`, `scripts/informarl_n.sh` |
| 10 | 20M | `scripts/compare10.sh` (`scripts/informarl_10n.sh` uses 5M) |

The paper does not state the training budget per n; these are the values in the authors' scripts.

Seeds: 1, 2, 3 (the paper uses 5 for its training curves; Table 2 does not state seeds).

**Evaluation** (`eval_transfer.py`, Section 4.3 of the paper): 100 episodes per test size m, deterministic actions,
number of obstacles drawn uniformly from {0, …, 10} per episode (the paper says "randomly chosen from (0, 10)"; we
include both ends). Metrics reuse the authors' helpers in `base_runner.py`:

- **Reward/m**: team episode reward / m (rewards are shared, `collaborative=True`)
- **T**: mean over agents of first-arrival time / episode length (1 if never reached)
- **(# col)/m**: agent-agent + agent-obstacle collision counts as the authors count them, / m
- **S%**: % of episodes in which every agent reaches its goal

## How to run

```bash
bash scripts/repro/setup_env.sh                       # once
SEEDS="1 2 3" bash scripts/repro/launch_table2.sh     # 9 background training runs
bash scripts/repro/eval_table2.sh                     # after training: results_repro/table2.md
```

## Notes

- Training time is dominated by the PPO update (about 95%), not the environment: the auto mini-batch setting gives
  75 mini-batches per epoch at n = 3 and 250 at n = 10, each a small GNN forward/backward. CPU and GPU give the same speed.
- `NaN or Inf found in input tensor` warnings come from tensorboardX logging the mean of an empty metric list; harmless.

## Results

See `results_repro/table2.md` once the runs finish.
