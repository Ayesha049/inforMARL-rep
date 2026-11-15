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
| 10 | 5M | `scripts/informarl_10n.sh` (`scripts/compare10.sh` uses 20M) |

The paper's text does not state the budget, but the x-axes of its Figure 3 end at 2M (n=3), 4M (n=7) and 5M (n=10),
so our budgets equal the paper's for n=3 and n=10 and exceed it for n=7.

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
SEEDS="1 2 3" WAVES="10,7 3" nohup setsid bash scripts/repro/launch_table2.sh \
    > logs/repro/launcher.log 2>&1 &                # 9 runs: wave 1 = n=10,7; wave 2 = n=3
bash scripts/repro/eval_table2.sh                     # after training: results_repro/table2.md
```

## Notes

- Training time is dominated by the PPO update (about 95%), not the environment: the auto mini-batch setting gives
  75 mini-batches per epoch at n = 3 and 250 at n = 10, each a small GNN forward/backward. CPU and GPU give the same speed.
- `NaN or Inf found in input tensor` warnings come from tensorboardX logging the mean of an empty metric list; harmless.

## Results

Full table (ours vs paper, mean ± std over 3 seeds): [`results_repro/table2.md`](results_repro/table2.md).
All 9 runs trained to completion in fp32 (no non-finite gradient steps in any log).

| Train n | Reward/m vs paper | Time T vs paper | Success vs paper | Collisions/agent vs paper |
|---|---|---|---|---|
| 3 (2M steps) | within about 1 at m ≥ 7 (60.7 vs 61.2, 52.5 vs 53.2); 66.6 vs 63.2 at m=3 | within 0.02 | 98-100% vs 99-100% | about 2x higher (see below) |
| 7 (5M) | 5-7 lower (55.2 vs 62.2 at m=7) | within 0.02 | 93-100% vs 99-100% | about 2x higher (see below) |
| 10 (5M) | 6-8 lower (53.5 vs 61.3 at m=7) | within 0.04 | 86% at m=15 vs 99%, 93% at m=7 | about 2x higher (see below) |

**Reading.** The qualitative claim reproduces: a policy trained with n agents runs zero-shot at 3-15 agents with high
success, and the n=3 policy matches the paper closely on reward, time and success. The n=7 and n=10 policies are
weaker than the paper's, most at m=15 (success 86% for n=10). The paper does not state its training budget; our
n=7/10 budget (5M, `scripts/informarl_10n.sh`) may simply be too small (`scripts/compare10.sh` uses 20M). Not tested.

**Collisions (split by type, see the breakdown at the end of `results_repro/table2.md`).** Ours is about 2x the paper
in every cell (n=3: 0.81 / 1.28 / 1.76 / 2.62 against 0.40 / 0.74 / 0.95 / 1.28 at m = 3 / 7 / 10 / 15). Splitting our count:
agent-agent collisions grow with m (0.24 -> 2.04 per agent) while agent-obstacle collisions stay at about 0.55 per agent.
The code counts each agent-agent collision once per agent involved. Halving only that part gives a slope that matches the
paper (n=3, m=3 -> 15: +0.90 ours vs +0.88 paper), but a constant offset remains (0.69 vs 0.40 at m=3): our obstacle
collisions (about 0.55) are above the paper's implied constant (about 0.3). The cause of that offset is open; it could be
obstacle size or placement, the obstacle-count distribution (paper: "randomly chosen from (0, 10)"), or how the paper
counts obstacle collisions. Not tested. Re-running the evaluation reproduces the other metrics exactly (max difference 0).

**Training curves vs Figure 3** ([`results_repro/fig3/fig3_compare.png`](results_repro/fig3/fig3_compare.png)).
The paper's InforMARL mean curves were digitized from the PDF (`scripts/repro/digitize_fig3.py`, calibrated on the
gridlines: residual ≤ 0.04% of the x-range and ≤ 0.4 reward); the digitized start values (−85, −212, −316) match the first
logged values of our runs. Ours: mean ± std over 3 seeds, 5-point moving average.

| n | Paper at end of its x-axis | Ours at the same step | Steps to 90% of the paper's gain (paper / ours) |
|---|---|---|---|
| 3 | 185 at 2M | 190 ± 2 | 0.95M / 0.50M |
| 7 | 339 at 4M | 356 ± 10 | 2.57M / 2.13M |
| 10 | 425 at 5M | 418 ± 16 | 2.77M / 3.75M |

Final training reward matches the paper for every n. n=3 and n=7 learn as fast or faster; n=10 is slower in the middle
(it stays near −220 until about 1M steps; at 1.25M ours is −83 ± 17 against the paper's 195) and catches up by 5M.
Since the n=10 training curve reaches the paper's final value, the n=10 gap in Table 2 is probably not under-training;
it may come from the evaluation (deterministic actions, 0-10 obstacles) rather than the policy. Not tested.
Caveats: the paper's curve smoothing is unknown, its std band is not digitized, and we have 3 seeds against its 5.

Not claimed: that the implementation is "very close" to the original. Table 2 is near its ceiling (success about 100%),
so matching it is weak evidence; the paper's Figure 3 training curves are the stricter comparison and have not been made.

