#!/usr/bin/env python
"""
Team-size transfer evaluation for InforMARL (reproduction of Table 2 in
Nayak et al., ICML 2023).

A policy trained with n agents is loaded and tested with m agents. Following
Section 4.3 of the paper, each test cell runs 100 episodes and the number of
obstacles is drawn at random at the start of each episode (here: an integer
uniformly in [obs_min, obs_max], default [0, 10]).

Metrics follow the paper and reuse the authors' helpers in base_runner.py:
  • Reward/m   : team episode reward divided by m (rewards are shared when
                 `collaborative=True`, so the per-agent reward equals the team
                 reward; dividing by m gives the Table 2 normalisation)
  • T          : mean over agents of (first time at goal) / (episode length),
                 with T = 1 for agents that never reach the goal
  • (# col)/m  : agent-agent + agent-obstacle collision counts as counted by
                 the authors (agent-agent counted once per agent involved)
                 divided by m
  • S%         : percentage of episodes in which *all* agents reach their goal

Usage:
python onpolicy/scripts/eval_transfer.py \
    --model_dir results/GraphMPE/navigation_graph/rmappo/informarl_n3_s1/run1/models \
    --test_num_agents 3 7 10 15 --eval_episodes 100 --out_json eval.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.config import get_config, graph_config
from multiagent.MPE_env import GraphMPEEnv
from onpolicy.envs.env_wrappers import GraphDummyVecEnv
from onpolicy.scripts.train_mpe import parse_args as train_parse_args


def _t2n(x):
    return x.detach().cpu().numpy()


def load_train_args(model_dir: Path) -> dict:
    """Training arguments saved by train_mpe.py next to the run directory."""
    for cand in [model_dir / "config.json", model_dir.parent / "config.json"]:
        if cand.exists():
            with open(cand) as f:
                return json.load(f)
    raise FileNotFoundError(f"No config.json found in {model_dir} or its parent")


# args that must come from the evaluation, not from the training config
EVAL_OVERRIDES = ("num_agents", "num_obstacles", "seed", "model_dir")


def build_args(cli: argparse.Namespace, m: int, num_obstacles: int, n_envs: int):
    parser = get_config()
    args, parser = train_parse_args([], parser)
    args, _ = graph_config([], parser)
    train_cfg = load_train_args(Path(cli.model_dir))
    for k, v in train_cfg.items():
        if k in EVAL_OVERRIDES:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    args.num_agents = m
    args.num_obstacles = num_obstacles
    args.model_dir = cli.model_dir
    args.seed = cli.seed
    args.cuda = False
    args.use_wandb = False
    args.use_render = True  # only restore the actor
    args.save_gifs = False
    args.n_rollout_threads = n_envs
    args.n_eval_rollout_threads = n_envs
    args.n_render_rollout_threads = n_envs
    return args


def make_envs(args, n_envs: int, seed_base: int):
    def get_env_fn(rank):
        def init_env():
            env = GraphMPEEnv(args)
            env.seed(seed_base + rank)
            return env

        return init_env

    return GraphDummyVecEnv([get_env_fn(i) for i in range(n_envs)])


def to_env_actions(actions, action_space):
    if action_space.__class__.__name__ == "MultiDiscrete":
        for i in range(action_space.shape):
            uc = np.eye(action_space.high[i] + 1)[actions[:, :, i]]
            actions_env = uc if i == 0 else np.concatenate((actions_env, uc), axis=2)
        return actions_env
    if action_space.__class__.__name__ == "Discrete":
        return np.squeeze(np.eye(action_space.n)[actions], 2)
    raise NotImplementedError


@torch.no_grad()
def run_group(cli, m: int, num_obstacles: int, n_eps: int, seed_base: int):
    """Run `n_eps` episodes in parallel with m agents and k obstacles."""
    from onpolicy.runner.shared.graph_mpe_runner import GMPERunner

    args = build_args(cli, m, num_obstacles, n_eps)
    envs = make_envs(args, n_eps, seed_base)
    runner = GMPERunner(
        {
            "all_args": args,
            "envs": envs,
            "eval_envs": None,
            "num_agents": m,
            "device": torch.device("cpu"),
            "run_dir": None,
        }
    )
    policy = runner.trainer.policy
    runner.trainer.prep_rollout()

    obs, agent_id, node_obs, adj = envs.reset()
    rnn_states = np.zeros(
        (n_eps, m, runner.recurrent_N, runner.hidden_size), dtype=np.float32
    )
    masks = np.ones((n_eps, m, 1), dtype=np.float32)
    ep_rewards = np.zeros((n_eps, m), dtype=np.float64)
    infos = None
    for _ in range(args.episode_length):
        action, rnn_states = policy.act(
            np.concatenate(obs),
            np.concatenate(node_obs),
            np.concatenate(adj),
            np.concatenate(agent_id),
            np.concatenate(rnn_states),
            np.concatenate(masks),
            deterministic=True,
        )
        actions = np.array(np.split(_t2n(action), n_eps))
        rnn_states = np.array(np.split(_t2n(rnn_states), n_eps))
        actions_env = to_env_actions(actions, envs.action_space[0])
        obs, agent_id, node_obs, adj, rewards, dones, infos = envs.step(actions_env)
        ep_rewards += np.asarray(rewards, dtype=np.float64).reshape(n_eps, m)
        rnn_states[dones == True] = 0.0
        masks = np.ones((n_eps, m, 1), dtype=np.float32)
        masks[dones == True] = 0.0
    envs.close()

    results = []
    for e in range(n_eps):
        env_infos = runner.process_infos([infos[e]])
        frac, success = runner.get_fraction_episodes(env_infos)
        collisions = runner.get_collisions(env_infos)
        per_agent_rew = float(np.mean(ep_rewards[e]))
        results.append(
            {
                "num_obstacles": num_obstacles,
                # same quantity as `rewards_arr` in GMPERunner.render
                "reward_per_agent_mean": per_agent_rew,
                "team_reward": float(np.sum(ep_rewards[e]))
                if not args.collaborative
                else per_agent_rew,
                "frac_episode": float(np.mean(frac)),
                "success": int(success),
                "collisions": float(collisions),
            }
        )
    return results


def summarise(m: int, eps: list) -> dict:
    team = np.array([e["team_reward"] for e in eps])
    frac = np.array([e["frac_episode"] for e in eps])
    succ = np.array([e["success"] for e in eps])
    col = np.array([e["collisions"] for e in eps])
    return {
        "m": m,
        "episodes": len(eps),
        "reward_per_m": float(np.mean(team) / m),
        "T": float(np.mean(frac)),
        "col_per_m": float(np.mean(col) / m),
        "S_pct": float(100.0 * np.mean(succ)),
        "reward_per_m_std": float(np.std(team / m)),
        "T_std": float(np.std(frac)),
        "col_per_m_std": float(np.std(col / m)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model_dir", required=True, help="dir with actor.pt")
    p.add_argument("--test_num_agents", type=int, nargs="+", default=[3, 7, 10, 15])
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--obs_min", type=int, default=0)
    p.add_argument("--obs_max", type=int, default=10, help="inclusive")
    p.add_argument(
        "--fixed_obstacles",
        type=int,
        default=None,
        help="if set, use this many obstacles in every episode instead",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out_json", type=str, default=None)
    cli = p.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    torch.set_num_threads(1)

    summary, per_episode = [], {}
    for m in cli.test_num_agents:
        rng = np.random.default_rng(cli.seed + m)
        if cli.fixed_obstacles is not None:
            ks = np.full(cli.eval_episodes, cli.fixed_obstacles)
        else:
            ks = rng.integers(cli.obs_min, cli.obs_max + 1, size=cli.eval_episodes)
        eps = []
        for k in sorted(set(ks.tolist())):
            n_k = int(np.sum(ks == k))
            seed_base = cli.seed * 1000 + m * 100_000 + k * 1000
            eps += run_group(cli, m, int(k), n_k, seed_base)
        s = summarise(m, eps)
        summary.append(s)
        per_episode[m] = eps
        print(
            f"m={m:3d}  Reward/m={s['reward_per_m']:7.2f}  T={s['T']:.2f}  "
            f"(#col)/m={s['col_per_m']:.2f}  S%={s['S_pct']:.0f}",
            flush=True,
        )

    if cli.out_json:
        Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cli.out_json, "w") as f:
            json.dump(
                {"model_dir": cli.model_dir, "args": vars(cli), "summary": summary,
                 "episodes": per_episode},
                f,
                indent=1,
            )
        print(f"saved {cli.out_json}")


if __name__ == "__main__":
    main()
