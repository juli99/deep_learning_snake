"""
evaluate.py — measure a trained agent fairly: greedy play (no exploration),
many games, no drawing.

    python evaluate.py                                  # best model in runs/, 500 games
    python evaluate.py --model runs/main/best.pth --games 1000
    python evaluate.py --legacy model/model.pth         # the original 11-feature model

Prints mean / median / max score, and the causes of death
(wall, own body, or starvation = 100·length moves without food).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent
from snake_env import DX, DY, VecSnakeEnv


def run_games(policy, n_games, n_envs=100, seed=123, max_steps=5_000_000):
    env = VecSnakeEnv(n_envs=n_envs, seed=seed)
    scores, causes, steps = [], {"wall": 0, "body": 0, "starved": 0}, 0
    while len(scores) < n_games and steps < max_steps:
        obs = env.observe()
        actions = policy(obs)
        # cause of death, computed before the step (same rules as the env)
        d = (env.dir + np.where(actions == 1, 1, np.where(actions == 2, -1, 0))) % 4
        nx, ny = env.hx + DX[d], env.hy + DY[d]
        wall = (nx < 0) | (nx >= env.w) | (ny < 0) | (ny >= env.h)
        _, dones, final = env.step(actions)
        steps += n_envs
        for i in np.flatnonzero(dones):
            if len(scores) >= n_games:
                break
            scores.append(int(final[i]))
            causes["wall" if wall[i] else ("starved" if obs[i, 14] >= 1.0 - 1e-6 else "body")] += 1
    return np.array(scores), causes


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--legacy", default=None, help="evaluate an original model.pth (11 features, 1 hidden layer)")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.legacy:
        from model import Linear_QNet
        ck = torch.load(args.legacy, map_location="cpu", weights_only=False)
        net = Linear_QNet(11, 256, 3)
        net.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
        name = f"legacy {args.legacy}"
        # The 11 original features are the first 11 of ours (same order and meaning)
        policy = lambda o: net(torch.from_numpy(o[:, :11])).argmax(dim=1).numpy()  # noqa: E731
    else:
        from model_select import best_model
        path = Path(args.model) if args.model else best_model()
        net, _ = DQNAgent.load_policy(path)
        name = str(path)
        nf = net.net[0].in_features
        policy = lambda o: net(torch.from_numpy(o[:, :nf])).argmax(dim=1).numpy()  # noqa: E731

    with torch.no_grad():
        scores, causes = run_games(policy, args.games, seed=args.seed)
    print(f"{name}\n  games {len(scores)}   mean {scores.mean():.1f}   median {np.median(scores):.0f}   "
          f"max {scores.max()}   min {scores.min()}")
    print("  deaths: " + ", ".join(f"{k} {v} ({100 * v / len(scores):.0f}%)" for k, v in causes.items()))


if __name__ == "__main__":
    main()
