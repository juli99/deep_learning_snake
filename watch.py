"""
watch.py — watch a trained agent play several games side by side (split screen).

    python watch.py                              # best model found in runs/, 3x3 games
    python watch.py --model runs/main/best.pth --show 16 --fps 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent
from snake_env import VecSnakeEnv
from split_view import SplitView
from model_select import best_model


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--show", type=int, default=9, help="games side by side")
    ap.add_argument("--fps", type=float, default=20, help="moves per second (0 = as fast as possible)")
    ap.add_argument("--cell", type=int, default=12, help="pixels per cell")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    path = Path(args.model) if args.model else best_model()
    net, ck = DQNAgent.load_policy(path)
    nf = net.net[0].in_features  # 11 or 16 sensors
    print(f"Model: {path}  (greedy eval mean: {ck.get('eval_mean', ck.get('avg200', '?'))}, training record {ck.get('record', '?')})")

    env = VecSnakeEnv(n_envs=args.show, seed=args.seed)
    view = SplitView(env, n_show=args.show, cell=args.cell, title=f"Snake — {path.parent.name}")
    finished, best = [], 0
    while view.handle_events():
        with torch.no_grad():
            actions = net(torch.from_numpy(env.observe()[:, :nf])).argmax(dim=1).numpy()
        _, dones, final = env.step(actions)
        for sc in final[dones]:
            finished.append(int(sc))
            best = max(best, int(sc))
        mean = f"{np.mean(finished):.1f}" if finished else "-"
        view.draw(f"{path.parent.name}   finished games {len(finished)}   mean {mean}   best {best}")
        if args.fps > 0:
            time.sleep(1 / args.fps)
    view.close()


if __name__ == "__main__":
    main()
