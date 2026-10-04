"""
train.py — fast training: many Snake games in parallel, headless, Double DQN.

    python train.py                          # 64 games in parallel, 1.5M moves
    python train.py --envs 128 --steps 3e6   # more games / longer
    python train.py --view                   # also watch 9 of the games (split screen)
    python train.py --view --show 16         # watch 16 games

Every step, all games move at once: the network picks actions for the whole
batch, every game's move goes into the replay buffer, and the network does one
gradient step. Output (in runs/<name>/):
    best.pth      network with the best average score over the last 200 games
    last.pth      network at the end
    log.csv       one line per finished game
    progress.png  learning curve (average score vs. games and vs. time)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent
from snake_env import N_FEATURES, VecSnakeEnv


def parse():
    ap = argparse.ArgumentParser(description="Train a Snake agent with parallel games")
    ap.add_argument("--envs", type=int, default=64, help="games played in parallel")
    ap.add_argument("--steps", type=float, default=1e6, help="total moves (all games together)")
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.001)
    ap.add_argument("--eps-frac", type=float, default=0.25, help="share of training over which epsilon decays")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5_000, help="moves collected before learning starts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run folder name (default: time stamp)")
    ap.add_argument("--view", action="store_true", help="show some games in a split-screen window")
    ap.add_argument("--show", type=int, default=9, help="games to show with --view")
    ap.add_argument("--view-every", type=int, default=4, help="redraw every N steps with --view")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=float, default=1e5, help="greedy evaluation every N moves (picks best.pth)")
    ap.add_argument("--eval-games", type=int, default=50)
    ap.add_argument("--features", type=int, default=N_FEATURES, choices=[11, N_FEATURES],
                    help="11 = only the original sensors (for comparison), 16 = with flood-fill sensors")
    return ap.parse_args()


def save_plot(games_log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.array([g[2] for g in games_log], dtype=float)
    times = np.array([g[4] for g in games_log], dtype=float) / 60
    k = 200
    avg = np.convolve(scores, np.ones(k) / k, mode="valid") if len(scores) >= k else np.array([])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(scores, color="#B8C7DC", lw=0.5, label="score per game")
    if avg.size:
        a1.plot(np.arange(k - 1, len(scores)), avg, color="#0B3C74", lw=2, label=f"average of last {k}")
        a2.plot(times[k - 1:], avg, color="#2E9E6B", lw=2)
    a1.set_xlabel("games"), a1.set_ylabel("score"), a1.legend(frameon=False)
    a2.set_xlabel("training time (minutes)"), a2.set_ylabel(f"average score (last {k})")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def greedy_eval(agent, nf, n_games, seed=999):
    """Mean score of n_games played with no exploration (the real skill of the network)."""
    env = VecSnakeEnv(n_envs=n_games, seed=seed)
    done_once = np.zeros(n_games, dtype=bool)
    scores = np.zeros(n_games)
    for _ in range(200_000):
        with torch.no_grad():
            a = agent.q(torch.from_numpy(env.observe()[:, :nf])).argmax(dim=1).numpy()
        _, dones, final = env.step(a)
        new = dones & ~done_once
        scores[new] = final[new]
        done_once |= dones
        if done_once.all():
            break
    return float(scores.mean())


def main():
    # Windows consoles default to a legacy code page; make printing safe
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse()
    torch.set_num_threads(args.threads)
    run = Path("runs") / (args.name or time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)

    env = VecSnakeEnv(n_envs=args.envs, seed=args.seed)
    nf = args.features
    agent = DQNAgent(nf, lr=args.lr, gamma=args.gamma, batch_size=args.batch, seed=args.seed)
    view = None
    if args.view:
        from split_view import SplitView
        view = SplitView(env, n_show=args.show, title="Snake — training (split screen)")

    total_steps = int(args.steps)
    iters = total_steps // args.envs
    eps_iters = int(args.eps_frac * iters)
    games_log, best_avg, record = [], -1.0, 0
    obs = env.observe()[:, :nf]
    t0 = t_print = time.time()
    log_f = open(run / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["game", "move", "score", "epsilon", "seconds"])
    eval_every = int(args.eval_every)
    eval_f = open(run / "eval.csv", "w", newline="")
    evals = csv.writer(eval_f)
    evals.writerow(["moves", "seconds", "greedy_mean"])

    print(f"Training {args.envs} games in parallel for {total_steps:,} moves → {run}")
    for it in range(iters):
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * it / max(1, eps_iters))
        actions = agent.act(obs, eps)
        rewards, dones, final = env.step(actions)
        next_obs = env.observe()[:, :nf]
        agent.buffer.add_batch(obs, actions, rewards, next_obs, dones.astype(np.float32))
        obs = next_obs

        if agent.buffer.size >= args.warmup:
            agent.learn()

        now = time.time()
        for sc in final[dones]:
            record = max(record, int(sc))
            games_log.append((len(games_log) + 1, (it + 1) * args.envs, int(sc), round(eps, 4), round(now - t0, 1)))
            log.writerow(games_log[-1])

        moves = (it + 1) * args.envs
        if moves // eval_every != (moves - args.envs) // eval_every and agent.buffer.size >= args.warmup:
            score = greedy_eval(agent, nf, args.eval_games)
            evals.writerow([moves, round(now - t0, 1), round(score, 2)])
            eval_f.flush()
            print(f"  greedy eval at {moves:,} moves: mean {score:.1f} over {args.eval_games} games", flush=True)
            if score > best_avg:
                best_avg = score
                agent.save(run / "best.pth", games=len(games_log), moves=moves, eval_mean=score,
                           record=record, seconds=now - t0)

        if view is not None and it % args.view_every == 0:
            if not view.handle_events():
                view.close()
                view = None
            else:
                avg_now = np.mean([g[2] for g in games_log[-200:]]) if games_log else 0
                view.draw(f"games {len(games_log):,}   avg(200) {avg_now:.1f}   record {record}   ε {eps:.2f}")

        if now - t_print > 10:
            t_print = now
            moves = (it + 1) * args.envs
            avg_now = np.mean([g[2] for g in games_log[-200:]]) if games_log else 0
            print(f"{now - t0:6.0f}s  moves {moves:>10,}  ({moves / (now - t0):7,.0f}/s)  games {len(games_log):>7,}  "
                  f"avg(200) {avg_now:5.1f}  best eval {best_avg:5.1f}  record {record:3d}  ε {eps:.3f}", flush=True)
            log_f.flush()

    log_f.close()
    final = greedy_eval(agent, nf, args.eval_games)
    evals.writerow([total_steps, round(time.time() - t0, 1), round(final, 2)])
    eval_f.close()
    if final > best_avg:
        best_avg = final
        agent.save(run / "best.pth", games=len(games_log), moves=total_steps, eval_mean=final, record=record,
                   seconds=time.time() - t0)
    agent.save(run / "last.pth", games=len(games_log), moves=total_steps, record=record, seconds=time.time() - t0)
    save_plot(games_log, run / "progress.png")
    print(f"Done in {(time.time() - t0) / 60:.1f} min · games {len(games_log):,} · "
          f"best greedy eval {best_avg:.1f} · record {record}\nSaved to {run.resolve()}")
    if view is not None:
        view.close()


if __name__ == "__main__":
    main()
