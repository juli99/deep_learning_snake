"""
snake_env.py — N Snake games running in parallel with NumPy (no graphics).

Why: the original game.py draws every frame and waits for clock.tick(40), so
training is capped at ~40 moves per second on one game. Here all games step
together as array operations, which gives thousands of moves per second.

Board representation
--------------------
body[i, y, x] = how many more steps cell (x, y) stays occupied in game i.
  - the head cell gets value = snake length
  - every normal step all values drop by 1, so the tail cell frees itself
  - when the snake eats, nothing is decremented, so the snake grows by one
This makes moving, growing and collision checks pure array operations.

Rules match game.py: 32x24 cells (640x480 / 20), start length 3 heading right,
+10 for food, -10 for dying, and a game ends if no food is eaten for
100 * length steps (the original "stuck" rule).
"""

import numpy as np
from scipy import ndimage

# Directions in clockwise order: RIGHT, DOWN, LEFT, UP  (same as game.py)
DX = np.array([1, 0, -1, 0])
DY = np.array([0, 1, 0, -1])

# 4-connectivity inside one board, no connection between different games
_STRUCT = np.zeros((3, 3, 3), dtype=bool)
_STRUCT[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

N_FEATURES = 16


class VecSnakeEnv:
    def __init__(self, n_envs=64, width=32, height=24, seed=None,
                 reward_food=10.0, reward_death=-10.0):
        self.n, self.w, self.h = n_envs, width, height
        self.rng = np.random.default_rng(seed)
        self.reward_food, self.reward_death = reward_food, reward_death
        self.body = np.zeros((n_envs, height, width), dtype=np.int32)
        self.hx = np.zeros(n_envs, dtype=np.int64)
        self.hy = np.zeros(n_envs, dtype=np.int64)
        self.dir = np.zeros(n_envs, dtype=np.int64)
        self.length = np.zeros(n_envs, dtype=np.int64)
        self.fx = np.zeros(n_envs, dtype=np.int64)
        self.fy = np.zeros(n_envs, dtype=np.int64)
        self.score = np.zeros(n_envs, dtype=np.int64)
        self.steps_since_food = np.zeros(n_envs, dtype=np.int64)
        self.idx = np.arange(n_envs)
        self.reset(np.ones(n_envs, dtype=bool))

    # ── Game logic ──────────────────────────────────────────────────
    def reset(self, mask):
        ids = np.flatnonzero(mask)
        if ids.size == 0:
            return
        self.body[ids] = 0
        self.hx[ids], self.hy[ids] = self.w // 2, self.h // 2
        self.dir[ids] = 0
        self.length[ids] = 3
        for k in range(3):  # head, then two cells to the left
            self.body[ids, self.h // 2, self.w // 2 - k] = 3 - k
        self.score[ids] = 0
        self.steps_since_food[ids] = 0
        self._place_food(ids)

    def _place_food(self, ids):
        """Random free cell for each game in ids (argmax of random noise over free cells)."""
        noise = self.rng.random((ids.size, self.h, self.w))
        noise[self.body[ids] > 0] = -1.0
        flat = noise.reshape(ids.size, -1).argmax(axis=1)
        self.fy[ids], self.fx[ids] = np.divmod(flat, self.w)

    def step(self, actions):
        """
        actions: int array (n,) — 0 straight, 1 turn right, 2 turn left (as in game.py)
        returns rewards (n,), dones (n,), scores of games that just ended (n, -1 if not ended)
        Finished games are reset automatically.
        """
        self.dir = (self.dir + np.where(actions == 1, 1, np.where(actions == 2, -1, 0))) % 4
        nx, ny = self.hx + DX[self.dir], self.hy + DY[self.dir]

        ate = (nx == self.fx) & (ny == self.fy)
        # Tail moves forward unless the snake eats
        moving = ~ate
        self.body[moving] = np.maximum(self.body[moving] - 1, 0)

        wall = (nx < 0) | (nx >= self.w) | (ny < 0) | (ny >= self.h)
        cx, cy = np.clip(nx, 0, self.w - 1), np.clip(ny, 0, self.h - 1)
        hit_self = self.body[self.idx, cy, cx] > 0
        self.steps_since_food = np.where(ate, 0, self.steps_since_food + 1)
        starved = self.steps_since_food > 100 * self.length
        dead = wall | hit_self | starved

        alive = ~dead
        self.length[ate & alive] += 1
        self.score[ate & alive] += 1
        self.hx[alive], self.hy[alive] = nx[alive], ny[alive]
        self.body[self.idx[alive], ny[alive], nx[alive]] = self.length[alive]

        rewards = np.where(dead, self.reward_death, np.where(ate, self.reward_food, 0.0)).astype(np.float32)
        final_scores = np.where(dead, self.score, -1)

        eaten = np.flatnonzero(ate & alive)
        if eaten.size:
            self._place_food(eaten)
        self.reset(dead)
        return rewards, dead, final_scores

    # ── Observation ─────────────────────────────────────────────────
    def observe(self):
        """
        16 features per game:
          0-2   danger straight / right / left (wall or body next step)   [original 11]
          3-6   moving left / right / up / down                          [original 11]
          7-10  food left / right / up / down                            [original 11]
          11-13 free area reachable after going straight / right / left,
                as a fraction of all free cells (flood fill)             [new]
          14    hunger: steps since last food / starvation limit         [new]
          15    snake length / board size                               [new]
        The flood-fill features let the snake see that a move leads into a
        dead end (a pocket closed by its own body) before it enters it.
        """
        n = self.n
        dirs = np.stack([self.dir, (self.dir + 1) % 4, (self.dir - 1) % 4], axis=1)  # straight, right, left
        px = self.hx[:, None] + DX[dirs]
        py = self.hy[:, None] + DY[dirs]
        inside = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        cpx, cpy = np.clip(px, 0, self.w - 1), np.clip(py, 0, self.h - 1)
        # a body cell with value 1 is the tail, which frees itself on the next move
        occupied = self.body[self.idx[:, None], cpy, cpx] > 1
        danger = ~inside | occupied

        # Flood fill of free cells (after the tail moves) for all games at once
        free = self.body <= 1
        free[self.idx, self.hy, self.hx] = False
        labels, _ = ndimage.label(free, structure=_STRUCT)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        lab = labels[self.idx[:, None], cpy, cpx]
        area = np.where(danger, 0, sizes[lab]).astype(np.float32)
        total_free = free.reshape(n, -1).sum(axis=1).astype(np.float32)
        area /= np.maximum(total_free, 1)[:, None]

        obs = np.zeros((n, N_FEATURES), dtype=np.float32)
        obs[:, 0:3] = danger
        obs[:, 3] = self.dir == 2
        obs[:, 4] = self.dir == 0
        obs[:, 5] = self.dir == 3
        obs[:, 6] = self.dir == 1
        obs[:, 7] = self.fx < self.hx
        obs[:, 8] = self.fx > self.hx
        obs[:, 9] = self.fy < self.hy
        obs[:, 10] = self.fy > self.hy
        obs[:, 11:14] = area
        obs[:, 14] = self.steps_since_food / (100 * self.length)
        obs[:, 15] = self.length / (self.w * self.h)
        return obs

    # ── For drawing ─────────────────────────────────────────────────
    def snapshot(self, i):
        """(body cells ordered head->tail, food cell, score) of game i."""
        ys, xs = np.nonzero(self.body[i])
        order = np.argsort(-self.body[i, ys, xs])
        return list(zip(xs[order], ys[order])), (int(self.fx[i]), int(self.fy[i])), int(self.score[i])
