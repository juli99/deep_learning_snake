"""
dqn.py — Double DQN agent with a target network and a NumPy replay buffer.

Differences from the original model.py / agent.py and why:
  * Two hidden layers (256-256) instead of one.
  * Target network: the Bellman target r + γ·Q_target(s', a*) uses a slowly
    updated copy of the network, so the target does not move with every step.
  * Double DQN: a* = argmax Q_online(s', ·), evaluated with Q_target — reduces
    the over-estimation of Q values that plain DQN suffers from.
  * Fully vectorized loss (no Python loop over samples) and Huber loss.
  * Replay buffer stored in preallocated NumPy arrays instead of a deque of tuples.
  * Epsilon decays with environment steps, not with the number of games.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, n_in, n_out=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, n_features):
        self.cap = capacity
        self.s = np.zeros((capacity, n_features), dtype=np.float32)
        self.s2 = np.zeros((capacity, n_features), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.pos, self.size = 0, 0

    def add_batch(self, s, a, r, s2, d):
        n = len(a)
        idx = (self.pos + np.arange(n)) % self.cap
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx] = s, a, r, s2, d
        self.pos = (self.pos + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, batch)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]), torch.from_numpy(self.d[idx]))


class DQNAgent:
    def __init__(self, n_features, n_actions=3, hidden=256, lr=1e-3, gamma=0.95,
                 buffer_size=200_000, batch_size=512, tau=0.01, seed=0):
        self.n_actions, self.gamma, self.batch_size, self.tau = n_actions, gamma, batch_size, tau
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.q = QNet(n_features, n_actions, hidden)
        self.q_target = QNet(n_features, n_actions, hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, n_features)
        self.hparams = dict(n_features=n_features, n_actions=n_actions, hidden=hidden, lr=lr,
                            gamma=gamma, batch_size=batch_size, tau=tau)

    @torch.no_grad()
    def act(self, obs, epsilon):
        """ε-greedy for a whole batch of games at once."""
        greedy = self.q(torch.from_numpy(obs)).argmax(dim=1).numpy()
        explore = self.rng.random(len(obs)) < epsilon
        return np.where(explore, self.rng.integers(0, self.n_actions, len(obs)), greedy)

    def learn(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size, self.rng)
        q_sa = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            a_star = self.q(s2).argmax(dim=1, keepdim=True)            # online net chooses
            q_next = self.q_target(s2).gather(1, a_star).squeeze(1)    # target net evaluates
            target = r + self.gamma * (1.0 - d) * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        # Soft update: target ← τ·online + (1-τ)·target
        with torch.no_grad():
            for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_t.mul_(1 - self.tau).add_(p, alpha=self.tau)
        return loss.item()

    def save(self, path, **meta):
        torch.save({"model_state": self.q.state_dict(), "hparams": self.hparams, **meta}, path)

    @staticmethod
    def load_policy(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        hp = ck["hparams"]
        net = QNet(hp["n_features"], hp["n_actions"], hp["hidden"])
        net.load_state_dict(ck["model_state"])
        net.eval()
        return net, ck
