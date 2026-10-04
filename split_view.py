"""
split_view.py — "split screen": draws several of the parallel games in one pygame window.

Drawing is the slow part of the original project, so training never waits for
the screen: the grid is redrawn only every few steps and only for a few games.
"""

import os

import pygame

BG, GRID_LINE = (12, 16, 28), (40, 48, 70)
SNAKE_HEAD, SNAKE_BODY, FOOD, TEXT = (0, 160, 255), (0, 90, 220), (220, 40, 40), (235, 235, 235)


class SplitView:
    def __init__(self, env, n_show=9, cell=10, title="Snake — split screen"):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        pygame.init()
        self.env = env
        self.n_show = min(n_show, env.n)
        self.cols = int(round(self.n_show ** 0.5 + 0.49))
        self.rows = -(-self.n_show // self.cols)
        self.cell = cell
        self.pw, self.ph = env.w * cell, env.h * cell      # one panel
        self.gap, self.top = 6, 34
        W = self.cols * self.pw + (self.cols + 1) * self.gap
        H = self.top + self.rows * self.ph + (self.rows + 1) * self.gap
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("arial", 16)
        self.small = pygame.font.SysFont("arial", 13)

    def handle_events(self):
        """Returns False when the window was closed."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
        return True

    def draw(self, header=""):
        s, c = self.screen, self.cell
        s.fill((0, 0, 0))
        s.blit(self.font.render(header, True, TEXT), (self.gap, 8))
        for k in range(self.n_show):
            r, q = divmod(k, self.cols)
            ox = self.gap + q * (self.pw + self.gap)
            oy = self.top + self.gap + r * (self.ph + self.gap)
            pygame.draw.rect(s, BG, (ox, oy, self.pw, self.ph))
            pygame.draw.rect(s, GRID_LINE, (ox - 1, oy - 1, self.pw + 2, self.ph + 2), 1)
            cells, (fx, fy), score = self.env.snapshot(k)
            pygame.draw.rect(s, FOOD, (ox + fx * c, oy + fy * c, c, c))
            for j, (x, y) in enumerate(cells):
                pygame.draw.rect(s, SNAKE_HEAD if j == 0 else SNAKE_BODY, (ox + x * c + 1, oy + y * c + 1, c - 2, c - 2))
            s.blit(self.small.render(str(score), True, TEXT), (ox + 4, oy + 2))
        pygame.display.flip()

    def close(self):
        pygame.quit()
