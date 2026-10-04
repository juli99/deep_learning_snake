"""Pick the trained model with the highest greedy-evaluation score across runs/."""

import sys
from pathlib import Path

import torch


def best_model():
    cands = []
    for p in Path("runs").glob("*/best.pth"):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cands.append((ck.get("eval_mean", -1), p))
    if not cands:
        sys.exit("No trained model found — run train.py first.")
    return max(cands)[1]
