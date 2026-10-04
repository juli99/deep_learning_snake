"""
compare_runs.py — plot greedy-evaluation curves of several runs on one chart.

    python compare_runs.py                       # every run that has eval.csv
    python compare_runs.py feat16 feat11         # chosen runs
Saves runs/comparison.png
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"feat16": "16 sensors (with flood fill)", "feat11": "11 original sensors"}
COLOURS = ["#0B3C74", "#2E9E6B", "#C8553D", "#64A8E8"]


def main():
    names = sys.argv[1:] or sorted(p.parent.name for p in Path("runs").glob("*/eval.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, colour in zip(names, COLOURS):
        rows = list(csv.DictReader(open(Path("runs") / name / "eval.csv")))[:-1]  # last row repeats the final eval
        moves = [int(r["moves"]) / 1e3 for r in rows]
        secs = [float(r["seconds"]) / 60 for r in rows]
        score = [float(r["greedy_mean"]) for r in rows]
        label = LABELS.get(name, name)
        a1.plot(moves, score, "-o", color=colour, lw=2, label=label)
        a2.plot(secs, score, "-o", color=colour, lw=2, label=label)
    for a in (a1, a2):
        a.axhline(33.5, color="grey", ls="--", lw=1)
        a.set_ylabel("mean score, greedy play (50 games)")
        a.spines[["top", "right"]].set_visible(False)
        a.set_ylim(bottom=0)
    a1.text(0.99, 42, "original model (1.5 h): 33.5", ha="right", color="grey", fontsize=9,
            transform=a1.get_yaxis_transform())
    a1.set_xlabel("training moves (thousands)")
    a2.set_xlabel("training time (minutes)")
    a1.legend(frameon=False)
    fig.tight_layout()
    out = Path("runs") / "comparison.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
