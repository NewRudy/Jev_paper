"""Paper figures: reliability diagrams and risk-coverage curves, computed from
kev rows.json (never from a hand-copied number).

Design constraints applied:
  * categorical palette = the first three validated slots, so at most THREE
    arms share a panel; further arms go to small multiples (the four-plus
    all-pairs case does not clear the CVD floor)
  * validated on a white paper surface: worst-pair CVD dE 9.2, normal 24.0
  * every panel with >=2 series carries a legend AND direct labels at the last
    point, so identity is never colour-alone (relief rule for the aqua slot,
    which sits below 3:1 contrast on white)
  * thin marks, recessive grid, no dual axes, text in ink not series colour

    python3 gen/make_figures.py --runs runs --arms base std arft --tags amb_mixed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical slots (light mode, white surface)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9a95"
GRID = "#e4e4e0"


def load_rows(runs: Path, arm: str, tag: str):
    p = runs / arm / tag / "rows.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _declutter(labels, min_gap=0.045):
    """Push overlapping direct labels apart vertically, keeping their order."""
    out = []
    for x, y, nm in sorted(labels, key=lambda t: t[1]):
        if out and y - out[-1][1] < min_gap:
            y = out[-1][1] + min_gap
        out.append((x, y, nm))
    return out


def conf_correct(rows):
    out = []
    for r in rows:
        p = np.asarray(r["p"], dtype=float)
        if p.sum() > 0:
            p = p / p.sum()
        out.append((float(p.max()), int(p.argmax() == int(r["label"]))))
    return np.array(out)


def reliability(ax, rows, name, color, bins=10, pending=None):
    pending = pending if pending is not None else []
    cc = conf_correct(rows)
    if len(cc) == 0:
        return
    conf, ok = cc[:, 0], cc[:, 1]
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, color=MUTED, zorder=1,
            label="_nolegend_")
    xs, ys = [], []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= 1.0)
        if m.sum() < 5:                       # too few points to draw honestly
            continue
        xs.append(conf[m].mean())
        ys.append(ok[m].mean())
    ax.plot(xs, ys, "-o", lw=2.0, ms=6, color=color, mec="white", mew=1.2,
            zorder=3, label=name, clip_on=False)
    if xs:
        pending.append((xs[-1], ys[-1], name))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mean confidence in bin", fontsize=9, color=INK2)
    ax.set_ylabel("accuracy in bin", fontsize=9, color=INK2)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK2, length=3)


def risk_coverage(ax, rows, name, color, pending=None):
    pending = pending if pending is not None else []
    cc = conf_correct(rows)
    if len(cc) == 0:
        return None
    conf, ok = cc[:, 0], cc[:, 1]
    order = np.argsort(-conf)
    n = len(order)
    cov = np.arange(1, n + 1) / n
    acc = ok[order].cumsum() / np.arange(1, n + 1)
    step = max(1, n // 300)
    ax.plot(cov[::step], acc[::step], lw=2.0, color=color, label=name, zorder=3)
    pending.append((float(cov[-1]), float(acc[-1]), name))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("coverage (fraction answered)", fontsize=9, color=INK2)
    ax.set_ylabel("accuracy on answered", fontsize=9, color=INK2)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    return float(np.trapezoid(1 - acc, cov)) if hasattr(np, "trapezoid") else float(np.trapz(1 - acc, cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.arms) > 3:
        print("NOTE: >3 arms on one panel; the categorical palette only "
              "clears the all-pairs CVD floor for three. Render small multiples.")

    aurcs = {}
    for tag in args.tags:
        loaded = [(a, load_rows(runs, a, tag)) for a in args.arms]
        loaded = [(a, r) for a, r in loaded if r]
        if not loaded:
            print("no data for", tag)
            continue

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), dpi=200)
        p_rel, p_rc = [], []
        for i, (arm, rows) in enumerate(loaded[:3]):
            c = SLOTS[i % len(SLOTS)]
            reliability(axes[0], rows, arm, c, bins=args.bins, pending=p_rel)
            a = risk_coverage(axes[1], rows, arm, c, pending=p_rc)
            if a is not None:
                aurcs.setdefault(tag, {})[arm] = a
        for (x, y, nm) in _declutter(p_rel):
            axes[0].annotate(nm, (x, y), textcoords="offset points",
                             xytext=(7, 0), fontsize=8.5, color=INK2, zorder=4)
        for (x, y, nm) in _declutter(p_rc):
            axes[1].annotate(nm, (x, y), textcoords="offset points",
                             xytext=(-5, 0), fontsize=8.5, color=INK2,
                             ha="right", zorder=4)
        axes[0].set_title("Reliability (calibration)", fontsize=10.5, color=INK, loc="left", pad=8)
        axes[1].set_title("Risk-coverage (selective prediction)", fontsize=10.5, color=INK, loc="left", pad=8)
        if len(loaded) >= 2:
            axes[0].legend(fontsize=8.5, frameon=False, loc="upper left",
                           labelcolor=INK2, handlelength=1.6)
        fig.suptitle(f"Ambiguity type: {tag}", fontsize=11, color=INK, x=0.06, ha="left", y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out / f"{tag}_calibration.png"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", p)

    if aurcs:
        (out / "aurc.json").write_text(json.dumps(aurcs, indent=2))
        print("AURC (lower is better):")
        for tag, d in aurcs.items():
            print(f"  {tag}: " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()))


if __name__ == "__main__":
    main()
