#!/usr/bin/env python3
"""Learning-curve figure for the diurnal (peak/off-peak week, condensed to
N episodes) training experiment: does the policy improve at handling
RECURRING peak-hour contention over the course of training, not just
show noisy per-episode compliance driven by that episode's demand level?

Reads experiments/results/offline_diurnal/<reward_mode>/seed<seed>/<algo>/
episode_metrics.jsonl (one row/episode: episode, mean_reward, compliance
per slice, day_of_week, is_weekend, shape -- see train_offline_diurnal.py).

Top panel: the diurnal "shape" (0=off-peak, 1=peak) schedule itself, for
context -- what demand level each episode actually faced.
Middle panel: rolling-window (default 40 ep) mean reward, ALL episodes.
Bottom panel: rolling-window mean SLA compliance, computed SEPARATELY for
peak-shape (>0.5) and off-peak-shape (<=0.5) episodes -- this is the
actual "learning over time" signal: whether compliance during peak
periods specifically improves as MORE peak episodes are seen in
training, decoupled from the raw noise of alternating demand levels.

Usage:
    python3 experiments/plots/fig_diurnal_learning_curve.py \
        --metrics experiments/results/offline_diurnal/qoe/seed256/dqn/episode_metrics.jsonl \
        --title "DQN, QoE reward" \
        --out experiments/plots/out/fig_diurnal_learning_curve_qoe
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SLICE_ORDER, SLICE_STYLE  # noqa: E402

PEAK_THRESHOLD = 0.5


def rolling_mean(values: list, window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.full(arr.shape, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = np.nanmean(arr[lo:i + 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.metrics)]
    episodes = [r["episode"] for r in rows]
    shapes = [r["shape"] for r in rows]
    rewards = [r["mean_reward"] for r in rows]
    compliance = {s: [r["compliance"].get(s, 0.0) * 100 for r in rows] for s in SLICE_ORDER}

    is_peak = [s > PEAK_THRESHOLD for s in shapes]
    peak_idx = [i for i, p in enumerate(is_peak) if p]
    offpeak_idx = [i for i, p in enumerate(is_peak) if not p]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(episodes, shapes, color="#555555", linewidth=0.8)
    ax.axhline(PEAK_THRESHOLD, color="#e34948", linestyle="--", linewidth=0.6, label="peak/off-peak threshold")
    ax.set_ylabel("Diurnal demand\nshape [0,1]")
    ax.set_title(f"Condensed peak/off-peak week schedule ({args.title})" if args.title else "Condensed peak/off-peak week schedule")
    ax.legend(loc="upper right", fontsize=6.5, frameon=False)

    ax = axes[1]
    ax.plot(episodes, rolling_mean(rewards, args.window), color="#2a78d6", linewidth=1.2)
    ax.set_ylabel(f"Reward\n(rolling {args.window}-ep mean)")

    ax = axes[2]
    for s in SLICE_ORDER:
        style = SLICE_STYLE[s]
        peak_roll = rolling_mean([compliance[s][i] for i in peak_idx], args.window)
        offpeak_roll = rolling_mean([compliance[s][i] for i in offpeak_idx], args.window)
        ax.plot(range(len(peak_roll)), peak_roll, color=style["color"], linestyle="-",
                linewidth=1.3, label=f"{style['label']} (peak eps)")
        ax.plot(range(len(offpeak_roll)), offpeak_roll, color=style["color"], linestyle=":",
                linewidth=1.0, alpha=0.7, label=f"{style['label']} (off-peak eps)")
    ax.set_ylabel(f"SLA compliance (%)\n(rolling {args.window}-ep mean,\nwithin peak/off-peak)")
    ax.set_xlabel("n-th episode of that type encountered during training")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=5.5, frameon=False, ncol=2)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))

    print(f"[fig_diurnal] wrote {out_path}.pdf / .png")
    print(f"[fig_diurnal] n_episodes={len(rows)}, n_peak={len(peak_idx)}, n_offpeak={len(offpeak_idx)}")
    for s in SLICE_ORDER:
        first10 = np.mean([compliance[s][i] for i in peak_idx[:10]]) if len(peak_idx) >= 10 else float("nan")
        last10 = np.mean([compliance[s][i] for i in peak_idx[-10:]]) if len(peak_idx) >= 10 else float("nan")
        print(f"  {s}: peak-episode compliance first10={first10:.1f}% -> last10={last10:.1f}%")


if __name__ == "__main__":
    main()
