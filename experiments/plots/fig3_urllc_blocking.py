#!/usr/bin/env python3
"""Figure 3: URLLC blocking per episode by arm -- box plots (distribution
across episode-rollups, pooled across seeds), reading only from live
evaluation omega logs.

Usage:
    python3 experiments/plots/fig3_urllc_blocking.py \
        --live-root experiments/results/live --seeds 256 257 258 \
        --out experiments/plots/out/fig3_urllc_blocking
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, ARMS, arm_run_dir, read_omega_log  # noqa: E402

ARM_REWARD_MODE = {
    "baseline": "sla", "dqn_sla": "sla", "a2c_sla": "sla",
    "dqn_qoe": "qoe", "a2c_qoe": "qoe",
}


def per_episode_urllc_blocks(omega_path: Path) -> list:
    vals = []
    for row in read_omega_log(omega_path):
        if row.step != -1:
            continue
        by_slice = row.evidence.get("episode_block_by_slice")
        if by_slice is not None:
            vals.append(by_slice.get("urllc", 0))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live")
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--out", default="experiments/plots/out/fig3_urllc_blocking")
    args = ap.parse_args()

    data = []
    labels = []
    colors = []
    n_episodes_used = {}

    for arm in ARMS:
        mode = ARM_REWARD_MODE[arm]
        pooled = []
        for seed in args.seeds:
            omega_path = arm_run_dir(args.live_root, arm, mode, seed) / "omega_log.jsonl"
            if omega_path.exists():
                pooled.extend(per_episode_urllc_blocks(omega_path))
        n_episodes_used[arm] = len(pooled)
        data.append(pooled if pooled else [0])
        labels.append(ARM_STYLE[arm]["label"])
        colors.append(ARM_STYLE[arm]["color"])

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                     medianprops={"color": "black", "linewidth": 1.0})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)

    ax.set_xticks(range(1, len(ARMS) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("URLLC blocks / episode")
    ax.set_title("URLLC blocking per episode by arm")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig3] wrote {out_path}.pdf / .png -- n_episodes pooled per arm: {n_episodes_used}")


if __name__ == "__main__":
    main()
