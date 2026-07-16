#!/usr/bin/env python3
"""Figure 1: training convergence -- reward vs. episode, mean +/- seed band,
for the 4 learned arms (dqn_sla, a2c_sla, dqn_qoe, a2c_qoe), reading only
from the offline-training omega logs
(experiments/results/offline/seed<N>/<algo>/offline_closed_loop/rep_0/omega_log.jsonl).

Usage:
    python3 experiments/plots/fig1_training_convergence.py \
        --offline-root experiments/results/offline --seeds 256 257 258 \
        --out experiments/plots/out/fig1_training_convergence
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, read_omega_log  # noqa: E402

# arm -> (algorithm string used in the offline run, reward_mode)
LEARNED_ARMS = {
    "dqn_sla": ("dqn", "sla"),
    "a2c_sla": ("a2c", "sla"),
    "dqn_qoe": ("dqn", "qoe"),
    "a2c_qoe": ("a2c", "qoe"),
}


def load_episode_rewards(omega_path: Path) -> np.ndarray:
    """Returns an array of per-episode mean reward, indexed by episode-1,
    from the rollup rows (step == -1) mc_runner.run_single emits."""
    rewards = {}
    for row in read_omega_log(omega_path):
        if row.step == -1 and "episode_mean_reward" in row.evidence:
            rewards[row.episode] = row.evidence["episode_mean_reward"]
    if not rewards:
        return np.array([])
    n = max(rewards)
    arr = np.full(n, np.nan)
    for ep, r in rewards.items():
        arr[ep - 1] = r
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline-root", default="experiments/results/offline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--out", default="experiments/plots/out/fig1_training_convergence")
    ap.add_argument("--source", default="offline_closed_loop")
    args = ap.parse_args()

    # 2x2 (one subplot per arm), NOT overlaid: eq.2 (sla) and eq.9 (qoe)
    # reward magnitudes are on genuinely different scales (confirmed, not a
    # plotting bug -- qoe-mode rewards converge near -0.5, sla-mode near
    # -4.5/-4.7) so overlaying all 4 on one axis visually flattens the qoe
    # arms to an uninformative near-flat line. The handover's Phase 4 spec
    # explicitly allows "2x2 or overlaid" for exactly this reason.
    fig, axes = plt.subplots(2, 2, figsize=(2 * 3.5, 2 * 2.2), sharex=True)
    axes = axes.flatten()
    n_seeds_used = {}

    for ax, (arm, (algo, mode)) in zip(axes, LEARNED_ARMS.items()):
        style = ARM_STYLE[arm]
        per_seed_curves = []
        for seed in args.seeds:
            omega_path = (
                Path(args.offline_root) / mode / f"seed{seed}" / algo / args.source / "rep_0" / "omega_log.jsonl"
            )
            if not omega_path.exists():
                continue
            curve = load_episode_rewards(omega_path)
            if curve.size:
                per_seed_curves.append(curve)

        if not per_seed_curves:
            print(f"[fig1] WARNING: no data found for arm={arm}", file=sys.stderr)
            continue

        min_len = min(c.size for c in per_seed_curves)
        stacked = np.stack([c[:min_len] for c in per_seed_curves])
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        episodes = np.arange(1, min_len + 1)

        ax.plot(episodes, mean, color=style["color"], linestyle=style["linestyle"], linewidth=1.0)
        ax.fill_between(episodes, mean - std, mean + std, color=style["color"], alpha=0.25, linewidth=0)
        ax.set_title(style["label"], fontsize=8)
        n_seeds_used[arm] = len(per_seed_curves)

    for ax in axes[2:]:
        ax.set_xlabel("Training episode")
    for ax in axes[::2]:
        ax.set_ylabel("Mean reward/step")
    fig.suptitle("Training convergence (mean ± seed std)", fontsize=9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig1] wrote {out_path}.pdf / .png -- n_seeds per arm: {n_seeds_used}")


if __name__ == "__main__":
    main()
