#!/usr/bin/env python3
"""Figure 2: per-episode SLA compliance reliability plot by arm.

Replaces the original grouped-bar mean+-std figure: mean+-std across n=3
seeds collapses a genuinely bimodal distribution (baseline: ~60% in 2
seeds, ~100% in the third) into a misleading "73%+-19%" summary, and hides
that every learned arm is at exactly 100.0% in EVERY one of its 15
episodes (3 seeds x 5 episodes), not just on average. This figure shows
the raw per-episode distribution directly.

Top panel: one dot per episode (mean SLA compliance across the 3 slices,
%), x-jittered deterministically (no RNG) within each arm's column, so a
stack of identical values (all 4 learned arms) renders as a tight flat
band while baseline's bimodal spread is visible as two clusters.

Bottom panel: two bars per arm -- fraction of episodes fully SLA-compliant
(compliance == 100.0% on all 3 slices) and worst single-episode compliance
-- pooled across all seeds/episodes (n=15/arm), not a mean-of-means.

Reads only from live-eval omega logs (episode-rollup rows, step==-1),
same data source as the original figure.

Usage:
    python3 experiments/plots/fig2_sla_compliance.py \
        --live-root experiments/results/live_campaign --seeds 950 951 952 \
        --out experiments/plots/out/fig2_sla_compliance
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, ARMS, SLICE_ORDER, arm_run_dir, read_omega_log  # noqa: E402

ARM_REWARD_MODE = {
    "baseline": "sla", "dqn_sla": "sla", "a2c_sla": "sla",
    "dqn_qoe": "qoe", "a2c_qoe": "qoe", "static_at_cap": "sla",
}

FULLY_COMPLIANT_THRESHOLD = 99.995  # % , tolerant of float roundoff at exactly 100.0


def per_episode_overall_compliance(omega_path: Path) -> list:
    """One value per episode-rollup row: mean SLA compliance (%) across the
    3 slices for that episode."""
    out = []
    for row in read_omega_log(omega_path):
        if row.step != -1:
            continue
        by_slice = row.evidence.get("episode_sla_compliance_by_slice")
        if not by_slice:
            continue
        vals = [by_slice[s] for s in SLICE_ORDER if s in by_slice]
        if vals:
            out.append(100.0 * float(np.mean(vals)))
    return out


def deterministic_jitter(n: int, width: float = 0.30) -> np.ndarray:
    """Symmetric, evenly-spaced offsets -- a fixed layout given n, not a
    random draw, so the figure is exactly reproducible from the data."""
    if n <= 1:
        return np.zeros(n)
    return (np.arange(n) - (n - 1) / 2) / (n - 1) * width


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live_campaign")
    ap.add_argument("--seeds", type=int, nargs="+", default=[950, 951, 952])
    ap.add_argument("--out", default="experiments/plots/out/fig2_sla_compliance")
    ap.add_argument("--arms", nargs="+", default=ARMS,
                     help="Subset of args.arms to plot (default: all 5). E.g. --arms baseline dqn_sla dqn_qoe.")
    args = ap.parse_args()

    arm_episode_vals: dict = {}
    n_seeds_used = {}

    for arm in args.arms:
        mode = ARM_REWARD_MODE[arm]
        pooled = []
        seeds_seen = 0
        for seed in args.seeds:
            omega_path = arm_run_dir(args.live_root, arm, mode, seed) / "omega_log.jsonl"
            if not omega_path.exists():
                continue
            vals = per_episode_overall_compliance(omega_path)
            if vals:
                seeds_seen += 1
            pooled.extend(vals)
        arm_episode_vals[arm] = pooled
        n_seeds_used[arm] = seeds_seen

    # Larger, double-column-width rendering for legibility -- scoped to this
    # figure only (rc_context, not a change to common.py's shared defaults,
    # which every other figure script still relies on unchanged).
    big_style = {
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.5,
    }
    with plt.rc_context(big_style):
        fig, (ax_top, ax_bot) = plt.subplots(
            1, 2, figsize=(7.16, 3.6), gridspec_kw={"width_ratios": [1.6, 1.0]}
        )

        x = np.arange(len(args.arms))
        for i, arm in enumerate(args.arms):
            vals = np.array(arm_episode_vals[arm])
            if vals.size == 0:
                continue
            jitter = deterministic_jitter(len(vals))
            style = ARM_STYLE[arm]
            ax_top.scatter(i + jitter, vals, color=style["color"], marker=style["marker"],
                            s=42, alpha=0.75, linewidths=0.5, edgecolors="white")

        ax_top.set_xticks(x)
        ax_top.set_xticklabels([ARM_STYLE[a]["label"] for a in args.arms], rotation=20, ha="right")
        ax_top.set_ylabel("Per-episode SLA compliance (%)")
        ax_top.set_ylim(-5, 105)
        n_eps_seen = {len(v) for v in arm_episode_vals.values() if v}
        n_eps_label = str(next(iter(n_eps_seen))) if len(n_eps_seen) == 1 else "varies"
        ax_top.set_title(f"(a) Per-episode compliance (n={n_eps_label}/arm)")

        frac_fully_compliant = []
        worst_episode = []
        for arm in args.arms:
            vals = np.array(arm_episode_vals[arm])
            if vals.size == 0:
                frac_fully_compliant.append(float("nan"))
                worst_episode.append(float("nan"))
                continue
            frac_fully_compliant.append(100.0 * float(np.mean(vals >= FULLY_COMPLIANT_THRESHOLD)))
            worst_episode.append(float(np.min(vals)))

        bar_width = 0.35
        ax_bot.bar(x - bar_width / 2, frac_fully_compliant, bar_width,
                   color="#2a78d6", label="Fully compliant (%)")
        ax_bot.bar(x + bar_width / 2, worst_episode, bar_width,
                   color="#e34948", label="Worst episode (%)")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([ARM_STYLE[a]["label"] for a in args.arms], rotation=20, ha="right")
        ax_bot.set_ylabel("%")
        ax_bot.set_ylim(0, 105)
        ax_bot.set_title("(b) Compliance summary")
        ax_bot.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

        fig.tight_layout()
        # tight_layout() only sizes the two axes -- it has no notion of the
        # legend sitting outside ax_bot via bbox_to_anchor, so without this
        # explicit reservation the legend renders past the figure's right
        # edge and bbox_inches="tight" (below) crops/rescales around it,
        # squeezing the two subplots together enough to overlap their titles.
        fig.subplots_adjust(right=0.80)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[fig2] wrote {out_path}.pdf / .png -- n_seeds per arm: {n_seeds_used}, "
          f"episodes fully compliant: {dict(zip(args.arms, frac_fully_compliant))}, "
          f"worst episode: {dict(zip(args.arms, worst_episode))}")


if __name__ == "__main__":
    main()
