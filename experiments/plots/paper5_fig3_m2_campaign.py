#!/usr/bin/env python3
"""Figure 3: M2 30-seed campaign, per-seed evidence rather than aggregate
bars -- (a) a paired slope chart, single_agent_dqn -> gat_ctde
mean_reward_per_step, one line per seed, colored by win/tie/loss (the
actual unit the Wilcoxon signed-rank test in Section results-m2 is
computed over, so the plot shows the same thing the p-value summarizes,
not a lossy proxy of it); (b) a per-seed reward strip across all three
arms, with gat_ctde split into "differentiated" (any block) vs. "still
collapsed" (zero blocks) -- the 22/8 split from docs/PAPER5_M2_gat_ctde.md
section 12. Deliberately NOT claiming differentiated seeds score higher:
several still-collapsed seeds have the highest reward of the whole arm
(always-accepting never pays the reject cost), so the strip is left to
show that honestly rather than a caption asserting a trend the data does
not support.

Reuses m2_correctness_metrics.per_seed_metrics (imported, not
reimplemented).

Usage:
    python3 experiments/plots/paper5_fig3_m2_campaign.py \
        --results experiments/results/m2_campaign/campaign_results.json \
        --campaign-dir experiments/results/m2_campaign \
        --out paper5/figures/fig3_m2_campaign
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m2_correctness_metrics import per_seed_metrics  # noqa: E402
from paper5_common import M2_ARM_STYLE, STATUS_COLORS, bootstrap_ci, eval_omega_path, load_m2_campaign  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/m2_campaign/campaign_results.json")
    ap.add_argument("--campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--out", default="paper5/figures/fig3_m2_campaign")
    args = ap.parse_args()

    all_seeds, _results = load_m2_campaign(args.results)

    reward_by_arm = {}
    blocks_by_arm = {}
    for arm in ["gat_ctde", "independent_dqn", "single_agent_dqn"]:
        rewards, blocks = [], []
        for s in all_seeds:
            path = eval_omega_path(args.campaign_dir, arm, s)
            mrps, _mmtc_b, total_b = per_seed_metrics(str(path))
            rewards.append(mrps)
            blocks.append(total_b)
        reward_by_arm[arm] = np.array(rewards)
        blocks_by_arm[arm] = np.array(blocks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---- (a) paired slope: single_agent_dqn -> gat_ctde, per seed ----
    single = reward_by_arm["single_agent_dqn"]
    gat = reward_by_arm["gat_ctde"]
    diff = gat - single
    n_win = int((diff > 0).sum())
    n_tie = int((diff == 0).sum())
    n_lose = int((diff < 0).sum())

    for s0, s1 in zip(single, gat):
        d = s1 - s0
        if d > 0:
            color, ls = STATUS_COLORS["good"], "-"
        elif d < 0:
            color, ls = STATUS_COLORS["critical"], "--"
        else:
            color, ls = STATUS_COLORS["neutral"], ":"
        ax1.plot([0, 1], [s0, s1], color=color, linestyle=ls, linewidth=0.9, alpha=0.55, zorder=2)

    mean0, mean1 = single.mean(), gat.mean()
    ax1.plot([0, 1], [mean0, mean1], color="#0b0b0b", linewidth=2.2, marker="D",
              markersize=6, zorder=5, label="Mean")

    ax1.set_xlim(-0.25, 1.25)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Single-agent\nDQN", "GAT-CTDE\n(proposed)"])
    ax1.set_ylabel("Mean reward per step")
    ax1.set_title(f"(a) Paired per-seed comparison (n={len(single)})")

    lo, hi = bootstrap_ci(diff)
    ax1.text(0.5, 0.02,
              f"mean $\\Delta$=+{diff.mean():.3f} [{lo:.2f}, {hi:.2f}]\n"
              f"Wilcoxon $p$=0.0001",
              transform=ax1.transAxes, ha="center", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fcfcfb", edgecolor="#c3c2b7", linewidth=0.6))

    legend_handles = [
        Line2D([0], [0], color=STATUS_COLORS["good"], linestyle="-", linewidth=1.4,
               label=f"GAT-CTDE higher ({n_win})"),
        Line2D([0], [0], color=STATUS_COLORS["neutral"], linestyle=":", linewidth=1.4,
               label=f"Tie ({n_tie})"),
        Line2D([0], [0], color=STATUS_COLORS["critical"], linestyle="--", linewidth=1.4,
               label=f"GAT-CTDE lower ({n_lose})"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=False, handlelength=1.6)

    # ---- (b) per-seed reward strip, all 3 arms, gat_ctde split by collapse status ----
    rng = np.random.RandomState(0)
    x_pos = {"gat_ctde": 0, "independent_dqn": 1, "single_agent_dqn": 2}

    gat_diff_mask = blocks_by_arm["gat_ctde"] > 0
    n_diff = int(gat_diff_mask.sum())
    n_collapsed = int((~gat_diff_mask).sum())

    jitter = rng.uniform(-0.13, 0.13, size=len(all_seeds))
    ax2.scatter(x_pos["gat_ctde"] + jitter[gat_diff_mask], reward_by_arm["gat_ctde"][gat_diff_mask],
                s=16, color=M2_ARM_STYLE["gat_ctde"]["color"], marker="o", alpha=0.75,
                linewidth=0, zorder=3, label=f"GAT-CTDE, differentiated ({n_diff})")
    ax2.scatter(x_pos["gat_ctde"] + jitter[~gat_diff_mask], reward_by_arm["gat_ctde"][~gat_diff_mask],
                s=22, color="#898781", marker="x", alpha=0.9, linewidth=1.1,
                zorder=3, label=f"GAT-CTDE, still collapsed ({n_collapsed})")

    for arm in ["independent_dqn", "single_agent_dqn"]:
        style = M2_ARM_STYLE[arm]
        ax2.scatter(x_pos[arm] + jitter, reward_by_arm[arm], s=16, color=style["color"],
                    marker=style["marker"], alpha=0.65, linewidth=0, zorder=3, label=style["label"])

    for arm in ["gat_ctde", "independent_dqn", "single_agent_dqn"]:
        v = reward_by_arm[arm]
        lo, hi = bootstrap_ci(v)
        ax2.errorbar(x_pos[arm] + 0.32, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                     fmt="D", color="#0b0b0b", markersize=5, capsize=3, linewidth=1.0, zorder=5)

    ax2.set_xlim(-0.5, 2.7)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["GAT-CTDE", "Independent\nDQN", "Single-agent\nDQN"])
    ax2.set_ylabel("Mean reward per step")
    ax2.set_title(f"(b) Per-seed distribution (n={len(all_seeds)}/arm)")

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2.append(Line2D([0], [0], marker="D", color="#0b0b0b", linestyle="none",
                            markersize=5, label="Mean $\\pm$ 95% CI"))
    labels2.append("Mean $\\pm$ 95% CI")
    ax2.legend(handles2, labels2, loc="lower center", bbox_to_anchor=(0.5, -0.62),
               ncol=1, frameon=False, fontsize=6, handlelength=1.3)

    fig.suptitle("M2 campaign: per-seed reward evidence, 30 seeds/arm", y=1.02, fontsize=9)
    fig.subplots_adjust(bottom=0.34, wspace=0.32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig3] wrote {out_path}.pdf / .png")
    print(f"  paired single_agent_dqn->gat_ctde: mean diff={diff.mean():.3f}, "
          f"win={n_win} tie={n_tie} lose={n_lose}")
    print(f"  gat_ctde differentiated={n_diff}, still collapsed={n_collapsed}")
    for arm in ["gat_ctde", "independent_dqn", "single_agent_dqn"]:
        v = reward_by_arm[arm]
        print(f"  {arm}: reward mean={v.mean():.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
