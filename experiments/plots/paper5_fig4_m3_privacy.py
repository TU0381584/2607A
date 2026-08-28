#!/usr/bin/env python3
"""Figure 4: M3 federated + DP results, per-seed evidence --
(a) a paired slope chart, centralized GAT-CTDE -> federated/no-DP
mean_reward_per_step, one line per seed (same 10 seeds), colored by
whether federation cost or helped reward that seed -- the actual paired
unit the Wilcoxon result (computed fresh below, not hardcoded -- an
earlier version of this figure had the p-value baked in as a literal
string, which silently went stale after the M2 gat_ctde eval-log
correction; see docs/PAPER5_M2_gat_ctde.md's correction section) is
computed over; (b) the block_precision vs. DP noise multiplier sigma
privacy-utility curve,
with individual per-seed points (jittered) shown behind the mean, and
the number of seeds that ever blocked anything (precision is undefined
for zero-block seeds, so this count shrinks and matters) printed at each
sigma level instead of left implicit in a smooth line.

Reuses m2_correctness_metrics.per_seed_metrics (imported, not
reimplemented) for both metrics.

Usage:
    python3 experiments/plots/paper5_fig4_m3_privacy.py \
        --m3-results experiments/results/m3_campaign/campaign_results.json \
        --m3-campaign-dir experiments/results/m3_campaign \
        --m2-results experiments/results/m2_campaign/campaign_results.json \
        --m2-campaign-dir experiments/results/m2_campaign \
        --out Papers_4-5/Paper_5/IEEE_Access/figures/fig4_m3_privacy
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m2_correctness_metrics import per_seed_metrics  # noqa: E402
from paper5_common import M3_STYLE, STATUS_COLORS, bootstrap_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m3-results", default="experiments/results/m3_campaign/campaign_results.json")
    ap.add_argument("--m3-campaign-dir", default="experiments/results/m3_campaign")
    ap.add_argument("--m2-results", default="experiments/results/m2_campaign/campaign_results.json")
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--out", default="Papers_4-5/Paper_5/IEEE_Access/figures/fig4_m3_privacy")
    args = ap.parse_args()

    with open(args.m3_results) as fh:
        m3 = json.load(fh)
    seeds = m3["seeds"]

    centralized_rewards = []
    for seed in seeds:
        path = Path(args.m2_campaign_dir) / "gat_ctde" / f"seed{seed}" / "eval" / "omega_log.jsonl"
        mrps, _, _ = per_seed_metrics(str(path))
        centralized_rewards.append(mrps)
    centralized_rewards = np.array(centralized_rewards)

    fl_no_dp_rewards = []
    for seed in seeds:
        path = Path(args.m3_campaign_dir) / "fl_gat_ctde_sigma0.0" / f"seed{seed}" / "eval" / "omega_log.jsonl"
        mrps, _, _ = per_seed_metrics(str(path))
        fl_no_dp_rewards.append(mrps)
    fl_no_dp_rewards = np.array(fl_no_dp_rewards)

    sigmas_sorted = sorted(m3["results"].keys(), key=float)
    precision_by_sigma, reward_by_sigma = {}, {}
    for sigma_key in sigmas_sorted:
        sigma = float(sigma_key)
        rewards, precisions = [], []
        for seed in seeds:
            path = Path(args.m3_campaign_dir) / f"fl_gat_ctde_sigma{sigma}" / f"seed{seed}" / "eval" / "omega_log.jsonl"
            if not path.exists():
                continue
            mrps, mmtc_b, total_b = per_seed_metrics(str(path))
            rewards.append(mrps)
            if total_b > 0:
                precisions.append(mmtc_b / total_b)
        reward_by_sigma[sigma] = rewards
        precision_by_sigma[sigma] = precisions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---- (a) paired slope: centralized -> federated/no-DP, per seed ----
    diff_c_minus_f = centralized_rewards - fl_no_dp_rewards
    n_cost = int((diff_c_minus_f > 0).sum())   # centralized higher = federation cost this seed
    n_tie = int((diff_c_minus_f == 0).sum())
    n_help = int((diff_c_minus_f < 0).sum())   # federated higher = federation helped this seed

    for c, f in zip(centralized_rewards, fl_no_dp_rewards):
        d = c - f
        if d > 0:
            color, ls = STATUS_COLORS["critical"], "--"
        elif d < 0:
            color, ls = STATUS_COLORS["good"], "-"
        else:
            color, ls = STATUS_COLORS["neutral"], ":"
        ax1.plot([0, 1], [c, f], color=color, linestyle=ls, linewidth=1.1, alpha=0.65, zorder=2)

    mean_c, mean_f = centralized_rewards.mean(), fl_no_dp_rewards.mean()
    ax1.plot([0, 1], [mean_c, mean_f], color="#0b0b0b", linewidth=2.2, marker="D",
              markersize=6, zorder=5, label="Mean")

    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin - 0.34 * (ymax - ymin), ymax)
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Centralised\n(GAT-CTDE)", "Federated\n(no DP)"])
    ax1.set_ylabel("Mean reward per step")
    ax1.set_title(f"(a) Federation cost, per seed (n={len(seeds)})")

    diff_mean = diff_c_minus_f.mean()
    lo, hi = bootstrap_ci(diff_c_minus_f)
    _w_stat, w_p = stats.wilcoxon(centralized_rewards, fl_no_dp_rewards)
    ax1.text(0.5, 0.02,
              f"mean $\\Delta$=+{diff_mean:.3f} [{lo:.2f}, {hi:.2f}]\n"
              f"Wilcoxon $p$={w_p:.4f}",
              transform=ax1.transAxes, ha="center", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fcfcfb", edgecolor="#c3c2b7", linewidth=0.6))

    legend_handles = [
        Line2D([0], [0], color=STATUS_COLORS["critical"], linestyle="--", linewidth=1.4,
               label=f"Federation costs reward ({n_cost})"),
        Line2D([0], [0], color=STATUS_COLORS["neutral"], linestyle=":", linewidth=1.4,
               label=f"Tie ({n_tie})"),
        Line2D([0], [0], color=STATUS_COLORS["good"], linestyle="-", linewidth=1.4,
               label=f"Federation helps reward ({n_help})"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=False, handlelength=1.6, fontsize=6)

    # ---- (b) block_precision vs sigma, evenly-spaced categorical x, per-seed
    # jitter, n annotations (sigma is not spaced evenly -- 0/0.5/1/2/4 -- so a
    # linear numeric axis crowds 3 of 5 levels into a third of the width) ----
    rng = np.random.RandomState(0)
    sigma_vals = [float(s) for s in sigmas_sorted]
    x_cat = np.arange(len(sigma_vals))
    style = M3_STYLE["curve"]

    prec_means, prec_los, prec_his, n_seeds = [], [], [], []
    for xi, sigma in zip(x_cat, sigma_vals):
        p = precision_by_sigma[sigma]
        n_seeds.append(len(p))
        if p:
            v = np.array(p)
            jitter = rng.uniform(-0.14, 0.14, size=len(v))
            ax2.scatter(np.full(len(v), xi) + jitter, v, s=14, color=style["color"],
                        alpha=0.35, linewidth=0, zorder=2)
            lo, hi = bootstrap_ci(v)
            prec_means.append(v.mean())
            prec_los.append(v.mean() - lo)
            prec_his.append(hi - v.mean())
        else:
            prec_means.append(np.nan)
            prec_los.append(0)
            prec_his.append(0)

    ax2.errorbar(x_cat, prec_means, yerr=[prec_los, prec_his], color=style["color"],
                 marker=style["marker"], markersize=5.5, linewidth=1.6, capsize=3, zorder=4)
    for xi, n in zip(x_cat, n_seeds):
        ax2.annotate(f"n={n}", (xi, -0.17), ha="center", va="top", fontsize=5.8,
                     color="#898781", annotation_clip=False)

    ax2.set_xlim(-0.5, len(sigma_vals) - 0.5)
    ax2.set_xticks(x_cat)
    ax2.set_xticklabels([f"{s:g}" for s in sigma_vals])
    ax2.set_xlabel("DP noise multiplier $\\sigma$")
    ax2.set_ylabel("Block precision\n(fraction targeting mMTC)")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_title("(b) Privacy-utility (per-seed spread shown)")

    fig.suptitle("M3 federated GAT-CTDE: federation and privacy costs, per seed", y=1.02, fontsize=9)
    fig.subplots_adjust(bottom=0.28, wspace=0.32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig4] wrote {out_path}.pdf / .png")
    print(f"  centralized reward mean={centralized_rewards.mean():.3f}, "
          f"FL/no-DP reward mean={fl_no_dp_rewards.mean():.3f}")
    print(f"  paired centralized-federated: mean diff={diff_mean:.3f}, "
          f"cost={n_cost} tie={n_tie} help={n_help}")
    for sigma, n in zip(sigma_vals, n_seeds):
        p = precision_by_sigma[sigma]
        print(f"  sigma={sigma}: block_precision mean={np.mean(p) if p else float('nan'):.3f} (n={n})")


if __name__ == "__main__":
    main()
