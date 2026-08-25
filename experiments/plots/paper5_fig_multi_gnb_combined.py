#!/usr/bin/env python3
"""Multi-gNB results figures for the WPC-scoped copy (paper5_wpc/) only --
the centralised M2 campaign (paper5_fig3_m2_campaign.py's two panels)
and the federated/DP M3 sweep (paper5_fig4_m3_privacy.py's two panels),
each as its own single-row 2-panel figure. Originally merged into one
2x2 figure per M20's "one combined multi-gNB results figure"
instruction; split back into two per M23's single-column legibility
pass, since a 2x2 grid is too cramped to read at Springer's
single-column width. Content is untouched from the two source scripts
-- same data, same statistics, same per-arm styling -- only the layout
changed and in-figure titles stay bare (a)/(b) tags (the LaTeX caption
carries the description, matching the rest of the WPC copy's figures).
Does not touch paper5/'s existing fig3/fig4 PDFs or their generating
scripts.

Figure 1 (M2, --out-m2):
(a) Paired per-seed comparison, single-agent DQN -> GAT-CTDE.
(b) Per-seed reward distribution, all three arms, GAT-CTDE split by
    collapse status.

Figure 2 (M3, --out-m3):
(a) Paired per-seed federation cost, centralised -> federated (no DP).
(b) Privacy-utility curve, block precision vs. DP noise multiplier.

Usage:
    python3 experiments/plots/paper5_fig_multi_gnb_combined.py \
        --m2-results experiments/results/m2_campaign/campaign_results.json \
        --m2-campaign-dir experiments/results/m2_campaign \
        --m3-results experiments/results/m3_campaign/campaign_results.json \
        --m3-campaign-dir experiments/results/m3_campaign \
        --out-m2 paper5_wpc/figures/fig3_m2_results \
        --out-m3 paper5_wpc/figures/fig4_m3_privacy
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
from paper5_common import M2_ARM_STYLE, M3_STYLE, STATUS_COLORS, bootstrap_ci, eval_omega_path, load_m2_campaign  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m2-results", default="experiments/results/m2_campaign/campaign_results.json")
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--m3-results", default="experiments/results/m3_campaign/campaign_results.json")
    ap.add_argument("--m3-campaign-dir", default="experiments/results/m3_campaign")
    ap.add_argument("--out-m2", default="paper5_wpc/figures/fig3_m2_results")
    ap.add_argument("--out-m3", default="paper5_wpc/figures/fig4_m3_privacy")
    args = ap.parse_args()

    # ---- load M2 ----
    all_seeds, _ = load_m2_campaign(args.m2_results)
    reward_by_arm, blocks_by_arm = {}, {}
    for arm in ["gat_ctde", "independent_dqn", "single_agent_dqn"]:
        rewards, blocks = [], []
        for s in all_seeds:
            path = eval_omega_path(args.m2_campaign_dir, arm, s)
            mrps, _mmtc_b, total_b = per_seed_metrics(str(path))
            rewards.append(mrps)
            blocks.append(total_b)
        reward_by_arm[arm] = np.array(rewards)
        blocks_by_arm[arm] = np.array(blocks)

    # ---- load M3 ----
    with open(args.m3_results) as fh:
        m3 = json.load(fh)
    m3_seeds = m3["seeds"]
    centralized_rewards = np.array([
        per_seed_metrics(str(Path(args.m2_campaign_dir) / "gat_ctde" / f"seed{s}" / "eval" / "omega_log.jsonl"))[0]
        for s in m3_seeds
    ])
    fl_no_dp_rewards = np.array([
        per_seed_metrics(str(Path(args.m3_campaign_dir) / "fl_gat_ctde_sigma0.0" / f"seed{s}" / "eval" / "omega_log.jsonl"))[0]
        for s in m3_seeds
    ])
    sigmas_sorted = sorted(m3["results"].keys(), key=float)
    sigma_vals = [float(s) for s in sigmas_sorted]
    precision_by_sigma = {}
    for sigma in sigma_vals:
        precisions = []
        for seed in m3_seeds:
            path = Path(args.m3_campaign_dir) / f"fl_gat_ctde_sigma{sigma}" / f"seed{seed}" / "eval" / "omega_log.jsonl"
            if not path.exists():
                continue
            _mrps, mmtc_b, total_b = per_seed_metrics(str(path))
            if total_b > 0:
                precisions.append(mmtc_b / total_b)
        precision_by_sigma[sigma] = precisions

    # ==================== Figure 1: M2 ====================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---- (a) M2 paired slope: single_agent_dqn -> gat_ctde ----
    single, gat = reward_by_arm["single_agent_dqn"], reward_by_arm["gat_ctde"]
    diff = gat - single
    n_win, n_tie, n_lose = int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())
    for s0, s1 in zip(single, gat):
        d = s1 - s0
        color, ls = (STATUS_COLORS["good"], "-") if d > 0 else (STATUS_COLORS["critical"], "--") if d < 0 else (STATUS_COLORS["neutral"], ":")
        ax1.plot([0, 1], [s0, s1], color=color, linestyle=ls, linewidth=0.9, alpha=0.55, zorder=2)
    mean0, mean1 = single.mean(), gat.mean()
    ax1.plot([0, 1], [mean0, mean1], color="#0b0b0b", linewidth=2.2, marker="D", markersize=6, zorder=5, label="Mean")
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Single-agent\nDQN", "GAT-CTDE\n(proposed)"])
    ax1.set_ylabel("Mean reward per step")
    ax1.set_title("(a)", loc="left")
    lo, hi = bootstrap_ci(diff)
    _w, p_ab = stats.wilcoxon(gat, single)
    ax1.text(0.5, 0.02, f"mean $\\Delta$=+{diff.mean():.3f} [{lo:.2f}, {hi:.2f}]\nWilcoxon $p$={p_ab:.4f}",
              transform=ax1.transAxes, ha="center", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fcfcfb", edgecolor="#c3c2b7", linewidth=0.6))
    ax1.legend(handles=[
        Line2D([0], [0], color=STATUS_COLORS["good"], linestyle="-", linewidth=1.4, label=f"GAT-CTDE higher ({n_win})"),
        Line2D([0], [0], color=STATUS_COLORS["neutral"], linestyle=":", linewidth=1.4, label=f"Tie ({n_tie})"),
        Line2D([0], [0], color=STATUS_COLORS["critical"], linestyle="--", linewidth=1.4, label=f"GAT-CTDE lower ({n_lose})"),
    ], loc="upper left", frameon=False, handlelength=1.6, fontsize=6.5)

    # ---- (b) M2 per-seed reward strip, all 3 arms ----
    rng = np.random.RandomState(0)
    x_pos = {"gat_ctde": 0, "independent_dqn": 1, "single_agent_dqn": 2}
    gat_diff_mask = blocks_by_arm["gat_ctde"] > 0
    n_diff, n_collapsed = int(gat_diff_mask.sum()), int((~gat_diff_mask).sum())
    jitter = rng.uniform(-0.13, 0.13, size=len(all_seeds))
    ax2.scatter(x_pos["gat_ctde"] + jitter[gat_diff_mask], reward_by_arm["gat_ctde"][gat_diff_mask],
                s=16, color=M2_ARM_STYLE["gat_ctde"]["color"], marker="o", alpha=0.75, linewidth=0, zorder=3,
                label=f"GAT-CTDE, differentiated ({n_diff})")
    ax2.scatter(x_pos["gat_ctde"] + jitter[~gat_diff_mask], reward_by_arm["gat_ctde"][~gat_diff_mask],
                s=22, color="#898781", marker="x", alpha=0.9, linewidth=1.1, zorder=3,
                label=f"GAT-CTDE, still collapsed ({n_collapsed})")
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
    ax2.set_title("(b)", loc="left")
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2.append(Line2D([0], [0], marker="D", color="#0b0b0b", linestyle="none", markersize=5, label="Mean $\\pm$ 95% CI"))
    labels2.append("Mean $\\pm$ 95% CI")
    ax2.legend(handles2, labels2, loc="lower center", bbox_to_anchor=(0.5, -0.62),
               ncol=1, frameon=False, fontsize=6, handlelength=1.3)

    fig1.subplots_adjust(bottom=0.34, wspace=0.32)

    out_m2 = Path(args.out_m2)
    out_m2.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_m2.with_suffix(".pdf"), bbox_inches="tight")
    fig1.savefig(out_m2.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig-m2] wrote {out_m2}.pdf / .png")
    print(f"  (a) paired single_agent_dqn->gat_ctde: mean diff={diff.mean():.3f}, p={p_ab:.4f}, win={n_win} tie={n_tie} lose={n_lose}")
    print(f"  (b) gat_ctde differentiated={n_diff}, still collapsed={n_collapsed}")

    # ==================== Figure 2: M3 ====================
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---- (a) M3 paired slope: centralised -> federated/no-DP ----
    diff_cf = centralized_rewards - fl_no_dp_rewards
    n_cost, n_tie3, n_help = int((diff_cf > 0).sum()), int((diff_cf == 0).sum()), int((diff_cf < 0).sum())
    for c, f in zip(centralized_rewards, fl_no_dp_rewards):
        d = c - f
        color, ls = (STATUS_COLORS["critical"], "--") if d > 0 else (STATUS_COLORS["good"], "-") if d < 0 else (STATUS_COLORS["neutral"], ":")
        ax3.plot([0, 1], [c, f], color=color, linestyle=ls, linewidth=1.1, alpha=0.65, zorder=2)
    mean_c, mean_f = centralized_rewards.mean(), fl_no_dp_rewards.mean()
    ax3.plot([0, 1], [mean_c, mean_f], color="#0b0b0b", linewidth=2.2, marker="D", markersize=6, zorder=5, label="Mean")
    ymin, ymax = ax3.get_ylim()
    ax3.set_ylim(ymin - 0.34 * (ymax - ymin), ymax)
    ax3.set_xlim(-0.25, 1.25)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Centralised\n(GAT-CTDE)", "Federated\n(no DP)"])
    ax3.set_ylabel("Mean reward per step")
    ax3.set_title("(a)", loc="left")
    diff_mean = diff_cf.mean()
    lo, hi = bootstrap_ci(diff_cf)
    _w, p_cf = stats.wilcoxon(centralized_rewards, fl_no_dp_rewards)
    ax3.text(0.5, 0.02, f"mean $\\Delta$=+{diff_mean:.3f} [{lo:.2f}, {hi:.2f}]\nWilcoxon $p$={p_cf:.4f}",
              transform=ax3.transAxes, ha="center", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fcfcfb", edgecolor="#c3c2b7", linewidth=0.6))
    ax3.legend(handles=[
        Line2D([0], [0], color=STATUS_COLORS["critical"], linestyle="--", linewidth=1.4, label=f"Federation costs reward ({n_cost})"),
        Line2D([0], [0], color=STATUS_COLORS["neutral"], linestyle=":", linewidth=1.4, label=f"Tie ({n_tie3})"),
        Line2D([0], [0], color=STATUS_COLORS["good"], linestyle="-", linewidth=1.4, label=f"Federation helps reward ({n_help})"),
    ], loc="upper left", frameon=False, handlelength=1.6, fontsize=6)

    # ---- (b) M3 block precision vs sigma ----
    rng = np.random.RandomState(0)
    x_cat = np.arange(len(sigma_vals))
    style = M3_STYLE["curve"]
    prec_means, prec_los, prec_his, n_seeds = [], [], [], []
    for xi, sigma in zip(x_cat, sigma_vals):
        p = precision_by_sigma[sigma]
        n_seeds.append(len(p))
        if p:
            v = np.array(p)
            jitter = rng.uniform(-0.14, 0.14, size=len(v))
            ax4.scatter(np.full(len(v), xi) + jitter, v, s=14, color=style["color"], alpha=0.35, linewidth=0, zorder=2)
            lo, hi = bootstrap_ci(v)
            prec_means.append(v.mean())
            prec_los.append(v.mean() - lo)
            prec_his.append(hi - v.mean())
        else:
            prec_means.append(np.nan)
            prec_los.append(0)
            prec_his.append(0)
    ax4.errorbar(x_cat, prec_means, yerr=[prec_los, prec_his], color=style["color"],
                 marker=style["marker"], markersize=5.5, linewidth=1.6, capsize=3, zorder=4)
    for xi, n in zip(x_cat, n_seeds):
        ax4.annotate(f"n={n}", (xi, -0.17), ha="center", va="top", fontsize=5.8, color="#898781", annotation_clip=False)
    ax4.set_xlim(-0.5, len(sigma_vals) - 0.5)
    ax4.set_xticks(x_cat)
    ax4.set_xticklabels([f"{s:g}" for s in sigma_vals])
    ax4.set_xlabel("DP noise multiplier $\\sigma$")
    ax4.set_ylabel("Block precision\n(fraction targeting mMTC)")
    ax4.set_ylim(-0.05, 1.08)
    ax4.set_title("(b)", loc="left")

    fig2.subplots_adjust(bottom=0.28, wspace=0.32)

    out_m3 = Path(args.out_m3)
    out_m3.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_m3.with_suffix(".pdf"), bbox_inches="tight")
    fig2.savefig(out_m3.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig-m3] wrote {out_m3}.pdf / .png")
    print(f"  (a) paired centralized-federated: mean diff={diff_mean:.3f}, p={p_cf:.4f}, cost={n_cost} tie={n_tie3} help={n_help}")
    for sigma, n in zip(sigma_vals, n_seeds):
        p = precision_by_sigma[sigma]
        print(f"  (b) sigma={sigma}: block_precision mean={np.mean(p) if p else float('nan'):.3f} (n={n})")


if __name__ == "__main__":
    main()
