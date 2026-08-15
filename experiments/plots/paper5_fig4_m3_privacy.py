#!/usr/bin/env python3
"""Figure 4: M3 federated + DP results, two panels --
(a) federation cost (centralized GAT-CTDE vs. federated/no-DP,
mean_reward_per_step, same 10 seeds) and
(b) block_precision vs. DP noise multiplier sigma -- the privacy-utility
curve docs/PAPER5_M3_fl_dp.md found to be the trustworthy one (the
compliance-based sweep's sign flips between sigma levels; block_precision
shows a clean, interpretable threshold instead).

Reuses m2_correctness_metrics.per_seed_metrics (imported, not
reimplemented) for both mean_reward_per_step and block_precision.

Usage:
    python3 experiments/plots/paper5_fig4_m3_privacy.py \
        --m3-results experiments/results/m3_campaign/campaign_results.json \
        --m3-campaign-dir experiments/results/m3_campaign \
        --m2-results experiments/results/m2_campaign/campaign_results.json \
        --m2-campaign-dir experiments/results/m2_campaign \
        --out paper5/figures/fig4_m3_privacy
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m2_correctness_metrics import per_seed_metrics  # noqa: E402
from paper5_common import M3_STYLE, bootstrap_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m3-results", default="experiments/results/m3_campaign/campaign_results.json")
    ap.add_argument("--m3-campaign-dir", default="experiments/results/m3_campaign")
    ap.add_argument("--m2-results", default="experiments/results/m2_campaign/campaign_results.json")
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--out", default="paper5/figures/fig4_m3_privacy")
    args = ap.parse_args()

    with open(args.m3_results) as fh:
        m3 = json.load(fh)
    seeds = m3["seeds"]

    centralized_rewards = []
    for seed in seeds:
        path = Path(args.m2_campaign_dir) / "gat_ctde" / f"seed{seed}" / "eval" / "omega_log.jsonl"
        mrps, _, _ = per_seed_metrics(str(path))
        centralized_rewards.append(mrps)

    sigmas_sorted = sorted(m3["results"].keys(), key=float)
    fl_no_dp_rewards = []
    precision_by_sigma = {}
    reward_by_sigma = {}
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
        if sigma == 0.0:
            fl_no_dp_rewards = rewards

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    # (a) federation cost
    labels = [M3_STYLE["centralized"]["label"], M3_STYLE["federated"]["label"]]
    colors = [M3_STYLE["centralized"]["color"], M3_STYLE["federated"]["color"]]
    means, los, his = [], [], []
    for vals in [centralized_rewards, fl_no_dp_rewards]:
        v = np.array(vals)
        lo, hi = bootstrap_ci(v)
        means.append(v.mean())
        los.append(v.mean() - lo)
        his.append(hi - v.mean())
    x = np.arange(2)
    ax1.bar(x, means, yerr=[los, his], color=colors, width=0.55, capsize=3,
            edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Centralized", "Federated\n(no DP)"])
    ax1.set_ylabel("Mean reward per step")
    ax1.set_title(f"(a) Federation cost (n={len(seeds)})")

    # (b) block_precision vs sigma
    sigma_vals = [float(s) for s in sigmas_sorted]
    prec_means, prec_los, prec_his = [], [], []
    for sigma in sigma_vals:
        p = precision_by_sigma[sigma]
        if p:
            v = np.array(p)
            lo, hi = bootstrap_ci(v)
            prec_means.append(v.mean())
            prec_los.append(v.mean() - lo)
            prec_his.append(hi - v.mean())
        else:
            prec_means.append(np.nan)
            prec_los.append(0)
            prec_his.append(0)
    style = M3_STYLE["curve"]
    ax2.errorbar(sigma_vals, prec_means, yerr=[prec_los, prec_his], color=style["color"],
                 marker=style["marker"], markersize=5, linewidth=1.4, capsize=3)
    ax2.set_xlabel("DP noise multiplier $\\sigma$")
    ax2.set_ylabel("Block precision\n(fraction targeting mMTC)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Privacy-utility (block precision)")

    fig.suptitle("M3 federated GAT-CTDE: federation and privacy costs", y=1.04, fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig4] wrote {out_path}.pdf / .png")
    print(f"  centralized reward mean={np.mean(centralized_rewards):.3f}, "
          f"FL/no-DP reward mean={np.mean(fl_no_dp_rewards):.3f}")
    for sigma in sigma_vals:
        p = precision_by_sigma[sigma]
        print(f"  sigma={sigma}: block_precision mean={np.mean(p) if p else float('nan'):.3f} (n={len(p)})")


if __name__ == "__main__":
    main()
