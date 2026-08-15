#!/usr/bin/env python3
"""M3 correctness-aware metrics, mirroring m2_correctness_metrics.py's
mean_reward_per_step / block_precision definitions exactly (same
underlying per_seed_metrics function, reused not duplicated) -- added for
the same reason: sla_compliance_all_slices structurally rewards the
"accept everything" collapse over genuinely-correct differentiated
shedding (docs/PAPER5_M2_gat_ctde.md section 11/12), so the privacy-
utility story needs the same correctness lens the centralized campaign
got, not just the compliance-based curve m3_campaign_analysis.py reports.

Usage:
    python3 experiments/scripts/m3_correctness_metrics.py \
        --m3-results experiments/results/m3_campaign/campaign_results.json \
        --m2-results experiments/results/m2_campaign/campaign_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import bootstrap_ci, per_seed_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m3-results", default="experiments/results/m3_campaign/campaign_results.json")
    ap.add_argument("--m3-campaign-dir", default="experiments/results/m3_campaign")
    ap.add_argument("--m2-results", default="experiments/results/m2_campaign/campaign_results.json")
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    args = ap.parse_args()

    with open(args.m3_results) as fh:
        m3 = json.load(fh)
    seeds = m3["seeds"]

    # Centralized gat_ctde reference (same 10 seeds), for the federation-cost comparison.
    with open(args.m2_results) as fh:
        m2 = json.load(fh)
    centralized_rewards, centralized_precisions = [], []
    for seed in seeds:
        path = Path(args.m2_campaign_dir) / "gat_ctde" / f"seed{seed}" / "eval" / "omega_log.jsonl"
        mrps, mmtc_b, total_b = per_seed_metrics(str(path))
        centralized_rewards.append(mrps)
        if total_b > 0:
            centralized_precisions.append(mmtc_b / total_b)
    r = np.array(centralized_rewards)
    lo, hi = bootstrap_ci(r)
    print(f"=== centralized gat_ctde (same {len(seeds)} seeds) ===")
    print(f"  mean_reward_per_step: mean={r.mean():.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")
    if centralized_precisions:
        p = np.array(centralized_precisions)
        plo, phi = bootstrap_ci(p)
        print(f"  block_precision: mean={p.mean():.3f}, 95% CI=[{plo:.3f}, {phi:.3f}], "
              f"n_seeds_with_any_blocks={len(centralized_precisions)}/{len(seeds)}")
    print()

    per_sigma_rewards = {}
    for sigma_key in sorted(m3["results"].keys(), key=float):
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
        per_sigma_rewards[sigma] = rewards
        r = np.array(rewards)
        lo, hi = bootstrap_ci(r)
        print(f"--- sigma={sigma} (n={len(r)}) ---")
        print(f"  mean_reward_per_step: mean={r.mean():.3f}, 95% CI=[{lo:.3f}, {hi:.3f}], median={np.median(r):.3f}")
        if precisions:
            p = np.array(precisions)
            plo, phi = bootstrap_ci(p)
            print(f"  block_precision: mean={p.mean():.3f}, 95% CI=[{plo:.3f}, {phi:.3f}], "
                  f"n_seeds_with_any_blocks={len(precisions)}/{len(rewards)}")
        else:
            print(f"  block_precision: UNDEFINED (0/{len(rewards)} ever blocked anything)")
        print()

    if 0.0 in per_sigma_rewards and len(centralized_rewards) == len(per_sigma_rewards[0.0]):
        a, b = np.array(centralized_rewards), np.array(per_sigma_rewards[0.0])
        diff = a - b
        lo, hi = bootstrap_ci(diff)
        w_stat, w_p = stats.wilcoxon(a, b)
        print("=== paired: centralization cost (mean_reward_per_step), centralized - FL/no-DP ===")
        print(f"  mean diff = {diff.mean():.3f}, 95% CI = [{lo:.3f}, {hi:.3f}], "
              f"Wilcoxon p={w_p:.4f}, centralized wins {int((diff>0).sum())}/10, "
              f"ties {int((diff==0).sum())}, FL/no-DP wins {int((diff<0).sum())}")
        print()

    if 0.0 in per_sigma_rewards:
        baseline = per_sigma_rewards[0.0]
        for sigma, values in sorted(per_sigma_rewards.items()):
            if sigma == 0.0 or len(values) != len(baseline):
                continue
            a, b = np.array(baseline), np.array(values)
            diff = a - b
            lo, hi = bootstrap_ci(diff)
            w_stat, w_p = stats.wilcoxon(a, b)
            print(f"=== paired: privacy cost (mean_reward_per_step), FL/no-DP - FL/DP sigma={sigma} ===")
            print(f"  mean diff = {diff.mean():.3f}, 95% CI = [{lo:.3f}, {hi:.3f}], "
                  f"Wilcoxon p={w_p:.4f}, no-DP wins {int((diff>0).sum())}/10, "
                  f"ties {int((diff==0).sum())}, DP wins {int((diff<0).sum())}")


if __name__ == "__main__":
    main()
