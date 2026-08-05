#!/usr/bin/env python3
"""M2 hardening (Block E, task 2): analysis of the seed-campaign results
written by m2_seed_campaign.py. Reports, per arm: mean compliance, 95%
bootstrap percentile CI (not a normal-approximation CI -- compliance is
bounded [0,1] and empirically bimodal-ish, not normally distributed, per
the preliminary 3-seed result already in docs/PAPER5_M2_gat_ctde.md
section 3), and the full per-seed distribution (not just min/max range).
Also reports a paired per-seed comparison of gat_ctde vs single_agent_dqn
(same 30 seeds, same env realization per seed across arms, so a natural
pairing exists) via a paired bootstrap CI on the mean difference and a
Wilcoxon signed-rank test, matching this project's own established
preference for nonparametric tests (Fisher exact elsewhere) over
normal-approximation methods.

Usage:
    python3 experiments/scripts/m2_campaign_analysis.py \
        --results experiments/results/m2_campaign/campaign_results.json
"""
import argparse
import json

import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results/m2_campaign/campaign_results.json")
    args = ap.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    results = data["results"]
    seed_groups = data["seed_groups"]
    all_seeds = [s for g in seed_groups for s in g]

    per_arm_values = {}
    for arm, by_seed in results.items():
        values = [by_seed[str(s)]["sla_compliance_all_slices"] for s in all_seeds if str(s) in by_seed]
        per_arm_values[arm] = values
        v = np.array(values)
        lo, hi = bootstrap_ci(v)
        print(f"=== {arm} (n={len(v)}) ===")
        print(f"  mean = {v.mean():.4f}, 95% bootstrap CI = [{lo:.4f}, {hi:.4f}]")
        print(f"  median = {np.median(v):.4f}, std = {v.std():.4f}")
        print(f"  per-seed distribution: {dict(zip(all_seeds, [round(x, 4) for x in values]))}")
        print()

    if "gat_ctde" in per_arm_values and "single_agent_dqn" in per_arm_values:
        gat = np.array(per_arm_values["gat_ctde"])
        single = np.array(per_arm_values["single_agent_dqn"])
        diff = gat - single
        lo, hi = bootstrap_ci(diff)
        try:
            w_stat, w_p = stats.wilcoxon(gat, single)
        except ValueError as e:
            w_stat, w_p = float("nan"), float("nan")
            print(f"  (Wilcoxon failed: {e})")
        print("=== paired: gat_ctde - single_agent_dqn (same 30 seeds) ===")
        print(f"  mean paired diff = {diff.mean():.4f}, 95% bootstrap CI = [{lo:.4f}, {hi:.4f}]")
        print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.4f}")
        print(f"  gat_ctde wins (higher compliance) on {int((diff > 0).sum())}/{len(diff)} seeds, "
              f"ties on {int((diff == 0).sum())}, loses on {int((diff < 0).sum())}")


if __name__ == "__main__":
    main()
