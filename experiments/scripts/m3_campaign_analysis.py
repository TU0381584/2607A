#!/usr/bin/env python3
"""M3 analysis: privacy-utility curve (compliance vs noise multiplier),
plus two paired comparisons that separate the two costs this deliverable
is about:

  (1) centralized gat_ctde (Block E campaign, same 10 seeds) vs
      FL/no-DP (sigma=0.0) -- the pure federation cost (per-step joint
      training vs periodic-FedAvg local training), with no privacy noise
      involved at all.
  (2) FL/no-DP (sigma=0.0) vs each FL/DP sigma>0 level -- the pure privacy
      cost layered on top of federation.

Same bootstrap-CI / Wilcoxon methodology as m2_campaign_analysis.py (10k
resamples, nonparametric paired test), for direct comparability.

epsilon per noise level is the zCDP-composition upper bound
(qoe_oran_framework.marl.dp_sgd.zcdp_epsilon) computed from each seed's
OWN logged dp_step_count -- not assumed or invented -- at delta=1e-5. It
is explicitly a conservative (loose) bound, not a tight moments-accountant
figure (see dp_sgd.py's module docstring for why). sigma=0.0 has no
finite epsilon (no privacy) and is reported as such, not omitted.

Usage:
    python3 experiments/scripts/m3_campaign_analysis.py \
        --m3-results experiments/results/m3_campaign/campaign_results.json \
        --m2-results experiments/results/m2_campaign/campaign_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.marl.dp_sgd import zcdp_epsilon  # noqa: E402

DELTA = 1e-5


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_report(label, a, b, a_name, b_name):
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    lo, hi = bootstrap_ci(diff)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"  (Wilcoxon failed: {e})")
    print(f"=== paired: {label} ({a_name} - {b_name}, n={len(diff)}) ===")
    print(f"  mean paired diff = {diff.mean():.4f}, 95% bootstrap CI = [{lo:.4f}, {hi:.4f}]")
    print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.4f}")
    print(f"  {a_name} wins on {int((diff > 0).sum())}/{len(diff)}, ties on {int((diff == 0).sum())}, "
          f"{b_name} wins on {int((diff < 0).sum())}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3-results", default="experiments/results/m3_campaign/campaign_results.json")
    ap.add_argument("--m2-results", default="experiments/results/m2_campaign/campaign_results.json")
    args = ap.parse_args()

    with open(args.m3_results) as fh:
        m3 = json.load(fh)
    seeds = m3["seeds"]
    results = m3["results"]

    with open(args.m2_results) as fh:
        m2 = json.load(fh)
    gat_ctde_by_seed = m2["results"]["gat_ctde"]
    centralized = [gat_ctde_by_seed[str(s)]["sla_compliance_all_slices"] for s in seeds if str(s) in gat_ctde_by_seed]
    print(f"=== centralized gat_ctde (Block E, same {len(centralized)} seeds {seeds}) ===")
    v = np.array(centralized)
    lo, hi = bootstrap_ci(v)
    print(f"  mean = {v.mean():.4f}, 95% bootstrap CI = [{lo:.4f}, {hi:.4f}]\n")

    print("=== privacy-utility curve (FL/FedAvg arm) ===")
    per_sigma_values = {}
    per_sigma_epsilon = {}
    for sigma_key, by_seed in sorted(results.items(), key=lambda kv: float(kv[0])):
        sigma = float(sigma_key)
        values = [by_seed[str(s)]["sla_compliance_all_slices"] for s in seeds if str(s) in by_seed]
        per_sigma_values[sigma] = values
        v = np.array(values)
        lo, hi = bootstrap_ci(v)

        eps_per_seed = []
        for s in seeds:
            entry = by_seed.get(str(s))
            if entry is None:
                continue
            steps = max(entry["dp_step_count_per_client"]) if sigma > 0 else 0
            eps_per_seed.append(zcdp_epsilon(steps, sigma, DELTA))
        per_sigma_epsilon[sigma] = eps_per_seed
        eps_str = "inf (no DP)" if sigma == 0.0 else f"{np.mean(eps_per_seed):.2f} (zCDP upper bound, delta={DELTA})"

        print(f"--- sigma={sigma} (n={len(v)}), epsilon~{eps_str} ---")
        print(f"  mean = {v.mean():.4f}, 95% bootstrap CI = [{lo:.4f}, {hi:.4f}]")
        print(f"  median = {np.median(v):.4f}, std = {v.std():.4f}")
        print(f"  per-seed: {dict(zip(seeds, [round(x, 4) for x in values]))}")
        print()

    if 0.0 in per_sigma_values and len(centralized) == len(per_sigma_values[0.0]):
        paired_report("centralization cost: centralized gat_ctde vs FL/no-DP",
                       centralized, per_sigma_values[0.0], "centralized_gat_ctde", "fl_sigma0.0")

    if 0.0 in per_sigma_values:
        baseline = per_sigma_values[0.0]
        for sigma, values in sorted(per_sigma_values.items()):
            if sigma == 0.0 or len(values) != len(baseline):
                continue
            paired_report(f"privacy cost: FL/no-DP vs FL/DP sigma={sigma}",
                           baseline, values, "fl_sigma0.0", f"fl_sigma{sigma}")


if __name__ == "__main__":
    main()
