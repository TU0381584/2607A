#!/usr/bin/env python3
"""The formal paired test Papers_4-5/Paper_5/IEEE_Access/main.tex's Conclusion flags as missing:
"no formal paired test between architectures' disruption costs was run
here, only each architecture against its own undisrupted baseline."

For each (kind, severity), computes each architecture's own per-seed
disruption cost (baseline - disrupted, mean_reward_per_pending_request,
the same normalised metric Section~\\ref{sec:results-m4} already uses)
against its OWN undisrupted baseline (matching that section's existing
methodology exactly, m4_correctness_metrics.py's per_seed_metrics_normalized
and baseline_eval_path reused, not reimplemented), then runs a paired
Wilcoxon signed-rank test on (gat_ctde_cost - independent_dqn_cost)
across the SAME 10 seeds -- the comparison the paper's own text says was
never run.

Usage:
    python3 experiments/scripts/m4_cross_architecture_paired_test.py \
        --m4-campaign-dir experiments/results/m4_paired_test \
        --m2-campaign-dir experiments/results/m2_campaign
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import bootstrap_ci  # noqa: E402
from m4_correctness_metrics import baseline_eval_path, per_seed_metrics_normalized  # noqa: E402

SEEDS = list(range(900, 910))
KINDS = ["dropout", "churn"]
SEVERITIES = [1, 2, 3]
ARM_A, ARM_B = "gat_ctde", "independent_dqn"


def disrupted_eval_path(m4_campaign_dir: str, arm: str, kind: str, severity: int, seed: int) -> str:
    return f"{m4_campaign_dir}/{arm}/{kind}_sev{severity}/seed{seed}/eval/omega_log.jsonl"


def per_seed_cost(m4_campaign_dir: str, m2_campaign_dir: str, arm: str, kind: str, severity: int, seed: int) -> float:
    baseline_mrpr, _, _ = per_seed_metrics_normalized(baseline_eval_path(arm, seed, m2_campaign_dir=m2_campaign_dir))
    disrupted_mrpr, _, _ = per_seed_metrics_normalized(disrupted_eval_path(m4_campaign_dir, arm, kind, severity, seed))
    return baseline_mrpr - disrupted_mrpr  # cost: positive = disruption hurt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m4-campaign-dir", default="experiments/results/m4_paired_test")
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    args = ap.parse_args()

    print(f"{'kind':<10}{'sev':<5}{ARM_A+'_mean':<14}{ARM_B+'_mean':<14}"
          f"{'diff_mean':<12}{'diff_CI':<24}{'wilcoxon_p':<12}{'win/tie/loss':<14}")
    for kind in KINDS:
        for sev in SEVERITIES:
            costs_a = np.array([per_seed_cost(args.m4_campaign_dir, args.m2_campaign_dir, ARM_A, kind, sev, s) for s in SEEDS])
            costs_b = np.array([per_seed_cost(args.m4_campaign_dir, args.m2_campaign_dir, ARM_B, kind, sev, s) for s in SEEDS])
            diff = costs_a - costs_b  # positive = gat_ctde hurt MORE by this disruption than independent_dqn
            lo, hi = bootstrap_ci(diff)
            try:
                _w, p = stats.wilcoxon(costs_a, costs_b)
            except ValueError:
                p = float("nan")  # all-zero-difference edge case
            n_a_worse = int((diff > 0).sum())
            n_tie = int((diff == 0).sum())
            n_b_worse = int((diff < 0).sum())
            print(f"{kind:<10}{sev:<5}{costs_a.mean():<14.4f}{costs_b.mean():<14.4f}"
                  f"{diff.mean():<+12.4f}[{lo:+.4f},{hi:+.4f}]{'':<4}{p:<12.4f}"
                  f"{n_a_worse}/{n_tie}/{n_b_worse}")


if __name__ == "__main__":
    main()
