#!/usr/bin/env python3
"""M2 correctness-aware metrics, added alongside sla_compliance_all_slices
after finding that metric structurally rewards the "accept everything"
collapse over genuinely-correct differentiated shedding (blocking mmtc
under congestion never rescues mmtc's own SLA margin in this stress
regime -- see docs/PAPER5_M2_gat_ctde.md section 11 -- so it only ever
costs compliance, never earns it back). Both metrics below are computed
from data ALREADY LOGGED in every arm's existing eval omega_log.jsonl --
no new instrumentation, no re-running any campaign, no invented formula:

1. mean_reward_per_step: the actual RL training objective (the same
   quantity qoe_oran_framework.mc_runner.run_single computes and reports
   as RunSummary.mean_reward_per_step -- per episode, mean of every
   step's logged `reward`, then mean across episodes). The reward
   function's own calibration (saclb_offline_dqn.yaml's congestion_coeff/
   priority_weight comments) is what defines "differentiated shedding is
   correct" in the first place, so this directly tests whether a policy
   is doing what it was actually trained to maximize -- not a proxy for
   it.

2. block_precision: of every request an arm blocks in eval, what
   fraction target mmtc specifically -- the only slice the reward's own
   calibration says is ever reject-optimal under congestion (urllc:
   "always worth accepting"; embb: "marginal, mostly accept"). Computed
   from each eval run's existing per-episode episode_block_by_slice
   rollup field. Undefined (reported as such, not silently coerced to 0
   or 1) for seeds with zero blocks -- a collapsed ("always accept")
   seed has no blocking decisions to be precise or imprecise about.

Usage:
    python3 experiments/scripts/m2_correctness_metrics.py \
        --campaign-dir experiments/results/m2_campaign \
        --results experiments/results/m2_campaign/campaign_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_seed_metrics(eval_omega_path: str):
    """Returns (mean_reward_per_step, mmtc_blocks, total_blocks)."""
    episode_step_rewards = {}
    mmtc_blocks = 0
    total_blocks = 0
    with open(eval_omega_path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", rec)
            if not isinstance(ev, dict):
                continue
            if ev.get("rollup"):
                for slice_id, n in ev.get("episode_block_by_slice", {}).items():
                    total_blocks += n
                    if slice_id == "mmtc":
                        mmtc_blocks += n
            elif "reward" in ev:
                ep = rec["episode"]
                episode_step_rewards.setdefault(ep, []).append(ev["reward"])

    per_episode_means = [float(np.mean(v)) for v in episode_step_rewards.values() if v]
    mean_reward_per_step = float(np.mean(per_episode_means)) if per_episode_means else float("nan")
    return mean_reward_per_step, mmtc_blocks, total_blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--results", default="experiments/results/m2_campaign/campaign_results.json")
    args = ap.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    seed_groups = data["seed_groups"]
    all_seeds = [s for g in seed_groups for s in g]
    arms = list(data["results"].keys())

    per_arm_rewards = {}
    for arm in arms:
        rewards, precisions, seeds_with_blocks, seeds_total = [], [], 0, 0
        for seed in all_seeds:
            seed_dir = Path(args.campaign_dir) / arm / f"seed{seed}" / "eval"
            flat_path = seed_dir / "omega_log.jsonl"
            # single_agent_dqn runs via mc_runner.run_mc, which nests its
            # output under dqn/offline_eval/rep_0/ -- the two new marl
            # policies (gat_ctde, independent_dqn) write flat eval/omega_log.jsonl.
            nested_path = seed_dir / "dqn" / "offline_eval" / "rep_0" / "omega_log.jsonl"
            path = flat_path if flat_path.exists() else nested_path
            if not path.exists():
                continue
            mrps, mmtc_b, total_b = per_seed_metrics(str(path))
            rewards.append(mrps)
            seeds_total += 1
            if total_b > 0:
                precisions.append(mmtc_b / total_b)
                seeds_with_blocks += 1
        per_arm_rewards[arm] = rewards

        r = np.array(rewards)
        lo, hi = bootstrap_ci(r)
        print(f"=== {arm} (n={len(r)}) ===")
        print(f"  mean_reward_per_step: mean={r.mean():.3f}, 95% CI=[{lo:.3f}, {hi:.3f}], "
              f"median={np.median(r):.3f}")
        if precisions:
            p = np.array(precisions)
            plo, phi = bootstrap_ci(p)
            print(f"  block_precision (mmtc-fraction-of-blocks): mean={p.mean():.3f}, "
                  f"95% CI=[{plo:.3f}, {phi:.3f}], n_seeds_with_any_blocks={seeds_with_blocks}/{seeds_total}")
        else:
            print(f"  block_precision: UNDEFINED for all seeds (0/{seeds_total} ever blocked anything)")
        print()

    if "gat_ctde" in per_arm_rewards and "single_agent_dqn" in per_arm_rewards:
        gat = np.array(per_arm_rewards["gat_ctde"])
        single = np.array(per_arm_rewards["single_agent_dqn"])
        diff = gat - single
        lo, hi = bootstrap_ci(diff)
        w_stat, w_p = stats.wilcoxon(gat, single)
        print("=== paired: gat_ctde - single_agent_dqn, mean_reward_per_step (same 30 seeds) ===")
        print(f"  mean paired diff = {diff.mean():.3f}, 95% bootstrap CI = [{lo:.3f}, {hi:.3f}]")
        print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.4f}")
        print(f"  gat_ctde wins on {int((diff > 0).sum())}/{len(diff)}, ties on {int((diff == 0).sum())}, "
              f"loses on {int((diff < 0).sum())}")


if __name__ == "__main__":
    main()
