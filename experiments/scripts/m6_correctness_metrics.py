#!/usr/bin/env python3
"""M6-specific correctness-aware metrics. Root cause this exists for
(docs/PAPER5_M6_topology.md's collapse-at-scale investigation): raw
mean_reward_per_step is NOT comparable across different N (cluster size)
values -- confirmed directly by reading action_mapping.py's
apply_actions(): accepted_counts[slice_id] sums accepted requests across
ALL gNBs in a single step (not per-gNB), so reward.compute_step_reward's
service_term = sum_slice(priority_weight * accept_reward * n_accepted)
scales mechanically with how many gNBs there are, independent of
decision quality -- the identical mechanism (accept-volume inflating raw
reward) M4's per-pending-request normalization already fixed for demand
spikes, showing up again on the N axis instead of the severity axis.

mean_reward_per_gnb below divides by N so the SAME per-agent-scale
quantity is compared across N=7 vs N=19, not a cluster-wide sum that
mechanically grows with cluster size. block_precision is untouched (it
was already volume-invariant -- a ratio -- which is exactly why M4 chose
it as ITS primary metric too).

Separately (also documented in PAPER5_M6_topology.md, not fixed here
because it is not this project's own instrumentation to normalize):
sla_compliance_all_slices is OR'd across every gNB per step
(reward.py::check_violations's own docstring says so directly) so it
structurally trends toward 0 as N grows regardless of policy quality --
the probability that AT LEAST ONE of N gNBs is briefly out of SLA on ANY
step rises with N by construction. This is reported, not corrected: it
is frozen source's own metric definition, and the point of this project's
correctness-aware metric pair was always to have an alternative that
does not inherit that specific structural flaw, not to patch the
flawed one.

Usage:
    python3 experiments/scripts/m6_correctness_metrics.py \
        --pilot-dir experiments/results/m6_pilot
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import bootstrap_ci  # noqa: E402


def per_seed_metrics_per_gnb(eval_omega_path: str, n_gnb: int):
    """Returns (mean_reward_per_gnb, mmtc_blocks, total_blocks). Same
    block-counting as m2_correctness_metrics.per_seed_metrics; reward is
    divided by n_gnb instead of left as a raw cluster-wide sum."""
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
                episode_step_rewards.setdefault(ep, []).append(ev["reward"] / n_gnb)

    per_episode_means = [float(np.mean(v)) for v in episode_step_rewards.values() if v]
    mean_reward_per_gnb = float(np.mean(per_episode_means)) if per_episode_means else float("nan")
    return mean_reward_per_gnb, mmtc_blocks, total_blocks


N_GNB_BY_COMBO = {
    "n7_fully_connected": 7, "n7_ring": 7, "n7_hex": 7,
    "n19_fully_connected": 19, "n19_ring": 19, "n19_hex": 19,
}
SEEDS = [900, 901, 902]
ARMS = ["gat_ctde", "independent_dqn", "single_agent_dqn"]


def eval_path(pilot_dir: Path, combo: str, arm: str, seed: int) -> Path:
    base = pilot_dir / combo / arm / f"seed{seed}" / "eval"
    flat = base / "omega_log.jsonl"
    if flat.exists():
        return flat
    return base / "dqn" / "offline_eval" / "rep_0" / "omega_log.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", default="experiments/results/m6_pilot")
    args = ap.parse_args()
    pilot_dir = Path(args.pilot_dir)

    print(f"{'combo':<22}{'arm':<18}{'mean_reward/gNB':<20}{'block_precision':<18}")
    reward_by = {}
    for combo, n_gnb in N_GNB_BY_COMBO.items():
        for arm in ARMS:
            vals, mmtc_total, all_total = [], 0, 0
            for seed in SEEDS:
                p = eval_path(pilot_dir, combo, arm, seed)
                if not p.exists():
                    continue
                mrpg, mmtc_b, total_b = per_seed_metrics_per_gnb(str(p), n_gnb)
                vals.append(mrpg)
                mmtc_total += mmtc_b
                all_total += total_b
            if not vals:
                continue
            v = np.array(vals)
            prec = mmtc_total / all_total if all_total > 0 else float("nan")
            reward_by[(combo, arm)] = v
            print(f"{combo:<22}{arm:<18}{v.mean():<20.4f}{prec:<18.3f}")
        print()

    print("=== paired GAT-CTDE - single-agent DQN, per-gNB-normalized reward (n=3, DIRECTIONAL ONLY) ===")
    for combo in N_GNB_BY_COMBO:
        g = reward_by.get((combo, "gat_ctde"))
        s = reward_by.get((combo, "single_agent_dqn"))
        if g is None or s is None:
            continue
        diff = g - s
        lo, hi = bootstrap_ci(diff)
        print(f"{combo:<22} mean diff={diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  per-seed={[f'{d:+.4f}' for d in diff]}")


if __name__ == "__main__":
    main()
