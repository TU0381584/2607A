#!/usr/bin/env python3
"""Groundwork for B1's still-unswept beta decision: runs accept_all and
reject_all against the frozen admission-efficiency config, capturing the
RAW eq.9 components (mos_norm, cost, sla_viol) per step, then recomputes
counterfactual reward under a grid of candidate beta values -- all
arithmetic on real, freshly-collected (post-bugfix, post-recalibration)
data, zero additional training.

CAVEAT, stated directly: this shows what reward a FIXED scripted policy's
ALREADY-OBSERVED behavior would have scored under a different beta -- not
what a policy trained under that beta would learn to do differently. It
answers "does cost currently dominate the reward signal" (yes/no), not
"what is beta's true optimal value" -- that still needs real training
curves. See CAMPAIGN_LOG.md's 2026-07-20 entries for why the EARLIER
retroactive beta sweep (against the old, buggy offline data) was void;
this one is against corrected data but carries the same methodological
caveat in kind, just not in cause.

Usage:
    python3 experiments/scripts/beta_sensitivity_probe.py \
        --seeds 256 257 258 --episodes 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import numpy as np  # noqa: E402
from admission_efficiency_env import make_env  # noqa: E402

BETAS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
ALPHA, GAMMA = 1.0, 0.5


def accept_all_decide(pending, cluster_state):
    return [1 for _ in pending]


def reject_all_decide(pending, cluster_state):
    return [0 for _ in pending]


def collect(decide_fn, seed, n_episodes):
    env = make_env(seed=seed, reward_mode="qoe")
    mos_norm, cost, sla_viol = [], [], []
    for _ in range(n_episodes):
        env.reset()
        for _ in range(env.cfg.episode.steps_per_episode):
            pending = env.pending_requests()
            actions = decide_fn(pending, env.last_cluster_state)
            result = env.step(actions)
            rb = result.info.get("reward_breakdown", {})
            if "mean_mos" in rb:
                mos_norm.append((rb["mean_mos"] - 1.0) / 4.0)
            if "cost" in rb:
                cost.append(rb["cost"])
            if "sla_viol" in rb:
                sla_viol.append(rb["sla_viol"])
    return np.array(mos_norm), np.array(cost), np.array(sla_viol)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    print("Pooled per-step components (mean), and counterfactual reward per candidate beta:\n")
    for name, fn in [("accept_all", accept_all_decide), ("reject_all", reject_all_decide)]:
        mos_all, cost_all, sla_all = [], [], []
        for seed in args.seeds:
            m, c, s = collect(fn, seed, args.episodes)
            mos_all.append(m); cost_all.append(c); sla_all.append(s)
        mos_norm = np.concatenate(mos_all)
        cost = np.concatenate(cost_all)
        sla_viol = np.concatenate(sla_all)
        print(f"=== {name} === mean mos_norm={mos_norm.mean():.4f}  mean cost={cost.mean():.4f}  "
              f"mean sla_viol={sla_viol.mean():.4f}")
        for beta in BETAS:
            r = ALPHA * mos_norm - beta * cost - GAMMA * sla_viol
            marker = "  <- current placeholder" if beta == 0.2 else ""
            print(f"  beta={beta:<5} counterfactual mean reward={r.mean():8.4f}  "
                  f"(alpha*mos={ALPHA*mos_norm.mean():.4f}, beta*cost={beta*cost.mean():.4f}, "
                  f"gamma*sla_viol={GAMMA*sla_viol.mean():.4f}){marker}")
        print()


if __name__ == "__main__":
    main()
