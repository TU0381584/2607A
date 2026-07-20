#!/usr/bin/env python3
"""A1's pre-registered validity check for the merged "admission efficiency
under overload" objective (experiments/NOTE_admission_objective_merge.md),
run against the frozen experiments/configs/saclb_admission_efficiency_v1.yaml
via admission_efficiency_env.make_env().

Zero training: runs 3 SCRIPTED, non-learning policies --
  - accept_all: admits every pending request.
  - reject_all: rejects every pending request.
  - static_threshold: the framework's own LbOnlyHeuristic comparator
    (qoe_oran_framework/comparators/lb_only_baseline.py) -- reject if the
    request's gNB is saturated OR its own slice is at/above quota, no
    learning, no SLA-priority weighting. NOT frozen-code-modified; called
    exactly as the framework exposes it.

Validity criterion (this objective's version of A1's original one): a
genuine overload regime should show real, non-saturated, per-slice
DIFFERENTIATION in SLA compliance between these three policies -- not all
three pinned near 0% (too harsh) or all three near 100% (too easy).

Usage:
    python3 experiments/scripts/run_admission_efficiency_baselines.py \
        --seeds 256 257 258 --episodes 10 \
        --out experiments/results/admission_efficiency/baseline_validity.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import numpy as np  # noqa: E402
from admission_efficiency_env import make_env  # noqa: E402
from qoe_oran_framework.comparators.lb_only_baseline import LbOnlyHeuristic  # noqa: E402

SLICE_ORDER = ["embb", "urllc", "mmtc"]


def accept_all_decide(pending, cluster_state):
    return [1 for _ in pending]


def reject_all_decide(pending, cluster_state):
    return [0 for _ in pending]


def run_policy(decide_fn, seed, n_episodes, needs_cfg=False, cfg=None):
    """NOTE (2026-07-20 correction): compliance is read from
    reward_breakdown["per_slice_compliant"] -- the framework's OWN boolean
    (queue_len_norm <= 1.0, non-strict), matching exactly how
    mc_runner.py's episode_sla_compliance_by_slice / every other figure
    and table in this project computes it. An earlier version of this
    script recomputed compliance as `per_slice_sla_margin > 0` (STRICT),
    which silently undercounts: margins in this environment sit almost
    exactly AT the 0.0 boundary the vast majority of the time (confirmed:
    91-98% of steps in a spot-checked trained-arm episode), so a strict
    `>0` check disagreed sharply with the framework's own non-strict
    compliance accounting. See CAMPAIGN_LOG.md's 2026-07-20 correction
    entry -- this bug was caught by comparing a training run's rollup
    episode_sla_compliance_by_slice (100%) against this script's own
    recomputed number (16-70%) on what should have been comparable data."""
    env = make_env(seed=seed, reward_mode="qoe")
    per_slice_margin = {s: [] for s in SLICE_ORDER}
    per_slice_compliant = {s: [] for s in SLICE_ORDER}
    blocks = {s: 0 for s in SLICE_ORDER}
    total_reqs = {s: 0 for s in SLICE_ORDER}
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        for _ in range(env.cfg.episode.steps_per_episode):
            pending = env.pending_requests()
            actions = decide_fn(pending, env.last_cluster_state)
            for req, act in zip(pending, actions):
                total_reqs[req.slice_id] += 1
                if act == 0:
                    blocks[req.slice_id] += 1
            result = env.step(actions)
            rewards.append(result.reward)
            rb = result.info.get("reward_breakdown", {})
            for s, m in rb.get("per_slice_sla_margin", {}).items():
                per_slice_margin[s].append(m)
            for s, c in rb.get("per_slice_compliant", {}).items():
                per_slice_compliant[s].append(bool(c))
    return per_slice_margin, per_slice_compliant, blocks, total_reqs, rewards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="experiments/results/admission_efficiency/baseline_validity.md")
    args = ap.parse_args()

    policies = {
        "accept_all": accept_all_decide,
        "reject_all": reject_all_decide,
    }

    rows = []
    for name, decide_fn in policies.items():
        agg_compliant = {s: [] for s in SLICE_ORDER}
        agg_blocks = {s: 0 for s in SLICE_ORDER}
        agg_reqs = {s: 0 for s in SLICE_ORDER}
        agg_reward = []
        for seed in args.seeds:
            _, c, b, r, rew = run_policy(decide_fn, seed, args.episodes)
            for s in SLICE_ORDER:
                agg_compliant[s].extend(c[s])
                agg_blocks[s] += b[s]
                agg_reqs[s] += r[s]
            agg_reward.extend(rew)
        rows.append((name, agg_compliant, agg_blocks, agg_reqs, agg_reward))

    # static_threshold needs the loaded cfg for LbOnlyHeuristic -- build once.
    env0 = make_env(seed=args.seeds[0], reward_mode="qoe")
    heuristic = LbOnlyHeuristic(env0.cfg)

    def static_threshold_decide(pending, cluster_state):
        return heuristic.decide(pending, cluster_state)

    agg_compliant = {s: [] for s in SLICE_ORDER}
    agg_blocks = {s: 0 for s in SLICE_ORDER}
    agg_reqs = {s: 0 for s in SLICE_ORDER}
    agg_reward = []
    for seed in args.seeds:
        _, c, b, r, rew = run_policy(static_threshold_decide, seed, args.episodes)
        for s in SLICE_ORDER:
            agg_compliant[s].extend(c[s])
            agg_blocks[s] += b[s]
            agg_reqs[s] += r[s]
        agg_reward.extend(rew)
    rows.append(("static_threshold", agg_compliant, agg_blocks, agg_reqs, agg_reward))

    lines = [
        "# Admission-efficiency baseline validity check",
        "",
        f"Config: `experiments/configs/saclb_admission_efficiency_v1.yaml` "
        f"(backlog_capacity=1000.0, oversub_of_cap=1.2 -- see admission_efficiency_env.py)",
        f"Seeds: {args.seeds}, episodes/seed: {args.episodes}",
        "",
        "Compliance = `reward_breakdown['per_slice_compliant']` (queue_len_norm <= 1.0, "
        "non-strict) -- the SAME field mc_runner.py's episode_sla_compliance_by_slice uses, "
        "matching every other figure/table in this project. (Corrected 2026-07-20: an earlier "
        "version of this script used a strict per_slice_sla_margin>0 check, which "
        "undercounts -- margins in this environment sit almost exactly at the 0.0 "
        "boundary most of the time.)",
        "",
        "| Policy | Slice | Frac compliant | Block rate | n samples |",
        "|---|---|---|---|---|",
    ]
    for name, compliant, blocks, reqs, reward in rows:
        for s in SLICE_ORDER:
            arr = np.array(compliant[s])
            frac = float(np.mean(arr)) if arr.size else float("nan")
            block_rate = blocks[s] / max(1, reqs[s])
            lines.append(f"| {name} | {s} | {frac:.3f} | {block_rate:.3f} | {arr.size} |")
    lines.append("")
    lines.append("| Policy | Mean per-step reward |")
    lines.append("|---|---|")
    for name, _, _, _, reward in rows:
        lines.append(f"| {name} | {np.mean(reward):.4f} |")

    lines.append("")
    lines.append("## Validity verdict")
    all_compliant = {
        name: {s: float(np.mean(compliant[s])) for s in SLICE_ORDER}
        for name, compliant, _, _, _ in rows
    }
    any_saturated_low = all(
        all(all_compliant[name][s] < 0.02 for name in all_compliant) for s in SLICE_ORDER
    )
    any_saturated_high = all(
        all(all_compliant[name][s] > 0.98 for name in all_compliant) for s in SLICE_ORDER
    )
    if any_saturated_low or any_saturated_high:
        lines.append("**FAIL** -- all policies saturated at the same extreme on every slice; "
                      "no differentiation. Design needs another iteration.")
    else:
        lines.append("**PASS** -- policies show real, non-saturated, per-slice differentiation "
                      "in SLA compliance (see table above).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[run_admission_efficiency_baselines] wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
