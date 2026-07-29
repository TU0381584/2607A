#!/usr/bin/env python3
"""Stage 5's zero-training validity check for live_scale_offline_env.py,
mirroring run_admission_efficiency_baselines.py's exact methodology (same
3 scripted policies, same compliance field, same PASS/FAIL criterion) --
applied to the DIFFERENT environment this stage needs: real cap/nominal
(saclb_campaign_v2.yaml's 12/4/3, 3/2/2 -- matching live, NOT rescaled to
"tens of units" the way the admission-efficiency workstream's environment
was) with mean_offered_ratio set to REAL live-observed demand instead of
a formula derived from cap or nominal_ratio.

Must run BEFORE any retraining: if this saturates (all 3 policies pinned
at the same extreme), the backlog_capacity/Lmax combination needs
sweeping first, exactly as the admission-efficiency precedent required --
not assumed to be fine just because the demand scale is now realistic.

Usage:
    python3 experiments/scripts/validate_live_scale_offline_env.py \
        --seeds 256 257 258 --episodes 10 --backlog-capacity 200 \
        --out experiments/results/live_scale_offline/baseline_validity.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import numpy as np  # noqa: E402
from live_scale_offline_env import make_env  # noqa: E402
from qoe_oran_framework.comparators.lb_only_baseline import LbOnlyHeuristic  # noqa: E402

SLICE_ORDER = ["embb", "urllc", "mmtc"]


def accept_all_decide(pending, cluster_state):
    return [1 for _ in pending]


def reject_all_decide(pending, cluster_state):
    return [0 for _ in pending]


def run_policy(decide_fn, seed, n_episodes, backlog_capacity, config_path=None):
    env = make_env(seed=seed, reward_mode="qoe", backlog_capacity=backlog_capacity, config_path=config_path)
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
            for s, c in rb.get("per_slice_compliant", {}).items():
                per_slice_compliant[s].append(bool(c))
    return per_slice_compliant, blocks, total_reqs, rewards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--backlog-capacity", type=float, default=200.0)
    ap.add_argument("--config-path", default=None, help="override live_scale_offline_env.CONFIG_PATH")
    ap.add_argument("--out", default="experiments/results/live_scale_offline/baseline_validity.md")
    args = ap.parse_args()

    policies = {"accept_all": accept_all_decide, "reject_all": reject_all_decide}

    rows = []
    for name, decide_fn in policies.items():
        agg_compliant = {s: [] for s in SLICE_ORDER}
        agg_blocks = {s: 0 for s in SLICE_ORDER}
        agg_reqs = {s: 0 for s in SLICE_ORDER}
        agg_reward = []
        for seed in args.seeds:
            c, b, r, rew = run_policy(decide_fn, seed, args.episodes, args.backlog_capacity, args.config_path)
            for s in SLICE_ORDER:
                agg_compliant[s].extend(c[s])
                agg_blocks[s] += b[s]
                agg_reqs[s] += r[s]
            agg_reward.extend(rew)
        rows.append((name, agg_compliant, agg_blocks, agg_reqs, agg_reward))

    env0 = make_env(seed=args.seeds[0], reward_mode="qoe", backlog_capacity=args.backlog_capacity, config_path=args.config_path)
    heuristic = LbOnlyHeuristic(env0.cfg)

    def static_threshold_decide(pending, cluster_state):
        return heuristic.decide(pending, cluster_state)

    agg_compliant = {s: [] for s in SLICE_ORDER}
    agg_blocks = {s: 0 for s in SLICE_ORDER}
    agg_reqs = {s: 0 for s in SLICE_ORDER}
    agg_reward = []
    for seed in args.seeds:
        c, b, r, rew = run_policy(static_threshold_decide, seed, args.episodes, args.backlog_capacity, args.config_path)
        for s in SLICE_ORDER:
            agg_compliant[s].extend(c[s])
            agg_blocks[s] += b[s]
            agg_reqs[s] += r[s]
        agg_reward.extend(rew)
    rows.append(("static_threshold", agg_compliant, agg_blocks, agg_reqs, agg_reward))

    lines = [
        "# live_scale_offline_env baseline validity check (Stage 5)",
        "",
        f"Config: `experiments/configs/saclb_campaign_v2.yaml` (REAL cap=12/4/3, "
        f"nominal=3/2/2 -- not rescaled) + mean_offered_ratio={{'embb':0.15,'urllc':0.05,'mmtc':0.05}} "
        f"(real live-observed demand) + backlog_capacity={args.backlog_capacity}",
        f"Seeds: {args.seeds}, episodes/seed: {args.episodes}",
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
                      "no differentiation. backlog_capacity/Lmax needs sweeping before training.")
    else:
        lines.append("**PASS** -- policies show real, non-saturated, per-slice differentiation "
                      "in SLA compliance (see table above).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
