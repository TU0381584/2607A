#!/usr/bin/env python3
"""A2's "give the baseline its best shot" requirement: sweep LbOnlyHeuristic's
two parameters (utilization_threshold, capacity_margin) on a HELD-OUT seed
(950 -- distinct from the 256/257/258 used for the baseline validity check
and reserved for the eventual learned-arm evaluation split), pick the best
by mean reward, then report that winner's performance on 256/257/258 for
the honest, non-overfit final number.

Zero training -- LbOnlyHeuristic is a fixed heuristic (qoe_oran_framework/
comparators/lb_only_baseline.py), not modified. Does not touch frozen code.

Usage:
    python3 experiments/scripts/tune_static_threshold.py \
        --tune-seed 950 --eval-seeds 256 257 258 --episodes 10 \
        --out experiments/results/admission_efficiency/static_threshold_tuning.md
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

UTIL_THRESHOLDS = [0.7, 0.8, 0.9, 0.97]
CAPACITY_MARGINS = [0.7, 0.85, 1.0, 1.15]


def run_heuristic(heuristic, seed, n_episodes):
    env = make_env(seed=seed, reward_mode="qoe")
    per_slice_margin = {s: [] for s in SLICE_ORDER}
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        for _ in range(env.cfg.episode.steps_per_episode):
            pending = env.pending_requests()
            actions = heuristic.decide(pending, env.last_cluster_state)
            result = env.step(actions)
            rewards.append(result.reward)
            rb = result.info.get("reward_breakdown", {})
            for s, m in rb.get("per_slice_sla_margin", {}).items():
                per_slice_margin[s].append(m)
    return per_slice_margin, rewards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune-seed", type=int, default=950)
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="experiments/results/admission_efficiency/static_threshold_tuning.md")
    args = ap.parse_args()

    env0 = make_env(seed=args.tune_seed, reward_mode="qoe")
    cfg = env0.cfg

    sweep_rows = []
    best = None
    for ut in UTIL_THRESHOLDS:
        for cm in CAPACITY_MARGINS:
            heuristic = LbOnlyHeuristic(cfg, utilization_threshold=ut, capacity_margin=cm)
            margins, rewards = run_heuristic(heuristic, args.tune_seed, args.episodes)
            mean_reward = float(np.mean(rewards))
            mean_compliance = float(np.mean([
                np.mean(np.array(margins[s]) > 0) for s in SLICE_ORDER if margins[s]
            ]))
            sweep_rows.append((ut, cm, mean_reward, mean_compliance))
            if best is None or mean_reward > best[2]:
                best = (ut, cm, mean_reward, mean_compliance)

    best_ut, best_cm, best_tune_reward, best_tune_compliance = best

    # Honest final numbers: re-evaluate the WINNING params on the eval seeds
    # (disjoint from the tuning seed) -- not the seed it was picked on.
    final_heuristic = LbOnlyHeuristic(cfg, utilization_threshold=best_ut, capacity_margin=best_cm)
    agg_margin = {s: [] for s in SLICE_ORDER}
    agg_reward = []
    for seed in args.eval_seeds:
        margins, rewards = run_heuristic(final_heuristic, seed, args.episodes)
        for s in SLICE_ORDER:
            agg_margin[s].extend(margins[s])
        agg_reward.extend(rewards)

    lines = [
        "# Static-threshold (LbOnlyHeuristic) honest tuning sweep",
        "",
        f"Tuned on held-out seed {args.tune_seed} (disjoint from eval seeds "
        f"{args.eval_seeds}); {args.episodes} episodes/seed, {len(UTIL_THRESHOLDS)}x"
        f"{len(CAPACITY_MARGINS)}={len(UTIL_THRESHOLDS)*len(CAPACITY_MARGINS)} grid points.",
        "",
        "## Sweep (on tuning seed only)",
        "",
        "| utilization_threshold | capacity_margin | mean reward | mean compliance |",
        "|---|---|---|---|",
    ]
    for ut, cm, r, c in sorted(sweep_rows, key=lambda x: -x[2]):
        marker = " **<- winner**" if (ut, cm) == (best_ut, best_cm) else ""
        lines.append(f"| {ut} | {cm} | {r:.4f} | {c:.3f} |{marker}")

    lines.append("")
    lines.append(f"## Winner: utilization_threshold={best_ut}, capacity_margin={best_cm}")
    lines.append("")
    lines.append(f"Honest, held-out performance on eval seeds {args.eval_seeds} "
                  f"(NOT the seed used to pick these parameters):")
    lines.append("")
    lines.append("| Slice | Mean margin | Frac compliant |")
    lines.append("|---|---|---|")
    for s in SLICE_ORDER:
        arr = np.array(agg_margin[s])
        lines.append(f"| {s} | {arr.mean():.3f} | {float(np.mean(arr > 0)):.3f} |")
    lines.append("")
    lines.append(f"Mean reward on eval seeds: {np.mean(agg_reward):.4f}")
    lines.append("")
    lines.append(f"(For comparison, default params utilization_threshold=0.97, "
                  f"capacity_margin=1.0 scored {[r for ut,cm,r,c in sweep_rows if ut==0.97 and cm==1.0][0]:.4f} "
                  f"mean reward on the TUNING seed -- the honest sweep is not just picking defaults.)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[tune_static_threshold] wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
