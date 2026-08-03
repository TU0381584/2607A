#!/usr/bin/env python3
"""Companion to fig4_ceiling_trajectories.py (baseline vs. best learned arm):
this one plots DQN-SLA vs. DQN-QoE directly, since the paper's central
live comparison is between those two reward variants, not baseline vs.
best. Same per-slice commanded-ceiling trace, same omega-log source, real
checkpoints only -- no synthesised data.

Defaults to seed 955, episode 1, batch0 for both arms: the one seed
where DQN-SLA collapses (embb=0.20, urllc=0.00, mmtc=0.067 compliant
step-fraction, i.e. URLLC never meets its SLA margin all episode) while
DQN-QoE, evaluated on the exact same seed/batch/episode, stays fully
compliant (1.0/1.0/1.0) -- found by scanning every episode-rollup row in
both arms' seed-955 omega logs for `episode_sla_compliance_by_slice`
(experiments/results/live_campaign_v2/{dqn_sla,dqn_qoe}/*/rep_seed955/omega_log.jsonl).
run_id is passed explicitly for both arms (not left to
load_episode_ceilings' first-encountered-run_id default) because episode
numbers reset per batch (fig4's own documented caveat) and seed 955 has
3 batches each.

Usage:
    python3 experiments/plots/fig_sla_vs_qoe_ceiling.py \
        --live-root experiments/results/live_campaign_v2 --seed 955 \
        --episode 1 \
        --sla-run-id dqn_sla_sla_seed955_batch0 \
        --qoe-run-id dqn_qoe_qoe_seed955_batch0 \
        --out experiments/plots/out/fig_sla_vs_qoe_ceiling
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, SLICE_ORDER, SLICE_STYLE, arm_run_dir, read_omega_log  # noqa: E402
from fig4_ceiling_trajectories import load_episode_ceilings, SLICE_CAP  # noqa: E402

ARM_REWARD_MODE = {"dqn_sla": "sla", "dqn_qoe": "qoe"}
GNB_ID = "gnb-0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", default="experiments/results/live_campaign_v2")
    ap.add_argument("--seed", type=int, default=955)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--sla-run-id", default="dqn_sla_sla_seed955_batch0")
    ap.add_argument("--qoe-run-id", default="dqn_qoe_qoe_seed955_batch0")
    ap.add_argument("--out", default="experiments/plots/out/fig_sla_vs_qoe_ceiling")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(SLICE_ORDER), 1, sharex=True,
                              figsize=(3.5, 0.75 * 3.5 * len(SLICE_ORDER) / 2))

    run_id_by_arm = {"dqn_sla": args.sla_run_id, "dqn_qoe": args.qoe_run_id}
    for arm in ["dqn_sla", "dqn_qoe"]:
        mode = ARM_REWARD_MODE[arm]
        omega_path = arm_run_dir(args.live_root, arm, mode, args.seed) / "omega_log.jsonl"
        if not omega_path.exists():
            print(f"[fig_sla_vs_qoe] WARNING: missing {omega_path}", file=sys.stderr)
            continue
        series = load_episode_ceilings(omega_path, args.episode, run_id=run_id_by_arm[arm])
        style = ARM_STYLE[arm]
        for ax, slice_id in zip(axes, SLICE_ORDER):
            steps, ratios = series[slice_id]
            if steps:
                ax.step(steps, ratios, where="post", color=style["color"],
                        linestyle=style["linestyle"], label=style["label"], linewidth=1.1)

    for ax, slice_id in zip(axes, SLICE_ORDER):
        ax.set_ylabel(f"{SLICE_STYLE[slice_id]['label']}\nmax_ratio")
        cap = SLICE_CAP[slice_id]
        ax.axhline(cap, color="black", linewidth=0.7, linestyle=(0, (1, 1)), alpha=0.6)
        ax.annotate(f"cap={cap}", xy=(1.0, cap), xycoords=("axes fraction", "data"),
                    xytext=(2, 2), textcoords="offset points", fontsize=6, ha="left", va="bottom")
        ax.set_ylim(top=cap * 1.25)
    axes[-1].set_xlabel("Step (within episode)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Commanded PRB ceiling: DQN-SLA vs.\\ DQN-QoE, seed 955 ep.\\ 1", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig_sla_vs_qoe] wrote {out_path}.pdf / .png "
          f"(dqn_sla run_id={args.sla_run_id} vs dqn_qoe run_id={args.qoe_run_id}, "
          f"seed {args.seed}, episode {args.episode})")


if __name__ == "__main__":
    main()
