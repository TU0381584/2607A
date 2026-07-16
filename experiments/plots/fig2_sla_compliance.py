#!/usr/bin/env python3
"""Figure 2: per-slice SLA compliance (%) by arm -- grouped bars with error
bars (mean +/- std across seeds), URLLC/eMBB/mMTC, reading only from live
evaluation omega logs (experiments/results/live/<arm>/<reward_mode>/rep_seed<N>/omega_log.jsonl,
see run_live_eval_arm.py). Uses each rep's LAST episode rollup as that
rep's summary compliance (matches RunSummary.sla_compliance_by_slice's own
"mean across episodes" semantics -- we instead take the mean across
episode-rollups directly here since a live-eval rep's episodes may span
several health-checked batches with independently-seeded RNG streams; see
CAMPAIGN_LOG.md).

Usage:
    python3 experiments/plots/fig2_sla_compliance.py \
        --live-root experiments/results/live --seeds 256 257 258 \
        --out experiments/plots/out/fig2_sla_compliance
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, ARMS, SLICE_ORDER, SLICE_STYLE, arm_run_dir, read_omega_log  # noqa: E402

ARM_REWARD_MODE = {
    "baseline": "sla", "dqn_sla": "sla", "a2c_sla": "sla",
    "dqn_qoe": "qoe", "a2c_qoe": "qoe",
}


def per_rep_mean_compliance(omega_path: Path) -> dict:
    """Mean, across all episode-rollup rows in this rep's log, of each
    slice's per-episode SLA compliance fraction."""
    per_slice_vals = {s: [] for s in SLICE_ORDER}
    for row in read_omega_log(omega_path):
        if row.step != -1:
            continue
        by_slice = row.evidence.get("episode_sla_compliance_by_slice")
        if not by_slice:
            continue
        for slice_id in SLICE_ORDER:
            if slice_id in by_slice:
                per_slice_vals[slice_id].append(by_slice[slice_id])
    return {s: (float(np.mean(v)) if v else float("nan")) for s, v in per_slice_vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live")
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--out", default="experiments/plots/out/fig2_sla_compliance")
    args = ap.parse_args()

    arm_slice_means: dict = {}
    arm_slice_stds: dict = {}
    n_seeds_used = {}

    for arm in ARMS:
        mode = ARM_REWARD_MODE[arm]
        per_slice_across_seeds = {s: [] for s in SLICE_ORDER}
        for seed in args.seeds:
            omega_path = arm_run_dir(args.live_root, arm, mode, seed) / "omega_log.jsonl"
            if not omega_path.exists():
                continue
            rep_means = per_rep_mean_compliance(omega_path)
            for s in SLICE_ORDER:
                if not np.isnan(rep_means[s]):
                    per_slice_across_seeds[s].append(rep_means[s])
        n_seeds_used[arm] = max((len(v) for v in per_slice_across_seeds.values()), default=0)
        arm_slice_means[arm] = {s: (float(np.mean(v)) * 100 if v else float("nan")) for s, v in per_slice_across_seeds.items()}
        arm_slice_stds[arm] = {s: (float(np.std(v)) * 100 if v else 0.0) for s, v in per_slice_across_seeds.items()}

    fig, ax = plt.subplots()
    n_arms = len(ARMS)
    n_slices = len(SLICE_ORDER)
    bar_width = 0.8 / n_slices
    x = np.arange(n_arms)

    for i, slice_id in enumerate(SLICE_ORDER):
        style = SLICE_STYLE[slice_id]
        means = [arm_slice_means[arm][slice_id] for arm in ARMS]
        stds = [arm_slice_stds[arm][slice_id] for arm in ARMS]
        offset = (i - (n_slices - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=2,
               color=style["color"], hatch=style["hatch"], label=style["label"],
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_STYLE[a]["label"] for a in ARMS], rotation=30, ha="right")
    ax.set_ylabel("SLA compliance (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-slice SLA compliance by arm")
    ax.legend(loc="lower right", frameon=False, ncol=3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig2] wrote {out_path}.pdf / .png -- n_seeds per arm: {n_seeds_used}")


if __name__ == "__main__":
    main()
