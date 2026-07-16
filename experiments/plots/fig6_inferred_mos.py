#!/usr/bin/env python3
"""Figure 6: inferred MOS per slice by arm -- grouped bars, mean +/- std
across seeds. Passive QoE diagnostics (mos_by_slice) are logged for EVERY
arm regardless of reward_mode (see env.py's cfg.qoe-is-not-None branch),
so this figure spans all 5 arms on equal footing -- the qoe-vs-sla-reward
comparison the campaign handover asks for.

Usage:
    python3 experiments/plots/fig6_inferred_mos.py \
        --live-root experiments/results/live --seeds 256 257 258 \
        --out experiments/plots/out/fig6_inferred_mos
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


def rep_mean_mos(omega_path: Path) -> dict:
    vals = {s: [] for s in SLICE_ORDER}
    for row in read_omega_log(omega_path):
        if row.step < 1:
            continue
        mos_by_slice = row.evidence.get("mos_by_slice")
        if not mos_by_slice:
            continue
        for s in SLICE_ORDER:
            if s in mos_by_slice:
                vals[s].append(mos_by_slice[s])
    return {s: (float(np.mean(v)) if v else float("nan")) for s, v in vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live")
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--out", default="experiments/plots/out/fig6_inferred_mos")
    args = ap.parse_args()

    arm_means, arm_stds = {}, {}
    for arm in ARMS:
        mode = ARM_REWARD_MODE[arm]
        per_slice = {s: [] for s in SLICE_ORDER}
        for seed in args.seeds:
            omega_path = arm_run_dir(args.live_root, arm, mode, seed) / "omega_log.jsonl"
            if not omega_path.exists():
                continue
            rep_vals = rep_mean_mos(omega_path)
            for s in SLICE_ORDER:
                if not np.isnan(rep_vals[s]):
                    per_slice[s].append(rep_vals[s])
        arm_means[arm] = {s: (float(np.mean(v)) if v else float("nan")) for s, v in per_slice.items()}
        arm_stds[arm] = {s: (float(np.std(v)) if v else 0.0) for s, v in per_slice.items()}

    fig, ax = plt.subplots()
    n_slices = len(SLICE_ORDER)
    bar_width = 0.8 / n_slices
    x = np.arange(len(ARMS))

    for i, slice_id in enumerate(SLICE_ORDER):
        style = SLICE_STYLE[slice_id]
        means = [arm_means[arm][slice_id] for arm in ARMS]
        stds = [arm_stds[arm][slice_id] for arm in ARMS]
        offset = (i - (n_slices - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=2,
               color=style["color"], hatch=style["hatch"], label=style["label"],
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_STYLE[a]["label"] for a in ARMS], rotation=30, ha="right")
    ax.set_ylabel("Inferred MOS (1-5)")
    ax.set_ylim(1, 5)
    ax.set_title("Inferred per-slice MOS by arm")
    ax.legend(loc="lower right", frameon=False, ncol=3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig6] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
