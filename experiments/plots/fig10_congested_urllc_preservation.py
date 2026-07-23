#!/usr/bin/env python3
"""Figure 10: DRL vs. baseline SLA compliance under congested, dynamic,
randomized multi-slice traffic, URLLC highlighted (the slice the reward
is designed to protect first -- priority_weight=5.0, violation_penalty=8.0,
both highest of the 3 slices). Reads
experiments/results/congested_vs_baseline/results.json (produced by
experiments/scripts/eval_congested_vs_baseline.py).

Usage:
    python3 experiments/plots/fig10_congested_urllc_preservation.py \
        --results-json experiments/results/congested_vs_baseline/results.json \
        --out experiments/plots/out/fig10_congested_urllc_preservation
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE  # noqa: E402

ARMS = ["baseline", "dqn_sla", "a2c_sla", "dqn_qoe", "a2c_qoe"]
SLICE_ORDER = ["urllc", "embb", "mmtc"]
SLICE_LABEL = {"urllc": "URLLC (priority-protected)", "embb": "eMBB", "mmtc": "mMTC"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-json", default="experiments/results/congested_vs_baseline/results.json")
    ap.add_argument("--out", default="experiments/plots/out/fig10_congested_urllc_preservation")
    args = ap.parse_args()

    with open(args.results_json) as fh:
        data = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharey=True)
    for ax, s in zip(axes, SLICE_ORDER):
        vals = [data[arm]["compliance_pct"][s] for arm in ARMS]
        colors = [ARM_STYLE[arm]["color"] for arm in ARMS]
        bars = ax.bar(range(len(ARMS)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([ARM_STYLE[a]["label"] for a in ARMS], rotation=35, ha="right", fontsize=6.5)
        ax.set_title(SLICE_LABEL[s], fontsize=8.5, fontweight=("bold" if s == "urllc" else "normal"))
        ax.set_ylim(0, 100)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=6)
    axes[0].set_ylabel("SLA compliance (%)")
    fig.suptitle(
        "SLA compliance under congested/dynamic/randomized traffic\n"
        "(held-out episodes, frozen weights) -- baseline vs. DRL", fontsize=9,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig10] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
