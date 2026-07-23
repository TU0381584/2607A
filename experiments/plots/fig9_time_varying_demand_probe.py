#!/usr/bin/env python3
"""Figure 9: SLA compliance per arm/slice under the offline-only,
preliminary within-episode demand-phase probe (see
experiments/scripts/probe_time_varying_demand_offline.py, which produces
this figure's input JSON -- no live rig time used for this figure).

Usage:
    python3 experiments/plots/fig9_time_varying_demand_probe.py \
        --results-json experiments/results/time_varying_demand_probe/results.json \
        --out experiments/plots/out/fig9_time_varying_demand_probe
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, SLICE_ORDER  # noqa: E402

ARMS = ["dqn_sla", "a2c_sla", "dqn_qoe", "a2c_qoe"]
PHASES = ["low", "high", "medium"]  # chronological order within episode
PHASE_TICK_LABELS = ["low\n(0.7x)", "high\n(1.3x)", "medium\n(1.0x)"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-json", default="experiments/results/time_varying_demand_probe/results.json")
    ap.add_argument("--out", default="experiments/plots/out/fig9_time_varying_demand_probe")
    args = ap.parse_args()

    with open(args.results_json) as fh:
        data = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    x = range(len(PHASES))
    width = 0.2
    for ax, s in zip(axes, SLICE_ORDER):
        for i, arm in enumerate(ARMS):
            vals = [data[arm][p][s] for p in PHASES]
            ax.bar([xi + (i - 1.5) * width for xi in x], vals, width=width,
                   color=ARM_STYLE[arm]["color"], label=ARM_STYLE[arm]["label"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(PHASE_TICK_LABELS, fontsize=6.5)
        ax.set_title(s, fontsize=9)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("SLA compliance (%)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=6)
    fig.suptitle(
        "Offline preliminary probe: SLA compliance under within-episode\n"
        "demand phases (constant-demand-trained frozen checkpoints)", fontsize=8.5,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig9] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
