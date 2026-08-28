#!/usr/bin/env python3
"""M1, Step 3 (figure): offline-vs-live compliance scatter, one point per
checkpoint, baseline (bc=2000, original frozen defaults) vs. recalibrated
env, side by side. Reuses experiments/plots/common.py's ARM_STYLE (all six
points are dqn_sla checkpoints, so styled with ARM_STYLE["dqn_sla"]'s
color/marker for consistency with every other dqn_sla series in the paper;
individual checkpoints are distinguished by their training-seed label since
ARM_STYLE has no per-checkpoint entries -- there was never a "per checkpoint"
concept before M1).

Usage:
    python3 experiments/plots/../scripts/m1_plot_correlation.py \
        --baseline experiments/results/m1_recalibration/held_out_baseline/compliance_baseline.json \
        --recalibrated experiments/results/m1_recalibration/held_out_recalibrated/compliance_recalibrated.json \
        --live-traces experiments/results/m1_recalibration/live_traces.json \
        --out CACS26/figures/m1_offline_live_correlation
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/plots")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from common import ARM_STYLE  # noqa: E402

SEEDS = [256, 257, 258, 259, 260, 261]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--recalibrated", required=True)
    ap.add_argument("--live-traces", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.baseline) as fh:
        base = json.load(fh)["compliance"]
    with open(args.recalibrated) as fh:
        recal = json.load(fh)["compliance"]
    with open(args.live_traces) as fh:
        live = json.load(fh)["per_checkpoint"]

    live_pct = [live[str(s)]["live_compliance_pct"] for s in SEEDS]
    base_pct = [base[str(s)]["pct"] for s in SEEDS]
    recal_pct = [recal[str(s)]["pct"] for s in SEEDS]

    rho_base, p_base = spearmanr(live_pct, base_pct)
    rho_recal, p_recal = spearmanr(live_pct, recal_pct)

    style = ARM_STYLE["dqn_sla"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    panels = [
        (axes[0], base_pct, rho_base, p_base, "Original env (bc=2000)"),
        (axes[1], recal_pct, rho_recal, p_recal, "Recalibrated env"),
    ]
    for ax, offline_pct, rho, p, label in panels:
        ax.scatter(live_pct, offline_pct, color=style["color"], marker=style["marker"],
                   s=40, zorder=3, label="dqn_sla checkpoint")
        for x, y, seed in zip(live_pct, offline_pct, SEEDS):
            ax.annotate(str(seed), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
        ax.set_xlabel("Live compliance (%)")
        ax.set_title(f"{label}\nρ={rho:.2f}, p={p:.2f}", fontsize=8.5)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.plot([-5, 105], [-5, 105], color="gray", linewidth=0.6, linestyle=":", zorder=1)
    axes[0].set_ylabel("Held-out offline compliance (%)")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[m1-plot] wrote {out_path}.pdf/.png")
    print(f"[m1-plot] Spearman rho: baseline={rho_base:.3f} (p={p_base:.3f}), "
          f"recalibrated={rho_recal:.3f} (p={p_recal:.3f})")
    print(f"[m1-plot] live: {dict(zip(SEEDS, live_pct))}")
    print(f"[m1-plot] baseline offline: {dict(zip(SEEDS, base_pct))}")
    print(f"[m1-plot] recalibrated offline: {dict(zip(SEEDS, recal_pct))}")


if __name__ == "__main__":
    main()
