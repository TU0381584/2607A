#!/usr/bin/env python3
"""M31: live 6-UE comparison, learned single-agent-DQN policy (seed-900
checkpoint) vs. the static lb_only baseline (fixed ceiling), 2 independent
runs per arm (seeds 900/901), both against the real OAI rig with real
per-slice UDP traffic. Reads every number directly from each run's raw
omega_log.jsonl (episode rollup records, evidence.rollup == true) -- not
retyped. Only 2 runs per arm, so no bootstrap CI (meaningless at n=2);
individual run values are shown as points behind the mean.

(a)-(c) SLA margin by slice (urllc/embb/mmtc), one linear-axis small
    multiple per slice -- same reasoning as the 3-vs-6-UE figure: the
    three slices span orders of magnitude, so a shared axis would hide
    the real differences.
(d) Mean reward per step.
(e) Block precision (fraction of blocks that correctly target mmtc,
    the only slice the reward calibration makes reject-optimal) --
    undefined for a run with zero total blocks, plotted as a gap, not a
    zero, since undefined is not the same claim as "zero precision."

Usage:
    python3 experiments/plots/paper5_fig_live_policy_vs_static.py \
        --dqn-jsonl experiments/results/live/m31_6ue/6ue_omega_log.jsonl \
                    experiments/results/live/m31_campaign/6ue_dqn_run2_omega_log.jsonl \
        --static-jsonl experiments/results/live/m31_campaign/6ue_static_run1_omega_log.jsonl \
                       experiments/results/live/m31_campaign/6ue_static_run2_omega_log.jsonl \
        --out Papers_4-5/Paper_5/WPC/figures/fig_live_policy_vs_static
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper5_common import IEEE_COLUMN_WIDTH_IN  # noqa: E402,F401

SLICES = ["urllc", "embb", "mmtc"]
ARM_STYLE = {
    "Learned policy (dqn)": {"color": "#2a78d6", "marker": "o"},
    "Static baseline (lb_only)": {"color": "#eb6834", "marker": "^"},
}


def load_run(path: str) -> dict:
    """One run's episode rollups + per-episode block-precision, all read
    directly from the raw omega log."""
    rollups = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", {})
            if ev.get("rollup"):
                rollups.append(ev)
    precisions = []
    for ev in rollups:
        by_slice = ev.get("episode_block_by_slice", {})
        total = sum(by_slice.values())
        if total > 0:
            precisions.append(by_slice.get("mmtc", 0) / total)
        else:
            precisions.append(None)  # undefined, not zero
    return {"rollups": rollups, "precisions": precisions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dqn-jsonl", nargs="+", required=True)
    ap.add_argument("--static-jsonl", nargs="+", required=True)
    ap.add_argument("--out", default="Papers_4-5/Paper_5/WPC/figures/fig_live_policy_vs_static")
    args = ap.parse_args()

    arms = {
        "Learned policy (dqn)": [load_run(p) for p in args.dqn_jsonl],
        "Static baseline (lb_only)": [load_run(p) for p in args.static_jsonl],
    }
    for label, runs in arms.items():
        print(f"[live-policy-vs-static] {label}: {len(runs)} runs")
        for i, run in enumerate(runs):
            for ep, (ev, prec) in enumerate(zip(run["rollups"], run["precisions"])):
                print(f"  run{i+1} ep{ep+1}: reward={ev['episode_mean_reward']:.3f} "
                      f"blocks={ev['episode_block_by_slice']} precision={prec}")

    fig, axes = plt.subplots(2, 3, figsize=(IEEE_COLUMN_WIDTH_IN * 2.35, IEEE_COLUMN_WIDTH_IN * 1.9))
    (ax_urllc, ax_embb, ax_mmtc), (ax_reward, ax_prec, ax_legend) = axes

    labels = list(arms.keys())
    xb = np.arange(len(labels))

    def bar_panel(ax, values_by_label, ylabel, tag, skip_none=False):
        means, plot_vals = [], []
        for vals in values_by_label:
            clean = [v for v in vals if v is not None] if skip_none else vals
            means.append(np.mean(clean) if clean else np.nan)
            plot_vals.append(clean)
        for i, label in enumerate(labels):
            style = ARM_STYLE[label]
            if not np.isnan(means[i]):
                ax.bar(xb[i], means[i], width=0.5, color=style["color"], alpha=0.85,
                       edgecolor="white", linewidth=0.5)
            ax.scatter([xb[i]] * len(plot_vals[i]), plot_vals[i],
                       color="#0b0b0b", s=16, zorder=5, marker=style["marker"])
        ax.set_xticks(xb)
        ax.set_xticklabels(["Learned\npolicy", "Static\nbaseline"])
        ax.set_ylabel(ylabel)
        ax.set_title(tag, loc="left")

    # ---- (a)-(c) SLA margin per slice, all episodes (n=4 per arm: 2 runs x 2 episodes) ----
    for ax, slice_name, tag in [(ax_urllc, "urllc", "(a)"), (ax_embb, "embb", "(b)"), (ax_mmtc, "mmtc", "(c)")]:
        vals = [
            [ev["episode_sla_margin_by_slice"][slice_name] for run in arms[label] for ev in run["rollups"]]
            for label in labels
        ]
        bar_panel(ax, vals, f"SLA margin -- {slice_name}", tag)
        ax.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=0)

    # ---- (d) mean reward per step, per episode ----
    reward_vals = [
        [ev["episode_mean_reward"] for run in arms[label] for ev in run["rollups"]]
        for label in labels
    ]
    bar_panel(ax_reward, reward_vals, "Mean reward per step", "(d)")

    # ---- (e) block precision, per episode (None = undefined, dropped from the mean/points) ----
    prec_vals = [
        [p for run in arms[label] for p in run["precisions"]]
        for label in labels
    ]
    n_undefined = [sum(1 for p in v if p is None) for v in prec_vals]
    bar_panel(ax_prec, prec_vals, "Block precision\n(mmtc-targeting fraction)", "(e)", skip_none=True)
    ax_prec.set_ylim(-0.05, 1.08)
    for i, n in enumerate(n_undefined):
        if n:
            ax_prec.annotate(f"{n} undefined\n(0 blocks)", (xb[i], -0.22), ha="center", va="top",
                              fontsize=6, color="#898781", annotation_clip=False)

    # ---- shared legend ----
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax_legend.axis("off")
    handles = []
    for label in labels:
        style = ARM_STYLE[label]
        handles.append(Patch(facecolor=style["color"], alpha=0.85, label=f"{label} (mean)"))
        handles.append(Line2D([0], [0], marker=style["marker"], color="#0b0b0b", linestyle="none",
                               markersize=6, label=f"{label} (per episode, n=4)"))
    ax_legend.legend(handles=handles, loc="center", frameon=False, fontsize=7.5)

    fig.subplots_adjust(wspace=0.6, hspace=0.5, bottom=0.1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[live-policy-vs-static] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
