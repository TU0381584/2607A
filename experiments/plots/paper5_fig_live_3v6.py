#!/usr/bin/env python3
"""Live single-gNB comparison: 3 UEs vs 6 UEs, same seed-900 single-agent-DQN
checkpoint, same 2-episode length, both runs against the real OAI rig with
real per-slice UDP traffic. Reads each run's own rollup records directly
from its raw omega_log.jsonl (evidence.rollup == true, one per episode) --
not retyped from the console summary. Only 2 episodes per condition, so no
bootstrap CI is computed (would be statistically meaningless at n=2);
individual episode values are shown as points behind the mean instead.

(a) SLA margin by slice (symlog y-axis: the three slices span more than
    three orders of magnitude, and margins are negative throughout, so a
    signed log scale is used rather than a misleading linear one).
(b) Mean reward per step.
(c) Mean cost (QoE-mapper diagnostic, not the actual reward signal --
    see the omega log's own per-step "limitation" field).

Usage:
    python3 experiments/plots/paper5_fig_live_3v6.py \
        --run3-jsonl experiments/results/live/m31_redo/3ue_omega_log.jsonl \
        --run6-jsonl experiments/results/live/m31_6ue/6ue_omega_log.jsonl \
        --out paper5_wpc/figures/fig_live_3v6
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
COND_STYLE = {
    "3 UEs": {"color": "#2a78d6", "marker": "o"},
    "6 UEs": {"color": "#eb6834", "marker": "^"},
}


def load_episode_rollups(path: str):
    """Returns a list of per-episode rollup dicts (evidence.rollup == true),
    ordered by episode number, read directly from the raw omega log."""
    rollups = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", {})
            if ev.get("rollup"):
                rollups.append(ev)
    return rollups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run3-jsonl", default="experiments/results/live/m31_redo/3ue_omega_log.jsonl")
    ap.add_argument("--run6-jsonl", default="experiments/results/live/m31_6ue/6ue_omega_log.jsonl")
    ap.add_argument("--out", default="paper5_wpc/figures/fig_live_3v6")
    args = ap.parse_args()

    runs = {
        "3 UEs": load_episode_rollups(args.run3_jsonl),
        "6 UEs": load_episode_rollups(args.run6_jsonl),
    }
    for label, eps in runs.items():
        print(f"[live-3v6] {label}: {len(eps)} episode rollups loaded from raw log")
        for i, e in enumerate(eps):
            print(f"  ep{i+1}: reward={e['episode_mean_reward']:.3f} "
                  f"margin={e['episode_sla_margin_by_slice']}")

    fig, axes = plt.subplots(2, 3, figsize=(IEEE_COLUMN_WIDTH_IN * 2.35, IEEE_COLUMN_WIDTH_IN * 1.9))
    (ax_urllc, ax_embb, ax_mmtc), (ax_reward, ax_cost, ax_legend) = axes

    labels = list(runs.keys())
    xb = np.arange(len(labels))

    def bar_panel(ax, values_by_label, ylabel, tag):
        means = [np.mean(v) for v in values_by_label]
        for i, label in enumerate(labels):
            style = COND_STYLE[label]
            ax.bar(xb[i], means[i], width=0.5, color=style["color"], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
            ax.scatter([xb[i]] * len(values_by_label[i]), values_by_label[i],
                       color="#0b0b0b", s=16, zorder=5, marker=style["marker"])
        ax.set_xticks(xb)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(tag, loc="left")
        ax.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=0)

    # ---- (a)-(c) SLA margin, one linear-axis small multiple per slice ----
    for ax, slice_name, tag in [(ax_urllc, "urllc", "(a)"), (ax_embb, "embb", "(b)"), (ax_mmtc, "mmtc", "(c)")]:
        vals = [[e["episode_sla_margin_by_slice"][slice_name] for e in runs[label]] for label in labels]
        bar_panel(ax, vals, f"SLA margin -- {slice_name}", tag)

    # ---- (d) mean reward per step ----
    reward_vals = [[e["episode_mean_reward"] for e in runs[label]] for label in labels]
    bar_panel(ax_reward, reward_vals, "Mean reward per step", "(d)")

    # ---- (e) mean cost (QoE-mapper diagnostic) ----
    cost_vals = []
    for label in labels:
        path = args.run3_jsonl if label == "3 UEs" else args.run6_jsonl
        by_ep = {1: [], 2: []}
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                ev = rec.get("evidence", {})
                if ev.get("rollup"):
                    continue
                ep = rec.get("episode")
                if ep in by_ep and "cost" in ev:
                    by_ep[ep].append(ev["cost"])
        cost_vals.append([np.mean(by_ep[1]), np.mean(by_ep[2])])
    bar_panel(ax_cost, cost_vals, "Mean cost (QoE diagnostic)", "(e)")

    # ---- shared legend in the unused 6th cell ----
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax_legend.axis("off")
    handles = []
    for label in labels:
        style = COND_STYLE[label]
        handles.append(Patch(facecolor=style["color"], alpha=0.85, label=f"{label} (mean)"))
        handles.append(Line2D([0], [0], marker=style["marker"], color="#0b0b0b", linestyle="none",
                               markersize=6, label=f"{label} (per episode, n=2)"))
    ax_legend.legend(handles=handles, loc="center", frameon=False, fontsize=8)

    fig.subplots_adjust(wspace=0.6, hspace=0.4, bottom=0.08)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[live-3v6] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
