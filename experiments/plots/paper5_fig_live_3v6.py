#!/usr/bin/env python3
"""Live single-gNB load comparison: same seed-900 single-agent-DQN
checkpoint, 3 UEs vs 6 UEs, both against the real OAI rig with real
per-slice UDP traffic, both 2 episodes long. Reads each run's own rollup
records directly from its raw omega_log.jsonl (evidence.rollup == true,
one per episode) -- not retyped. Only 2 episodes per condition, so no
bootstrap CI (meaningless at n=2); individual episode values are shown
as points behind the mean.

Two panels only (a compact 3-vs-6 story, not a per-slice-times-per-metric
grid):
(a) SLA margin, percent change from 3 UEs to 6 UEs, one bar per slice --
    this says "how much worse does load doubling get" in one panel
    instead of three separate same-shaped absolute-scale panels.
(b) Mean reward per step, 3 UEs vs 6 UEs.

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
from paper5_common import IEEE_COLUMN_WIDTH_IN, STATUS_COLORS  # noqa: E402,F401

SLICES = ["urllc", "embb", "mmtc"]
COND_STYLE = {
    "3 UEs": {"color": "#2a78d6", "marker": "o"},
    "6 UEs": {"color": "#eb6834", "marker": "^"},
}


def load_episode_rollups(path: str):
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_COLUMN_WIDTH_IN * 1.7, IEEE_COLUMN_WIDTH_IN * 0.9))

    # ---- (a) SLA margin, percent change 3 UEs -> 6 UEs, one bar per slice ----
    mean3 = {s: np.mean([e["episode_sla_margin_by_slice"][s] for e in runs["3 UEs"]]) for s in SLICES}
    mean6 = {s: np.mean([e["episode_sla_margin_by_slice"][s] for e in runs["6 UEs"]]) for s in SLICES}
    # margins are negative; "worse" means more negative, so pct change is
    # computed on |margin| so a positive bar always reads as "worse at 6 UEs".
    pct_change = [100.0 * (abs(mean6[s]) - abs(mean3[s])) / abs(mean3[s]) for s in SLICES]
    x = np.arange(len(SLICES))
    colors = [STATUS_COLORS["good"] if v <= 0 else STATUS_COLORS["critical"] for v in pct_change]
    ax1.bar(x, pct_change, width=0.5, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for xi, v in zip(x, pct_change):
        ax1.annotate(f"{v:+.0f}%", (xi, v), xytext=(0, 3 if v >= 0 else -3),
                     textcoords="offset points", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=7)
    ax1.axhline(0, color="#898781", linewidth=0.8, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SLICES)
    ax1.set_ylabel("SLA margin change,\n3 UEs $\\to$ 6 UEs (%)")
    ax1.set_title("(a)", loc="left")

    # ---- (b) mean reward per step, 3 UEs vs 6 UEs ----
    labels = list(runs.keys())
    reward_vals = [[e["episode_mean_reward"] for e in runs[label]] for label in labels]
    means_r = [np.mean(v) for v in reward_vals]
    xb = np.arange(len(labels))
    for i, label in enumerate(labels):
        style = COND_STYLE[label]
        ax2.bar(xb[i], means_r[i], width=0.5, color=style["color"], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.scatter([xb[i]] * len(reward_vals[i]), reward_vals[i], color="#0b0b0b", s=16, zorder=5, marker=style["marker"])
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean reward per step")
    ax2.set_title("(b)", loc="left")

    fig.subplots_adjust(wspace=0.5, bottom=0.18)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[live-3v6] wrote {out_path}.pdf / .png")
    for s, v in zip(SLICES, pct_change):
        print(f"  {s}: 3UE mean={mean3[s]:.1f}  6UE mean={mean6[s]:.1f}  change={v:+.1f}%")


if __name__ == "__main__":
    main()
