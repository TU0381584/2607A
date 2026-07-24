#!/usr/bin/env python3
"""Figure 8: sim-to-real transfer parity -- offline-predicted vs.
live-measured SLA compliance, per (arm, slice), for the four learned arms
(the frozen-weights policies actually deployed live; the static baseline
has no offline-trained counterpart to compare against, so it is excluded
here by construction, not omission).

"Offline-predicted" = mean SLA compliance over each seed's FINAL 5 rollup
episodes (out of the 300-episode training run), pooled across the same 3
training seeds (256/257/258) -- i.e. the tail of training once the policy
has settled, evaluated with the same window size (5 episodes/seed) as the
live campaign itself, so the two numbers are computed the same way and
differ only in offline-synthetic-KPM vs. real-E2 measurement.

"Live-measured" = the same per-arm, per-slice SLA compliance already
reported in Table II / Fig. 2, from experiments/results/live_campaign.

Usage:
    python3 experiments/plots/fig8_sim2real_parity.py \
        --offline-root experiments/results/offline \
        --live-root experiments/results/live_campaign \
        --offline-seeds 256 257 258 --live-seeds 950 951 952 \
        --out experiments/plots/out/fig8_sim2real_parity
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, SLICE_ORDER, SLICE_STYLE, arm_run_dir, read_omega_log  # noqa: E402

LEARNED_ARMS = ["dqn_sla", "dqn_qoe"]
ARM_REWARD_MODE = {"dqn_sla": "sla", "a2c_sla": "sla", "dqn_qoe": "qoe", "a2c_qoe": "qoe"}
ARM_ALGO = {"dqn_sla": "dqn", "a2c_sla": "a2c", "dqn_qoe": "dqn", "a2c_qoe": "a2c"}
SLICE_MARKER = {"embb": "o", "urllc": "^", "mmtc": "s"}


def offline_tail_compliance(offline_root: Path, arm: str, seeds: list, tail: int = 5) -> dict:
    """Per-slice mean compliance fraction over each seed's final `tail`
    rollup episodes, pooled across seeds."""
    mode = ARM_REWARD_MODE[arm]
    algo = ARM_ALGO[arm]
    pooled = {s: [] for s in SLICE_ORDER}
    for seed in seeds:
        path = Path(offline_root) / mode / f"seed{seed}" / algo / "offline_closed_loop" / "rep_0" / "omega_log.jsonl"
        if not path.exists():
            print(f"[fig8] WARNING: missing {path}", file=sys.stderr)
            continue
        rollups = [row for row in read_omega_log(path) if row.step == -1]
        for row in rollups[-tail:]:
            by_slice = row.evidence.get("episode_sla_compliance_by_slice") or {}
            for s in SLICE_ORDER:
                if s in by_slice:
                    pooled[s].append(by_slice[s])
    return {s: (float(np.mean(v)) * 100 if v else float("nan")) for s, v in pooled.items()}


def live_compliance(live_root: Path, arm: str, seeds: list) -> dict:
    mode = ARM_REWARD_MODE[arm]
    pooled = {s: [] for s in SLICE_ORDER}
    for seed in seeds:
        path = arm_run_dir(live_root, arm, mode, seed) / "omega_log.jsonl"
        if not path.exists():
            print(f"[fig8] WARNING: missing {path}", file=sys.stderr)
            continue
        for row in read_omega_log(path):
            if row.step != -1:
                continue
            by_slice = row.evidence.get("episode_sla_compliance_by_slice") or {}
            for s in SLICE_ORDER:
                if s in by_slice:
                    pooled[s].append(by_slice[s])
    return {s: (float(np.mean(v)) * 100 if v else float("nan")) for s, v in pooled.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline-root", default="experiments/results/offline")
    ap.add_argument("--live-root", default="experiments/results/live_campaign")
    ap.add_argument("--offline-seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--live-seeds", type=int, nargs="+", default=[950, 951, 952])
    ap.add_argument("--tail-episodes", type=int, default=5)
    ap.add_argument("--out", default="experiments/plots/out/fig8_sim2real_parity")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 100], [0, 100], color="black", linewidth=0.8, linestyle="-", zorder=1, label="y = x (perfect transfer)")

    # Small deterministic per-arm jitter: all 4 learned arms land on the
    # SAME (offline%, live%) pair per slice to machine precision (see module
    # docstring / script stderr output) -- without jitter, 12 points collapse
    # onto 2 visible dots. Jitter is display-only; exact values are printed
    # below and belong in the caption/prose, not read off the plot.
    n_arms = len(LEARNED_ARMS)
    jitter = {arm: (i - (n_arms - 1) / 2) * 2.4 for i, arm in enumerate(LEARNED_ARMS)}

    rows_for_print = []
    for arm in LEARNED_ARMS:
        off = offline_tail_compliance(Path(args.offline_root), arm, args.offline_seeds, args.tail_episodes)
        live = live_compliance(Path(args.live_root), arm, args.live_seeds)
        style = ARM_STYLE[arm]
        for s in SLICE_ORDER:
            x, y = off[s], live[s]
            if np.isnan(x) or np.isnan(y):
                continue
            ax.scatter(
                x + jitter[arm], y + jitter[arm], color=style["color"], marker=SLICE_MARKER[s],
                edgecolor="black", linewidth=0.4, s=45, zorder=3,
            )
            rows_for_print.append((arm, s, x, y, y - x))

    # Two-part legend: color = arm, marker shape = slice.
    from matplotlib.lines import Line2D
    arm_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ARM_STYLE[a]["color"],
                          markeredgecolor="black", markersize=6, label=ARM_STYLE[a]["label"]) for a in LEARNED_ARMS]
    slice_handles = [Line2D([0], [0], marker=SLICE_MARKER[s], color="w", markerfacecolor="grey",
                            markeredgecolor="black", markersize=6, label=SLICE_STYLE[s]["label"]) for s in SLICE_ORDER]
    leg1 = ax.legend(handles=arm_handles, loc="lower right", frameon=False, fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=slice_handles, loc="upper left", frameon=False, fontsize=6, title="marker = slice", title_fontsize=6)

    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(f"Offline-predicted SLA compliance (%)\n(final {args.tail_episodes} eps/seed, 3 training seeds)")
    ax.set_ylabel("Live-measured SLA compliance (%)\n(3 live seeds x 5 episodes)")
    ax.set_title("Sim-to-real transfer parity, frozen weights\n(points jittered; see caption for exact values)", fontsize=8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig8] wrote {out_path}.pdf / .png")
    print("[fig8] arm, slice, offline%, live%, live-offline:")
    for arm, s, x, y, d in rows_for_print:
        print(f"  {arm:10s} {s:6s} offline={x:6.1f} live={y:6.1f} diff={d:+6.1f}")


if __name__ == "__main__":
    main()
