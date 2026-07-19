#!/usr/bin/env python3
"""Figure 5: backlog per slice by arm.

Two rows per slice column:
  - Top row: per-step SLA margin (a continuous, Lmax-normalized proxy for
    backlog-driven SLA-violation severity: 1.0=comfortably within budget,
    0.0=at budget, unbounded negative under real contention -- see
    reward.py's ViolationCheck) over ONE representative episode, baseline
    (bad seed) vs. one learned arm, on a symlog y-axis -- the only way to
    show a +1.0-margin trajectory and a -1e6-margin trajectory on the same
    axes without one being invisible.
  - Bottom row: CDF of per-slice SLA margin pooled across all 3 seeds x 5
    episodes x 60 steps (kept from the original figure -- it is the
    aggregate complement to the top row's single-episode illustration).

CAVEAT, same as the original figure: this is NOT raw
dl_mac_buffer_occupation in bytes -- the standard per-step omega evidence
dict does not carry that raw value (only the framework's Lmax-normalized
derived margin). Phase 1's own contention-gate trace
(experiments/logs/phase1/embb_final_*.jsonl) carries the raw BYTE-level
dl_mac_buffer_occupation evidence for the gate's own pass/fail claim, not
this figure.

Usage:
    python3 experiments/plots/fig5_backlog.py \
        --live-root experiments/results/live_campaign --seeds 950 951 952 \
        --episode-arm dqn_sla --episode-seed 950 --episode 1 \
        --out experiments/plots/out/fig5_backlog
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

# per_slice_sla_margin is NOT pre-clipped to [0,1] -- reward.py's
# ViolationCheck computes it as an unbounded continuous distance, and under
# real contention it was observed going to roughly -1e6 (matching Phase 1's
# multi-order-of-magnitude backlog blowup). Clip at MARGIN_FLOOR for
# CDF display only (raw data is untouched) -- without this, a linear-axis
# CDF compresses the entire informative [-1,1] region into an invisible
# sliver next to the massively-negative tail, which silently made a
# badly-violating arm's CDF look like it was sitting at ~1.0 (comfortable)
# the whole time -- caught by cross-checking against fig2's compliance
# numbers before shipping this figure.
MARGIN_FLOOR = -1.5


def collect_margins(omega_path: Path) -> dict:
    out = {s: [] for s in SLICE_ORDER}
    for row in read_omega_log(omega_path):
        if row.step < 1:
            continue
        margins = row.evidence.get("per_slice_sla_margin") or {}
        for s in SLICE_ORDER:
            if s in margins:
                out[s].append(max(MARGIN_FLOOR, margins[s]))
    return out


def load_episode_margins(omega_path: Path, episode: int, run_id: str = None) -> dict:
    """slice_id -> (steps[], margins[]) for one specific episode, matching
    fig4_ceiling_trajectories.py's run_id-disambiguation convention: episode
    numbers repeat across health-checked batches, so filtering by episode
    alone would silently overlay distinct episodes (see that script's
    docstring / CAMPAIGN_LOG.md). If run_id is not given, the FIRST run_id
    encountered in the file is used."""
    series = {s: ([], []) for s in SLICE_ORDER}
    resolved_run_id = run_id
    for row in read_omega_log(omega_path):
        if row.step < 1 or row.episode != episode:
            continue
        if resolved_run_id is None:
            resolved_run_id = row.run_id
        if row.run_id != resolved_run_id:
            continue
        margins = row.evidence.get("per_slice_sla_margin") or {}
        for s in SLICE_ORDER:
            if s in margins:
                series[s][0].append(row.step)
                series[s][1].append(margins[s])
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live_campaign")
    ap.add_argument("--seeds", type=int, nargs="+", default=[950, 951, 952])
    ap.add_argument("--episode-arm", default="dqn_sla", help="learned arm to show in the time-series row")
    ap.add_argument("--episode-seed", type=int, default=950, help="seed for the time-series row (a baseline bad seed)")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--out", default="experiments/plots/out/fig5_backlog")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, len(SLICE_ORDER), figsize=(3.5 * len(SLICE_ORDER) / 1.3, 4.6), sharey="row")

    # ---- top row: representative-episode time series, symlog y-axis ----
    for arm in ["baseline", args.episode_arm]:
        mode = ARM_REWARD_MODE[arm]
        omega_path = arm_run_dir(args.live_root, arm, mode, args.episode_seed) / "omega_log.jsonl"
        if not omega_path.exists():
            print(f"[fig5] WARNING: missing {omega_path}", file=sys.stderr)
            continue
        series = load_episode_margins(omega_path, args.episode)
        style = ARM_STYLE[arm]
        for col, slice_id in enumerate(SLICE_ORDER):
            steps, margins = series[slice_id]
            if steps:
                axes[0, col].plot(steps, margins, color=style["color"], linestyle=style["linestyle"],
                                   label=style["label"], linewidth=1.1)

    for col, slice_id in enumerate(SLICE_ORDER):
        ax = axes[0, col]
        ax.set_yscale("symlog", linthresh=1.0)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(SLICE_STYLE[slice_id]["label"], fontsize=8)
        ax.set_xlabel("Step")
    axes[0, 0].set_ylabel(f"SLA margin\n(episode {args.episode}, seed {args.episode_seed}, symlog)")
    axes[0, -1].legend(loc="lower left", frameon=False, fontsize=5.5)

    # ---- bottom row: pooled CDF (unchanged from the original figure) ----
    for col, slice_id in enumerate(SLICE_ORDER):
        ax = axes[1, col]
        for arm in ARMS:
            mode = ARM_REWARD_MODE[arm]
            pooled = []
            for seed in args.seeds:
                omega_path = arm_run_dir(args.live_root, arm, mode, seed) / "omega_log.jsonl"
                if omega_path.exists():
                    pooled.extend(collect_margins(omega_path)[slice_id])
            if not pooled:
                continue
            sorted_vals = np.sort(pooled)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            style = ARM_STYLE[arm]
            ax.plot(sorted_vals, cdf, color=style["color"], linestyle=style["linestyle"],
                    label=style["label"], linewidth=1.0)
        ax.set_xlabel(f"SLA margin (1=comfortable, 0=at budget,\nclipped at {MARGIN_FLOOR})")
        ax.set_xlim(MARGIN_FLOOR - 0.05, 1.05)

    axes[1, 0].set_ylabel("CDF (pooled, n=3 seeds)")
    axes[1, -1].legend(loc="lower right", frameon=False, fontsize=5.5)

    fig.suptitle("Backlog-driven SLA margin: representative episode vs. pooled CDF", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig5] wrote {out_path}.pdf / .png "
          f"(top row: baseline vs {args.episode_arm}, seed {args.episode_seed}, episode {args.episode})")


if __name__ == "__main__":
    main()
