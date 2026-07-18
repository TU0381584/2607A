#!/usr/bin/env python3
"""Figure 4 (the Fig-5-style headline): commanded max_ratio per slice over
one representative episode, baseline vs. the best learned arm -- reads the
per-step `ceilings` field from each arm's live-eval omega log (rollup rows,
step==-1, are skipped; only step>=1 records carry a ceiling snapshot).

Usage:
    python3 experiments/plots/fig4_ceiling_trajectories.py \
        --live-root experiments/results/live --seed 256 \
        --best-arm dqn_qoe --episode 1 \
        --out experiments/plots/out/fig4_ceiling_trajectories
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, SLICE_ORDER, SLICE_STYLE, arm_run_dir, read_omega_log  # noqa: E402

ARM_REWARD_MODE = {
    "baseline": "sla", "dqn_sla": "sla", "a2c_sla": "sla",
    "dqn_qoe": "qoe", "a2c_qoe": "qoe",
}
GNB_ID = "gnb-0"


def load_episode_ceilings(omega_path: Path, episode: int, run_id: str = None) -> dict:
    """slice_id -> (steps[], max_ratio[]) for the given episode index.

    IMPORTANT: episode numbers are only unique WITHIN a run_id (batch) --
    experiments/scripts/run_live_eval_arm.py reseeds a fresh local episode
    counter starting at 1 for every batch, so "episode 1" appears once per
    batch (see CAMPAIGN_LOG.md's documented batch-reseeding caveat). Without
    also constraining run_id, filtering by episode number alone silently
    overlays multiple distinct episodes from different batches onto one
    plot. If run_id is not given, the FIRST run_id encountered in the file
    is used (deterministic given JSONL is append-ordered), not "all
    run_ids" -- this was caught by inspecting a rendered figure that showed
    two overlapping ceiling trajectories where only one was expected.
    """
    series = {s: ([], []) for s in SLICE_ORDER}
    resolved_run_id = run_id
    for row in read_omega_log(omega_path):
        if row.step < 1 or row.episode != episode:
            continue
        if resolved_run_id is None:
            resolved_run_id = row.run_id
        if row.run_id != resolved_run_id:
            continue
        ceilings = row.evidence.get("ceilings") or {}
        for slice_id in SLICE_ORDER:
            key = f"{GNB_ID}:{slice_id}"
            if key in ceilings:
                series[slice_id][0].append(row.step)
                series[slice_id][1].append(ceilings[key]["max_ratio"])
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live")
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--best-arm", required=True, help="e.g. dqn_qoe -- chosen after inspecting Phase 3 results")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--out", default="experiments/plots/out/fig4_ceiling_trajectories")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(SLICE_ORDER), 1, sharex=True,
                              figsize=(3.5, 0.75 * 3.5 * len(SLICE_ORDER) / 2))

    for arm in ["baseline", args.best_arm]:
        mode = ARM_REWARD_MODE[arm]
        omega_path = arm_run_dir(args.live_root, arm, mode, args.seed) / "omega_log.jsonl"
        if not omega_path.exists():
            print(f"[fig4] WARNING: missing {omega_path}", file=sys.stderr)
            continue
        series = load_episode_ceilings(omega_path, args.episode)
        style = ARM_STYLE[arm]
        for ax, slice_id in zip(axes, SLICE_ORDER):
            steps, ratios = series[slice_id]
            if steps:
                ax.step(steps, ratios, where="post", color=style["color"],
                        linestyle=style["linestyle"], label=style["label"], linewidth=1.1)

    for ax, slice_id in zip(axes, SLICE_ORDER):
        ax.set_ylabel(f"{SLICE_STYLE[slice_id]['label']}\nmax_ratio")
    axes[-1].set_xlabel("Step (within representative episode)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Commanded PRB ceiling: baseline vs. best learned arm", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig4] wrote {out_path}.pdf / .png (baseline vs {args.best_arm}, episode {args.episode}, seed {args.seed})")


if __name__ == "__main__":
    main()
