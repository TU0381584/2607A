#!/usr/bin/env python3
"""Figure 5: backlog per slice by arm -- CDF of per-slice SLA margin
(a continuous, Lmax-normalized proxy for backlog-driven SLA-violation
severity: 1.0=comfortably within budget, 0.0=at/beyond budget -- see
reward.py's ViolationCheck / RunSummary.sla_margin_by_slice's docstring).

CAVEAT, stated directly rather than glossed over: this is NOT raw
dl_mac_buffer_occupation in bytes -- the standard per-step omega evidence
dict does not carry that raw value (only the framework's Lmax-normalized
derived margin). This figure is what proves Phase 1's contention gate
carried into the campaign's actual results (do policies differ in how
often they keep slices near-comfortable vs. at-budget?); Phase 1's own
contention-gate script/log
(experiments/logs/phase1/embb_final_*.jsonl) carries the raw
BYTE-level dl_mac_buffer_occupation evidence for the gate's own pass/fail
claim and is the citation for THAT specific claim, not this figure.

Usage:
    python3 experiments/plots/fig5_backlog.py \
        --live-root experiments/results/live --seeds 256 257 258 \
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
# display only (raw data is untouched) -- without this, a linear-axis CDF
# compresses the entire informative [-1,1] region into an invisible sliver
# next to the massively-negative tail, which silently made a badly-violating
# arm's CDF look like it was sitting at ~1.0 (comfortable) the whole time --
# caught by cross-checking against fig2's compliance numbers before
# shipping this figure.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-root", default="experiments/results/live")
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--out", default="experiments/plots/out/fig5_backlog")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(SLICE_ORDER), figsize=(3.5 * len(SLICE_ORDER) / 1.3, 2.6), sharey=True)

    for slice_idx, slice_id in enumerate(SLICE_ORDER):
        ax = axes[slice_idx]
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
        ax.set_title(SLICE_STYLE[slice_id]["label"], fontsize=8)
        ax.set_xlabel(f"SLA margin (1=comfortable, 0=at budget,\nclipped at {MARGIN_FLOOR})")
        ax.set_xlim(MARGIN_FLOOR - 0.05, 1.05)

    axes[0].set_ylabel("CDF")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=5.5)
    fig.suptitle("Backlog-driven SLA margin CDF by arm", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig5] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
