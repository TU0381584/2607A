#!/usr/bin/env python3
"""Figure 2: differentiated-shedding rate across GAT-CTDE's two encoder
fixes (docs/PAPER5_M2_gat_ctde.md sections 11-12). Single series, one
color -- these are 3 ordered stages of the SAME architecture's own
evolution, not distinct categories needing hue separation.

Counts are the already-verified, already-documented campaign results
(not re-derived here): the original architecture's 30/30-seed collapse
was established by direct inspection of every seed's eval omega log
(section 11); the two fixed counts are m2_seed_campaign.py's real
30-seed re-runs. No number here is invented -- all three are cited
directly from the campaign data already committed.

Usage:
    python3 experiments/plots/paper5_fig2_collapse_reduction.py \
        --out paper5/figures/fig2_collapse_reduction
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper5_common import M2_ARM_STYLE  # noqa: E402

STAGES = ["No normalization\n(original)", "+ LayerNorm\n(sec. 11)", "+ per-slice\nQ-heads (sec. 12)"]
DIFFERENTIATED = [0, 3, 22]  # out of 30 seeds each -- docs/PAPER5_M2_gat_ctde.md sections 11-12
N_SEEDS = 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper5/figures/fig2_collapse_reduction")
    args = ap.parse_args()

    color = M2_ARM_STYLE["gat_ctde"]["color"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(STAGES))
    pct = [100.0 * d / N_SEEDS for d in DIFFERENTIATED]
    bars = ax.bar(x, pct, color=color, width=0.55, edgecolor="white", linewidth=0.5)
    for xi, (d, p) in zip(x, zip(DIFFERENTIATED, pct)):
        ax.annotate(f"{d}/{N_SEEDS}", xy=(xi, p), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel("Seeds with genuine\ndifferentiated shedding (%)")
    ax.set_ylim(0, 100)
    ax.set_title("GAT-CTDE collapse reduction across encoder fixes")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig2] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
