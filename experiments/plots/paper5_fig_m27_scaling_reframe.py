#!/usr/bin/env python3
"""M27 scaling reframe: collapse rate per arm, N=19, original M6 sample
(ClosedLoopKpmSource, already-published numbers -- single-agent DQN
15/15, GAT-CTDE pooled 35.4% [25.5%,45.8%] across 3 samples/192 cells,
independent DQN 0/36) vs. this session's recalibrated resample
(RealisticServedKpmSource, 12 seeds x 3 topologies = 36 cells/arm,
bootstrapped over the 12 independent seeds the same way M6's own
collapse-rate CI was computed, since collapse status correlates within
a seed across its three topologies).

The original M6 numbers are read from docs/PAPER5_M6_topology.md's own
already-published, already-reviewed values (not re-derived here --
that data predates this session's own campaign and is cited, not
recomputed) as a literal --orig flag per arm; the recalibrated numbers
are computed directly from this session's own raw omega logs via
m6_correctness_metrics.per_seed_metrics_per_gnb, exactly the same
function M6's own analysis used.

Usage:
    python3 experiments/plots/paper5_fig_m27_scaling_reframe.py \
        --results-dir experiments/results/m27_scaling_reframe \
        --out paper5_wpc/figures/fig9_m27_scaling_reframe
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paper5_common import IEEE_COLUMN_WIDTH_IN  # noqa: E402,F401
from m2_correctness_metrics import bootstrap_ci  # noqa: E402
from m6_correctness_metrics import per_seed_metrics_per_gnb  # noqa: E402

ARMS = ["single_agent_dqn", "gat_ctde", "independent_dqn"]
ARM_LABELS = {"single_agent_dqn": "single-agent\nDQN", "gat_ctde": "GAT-CTDE", "independent_dqn": "independent\nDQN"}
SEEDS = list(range(900, 912))
TOPOLOGIES = ["fully_connected", "ring", "hex"]

# docs/PAPER5_M6_topology.md's own already-published values, cited not recomputed.
ORIG_M6 = {
    "single_agent_dqn": (1.0, 1.0, 1.0),          # 15/15, CI not separately reported (point estimate only)
    "gat_ctde": (0.354, 0.255, 0.458),             # pooled 35.4% [25.5%, 45.8%], 192 cells / 64 seeds
    "independent_dqn": (0.0, 0.0, 0.0),            # 0/36
}


def eval_path(results_dir: str, arm: str, topo: str, seed: int) -> str:
    if arm == "single_agent_dqn":
        return f"{results_dir}/n19_{topo}/{arm}/seed{seed}/eval/dqn/offline_eval/rep_0/omega_log.jsonl"
    return f"{results_dir}/n19_{topo}/{arm}/seed{seed}/eval/omega_log.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="experiments/results/m27_scaling_reframe")
    ap.add_argument("--out", default="paper5_wpc/figures/fig9_m27_scaling_reframe")
    args = ap.parse_args()

    recal = {}
    for arm in ARMS:
        per_seed_frac = []
        for seed in SEEDS:
            collapsed = 0
            for topo in TOPOLOGIES:
                _, mmtc_b, total_b = per_seed_metrics_per_gnb(eval_path(args.results_dir, arm, topo, seed), 19)
                if total_b == 0:
                    collapsed += 1
            per_seed_frac.append(collapsed / len(TOPOLOGIES))
        arr = np.array(per_seed_frac)
        lo, hi = bootstrap_ci(arr)
        recal[arm] = (arr.mean(), lo, hi)
        print(f"[m27-reframe] {arm}: recalibrated collapse rate = {arr.mean():.3f} [{lo:.3f}, {hi:.3f}] "
              f"({int(arr.sum()*3)}/36 cells)")

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN * 1.3, IEEE_COLUMN_WIDTH_IN * 0.95))
    x = np.arange(len(ARMS))
    width = 0.32
    orig_color, recal_color = "#898781", "#1a9e8f"

    orig_vals = [ORIG_M6[a][0] for a in ARMS]
    orig_lo = [ORIG_M6[a][0] - ORIG_M6[a][1] for a in ARMS]
    orig_hi = [ORIG_M6[a][2] - ORIG_M6[a][0] for a in ARMS]
    recal_vals = [recal[a][0] for a in ARMS]
    recal_lo = [recal[a][0] - recal[a][1] for a in ARMS]
    recal_hi = [recal[a][2] - recal[a][0] for a in ARMS]

    ax.bar(x - width / 2, orig_vals, width, color=orig_color, alpha=0.85, edgecolor="white", linewidth=0.5,
           yerr=[orig_lo, orig_hi], capsize=3, ecolor="#0b0b0b", error_kw={"linewidth": 1.0},
           label="Original (M6, ClosedLoopKpmSource)")
    ax.bar(x + width / 2, recal_vals, width, color=recal_color, alpha=0.85, edgecolor="white", linewidth=0.5,
           yerr=[recal_lo, recal_hi], capsize=3, ecolor="#0b0b0b", error_kw={"linewidth": 1.0},
           label="Recalibrated (M27, RealisticServedKpmSource)")

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARMS])
    ax.set_ylabel("Collapse rate at $N{=}19$\n(mean $\\pm$ 95% CI)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.22), ncol=1)

    fig.subplots_adjust(bottom=0.18, top=0.78)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[m27-reframe] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
