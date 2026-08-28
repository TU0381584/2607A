#!/usr/bin/env python3
"""M27 scaling reframe: collapse rate per arm, at N=7 and N=19, original
M6 sample (ClosedLoopKpmSource) vs. this session's recalibrated resample
(RealisticServedKpmSource). Two panels because the two scales have very
different statistical power on the "original" side, and showing that
honestly matters more than a compact single panel:

(a) N=7: the original M6 number is a single n=3 pilot sample (0/3
    collapsed for every arm: GAT-CTDE and single-agent DQN both held
    1.000 precision, independent_dqn 0.787 -- see
    docs/PAPER5_M6_topology.md, explicitly never resampled at scale
    because M6's own writeup judged N=19's collapse-rate question the
    higher-value target). Plotted as a bar with NO error bar (there is
    none to report, not a zero-width one) -- the recalibrated side is a
    genuinely powered 12-seed x 3-topology resample.
(b) N=19: both sides are properly powered (original: 12-seed primary
    sample pooled with two replication samples per
    docs/PAPER5_M6_topology.md Part 11; recalibrated: this session's own
    12 seeds x 3 topologies), both bootstrapped over independent seeds.

The original M6 numbers are read from docs/PAPER5_M6_topology.md's own
already-published values (cited, not recomputed); the recalibrated
numbers are computed directly from this session's own raw omega logs
via m6_correctness_metrics.per_seed_metrics_per_gnb, the same function
M6's own analysis used.

Usage:
    python3 experiments/plots/paper5_fig_m27_scaling_reframe.py \
        --results-dir experiments/results/m27_scaling_reframe \
        --out Papers_4-5/Paper_5/WPC/figures/fig9_m27_scaling_reframe
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
ORIG_N19 = {
    "single_agent_dqn": (1.0, 1.0, 1.0),          # 15/15, point estimate (no separate CI reported)
    "gat_ctde": (0.354, 0.255, 0.458),             # pooled 35.4% [25.5%, 45.8%], 192 cells / 64 seeds
    "independent_dqn": (0.0, 0.0, 0.0),            # 0/36
}
# N=7 pilot, n=3 seeds, no CI ever reported (M6's own writeup: "not yet
# a powered result") -- point estimates only, plotted with no error bar.
ORIG_N7 = {
    "single_agent_dqn": 0.0,   # held 1.000 precision, 0/3 collapsed
    "gat_ctde": 0.0,           # held 1.000 precision, 0/3 collapsed
    "independent_dqn": 0.0,    # 0.787 precision, 0/3 collapsed
}


def eval_path(results_dir: str, n: int, arm: str, topo: str, seed: int) -> str:
    base = f"{results_dir}/n{n}_{topo}"
    if arm == "single_agent_dqn":
        return f"{base}/{arm}/seed{seed}/eval/dqn/offline_eval/rep_0/omega_log.jsonl"
    return f"{base}/{arm}/seed{seed}/eval/omega_log.jsonl"


def recalibrated_collapse_rates(results_dir: str, n: int):
    out = {}
    for arm in ARMS:
        per_seed_frac = []
        for seed in SEEDS:
            collapsed = 0
            for topo in TOPOLOGIES:
                _, mmtc_b, total_b = per_seed_metrics_per_gnb(eval_path(results_dir, n, arm, topo, seed), n)
                if total_b == 0:
                    collapsed += 1
            per_seed_frac.append(collapsed / len(TOPOLOGIES))
        arr = np.array(per_seed_frac)
        lo, hi = bootstrap_ci(arr)
        out[arm] = (arr.mean(), lo, hi)
        print(f"[m27-reframe] N={n} {arm}: recalibrated collapse rate = {arr.mean():.3f} [{lo:.3f}, {hi:.3f}] "
              f"({int(round(arr.sum()*3))}/36 cells)")
    return out


def draw_panel(ax, orig_vals, orig_errs, recal, tag, ylabel, show_legend):
    x = np.arange(len(ARMS))
    width = 0.32
    orig_color, recal_color = "#898781", "#1a9e8f"

    recal_vals = [recal[a][0] for a in ARMS]
    recal_lo = [recal[a][0] - recal[a][1] for a in ARMS]
    recal_hi = [recal[a][2] - recal[a][0] for a in ARMS]

    if orig_errs is None:
        ax.bar(x - width / 2, orig_vals, width, color=orig_color, alpha=0.85, edgecolor="white", linewidth=0.5,
               label="Original (M6, $n{=}3$ pilot, no CI)")
    else:
        orig_lo, orig_hi = orig_errs
        ax.bar(x - width / 2, orig_vals, width, color=orig_color, alpha=0.85, edgecolor="white", linewidth=0.5,
               yerr=[orig_lo, orig_hi], capsize=3, ecolor="#0b0b0b", error_kw={"linewidth": 1.0},
               label="Original (M6, ClosedLoopKpmSource)")
    ax.bar(x + width / 2, recal_vals, width, color=recal_color, alpha=0.85, edgecolor="white", linewidth=0.5,
           yerr=[recal_lo, recal_hi], capsize=3, ecolor="#0b0b0b", error_kw={"linewidth": 1.0},
           label="Recalibrated (M27, RealisticServedKpmSource)")

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    ax.set_title(tag, loc="left")
    if show_legend:
        ax.legend(loc="upper center", frameon=False, fontsize=6, bbox_to_anchor=(1.05, 1.30), ncol=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="experiments/results/m27_scaling_reframe")
    ap.add_argument("--out", default="Papers_4-5/Paper_5/WPC/figures/fig9_m27_scaling_reframe")
    args = ap.parse_args()

    recal_n7 = recalibrated_collapse_rates(args.results_dir, 7)
    recal_n19 = recalibrated_collapse_rates(args.results_dir, 19)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_COLUMN_WIDTH_IN * 2.0, IEEE_COLUMN_WIDTH_IN * 1.0))

    draw_panel(ax1, [ORIG_N7[a] for a in ARMS], None, recal_n7, "(a) $N{=}7$",
               "Collapse rate\n(mean $\\pm$ 95% CI)", show_legend=False)
    orig_n19_vals = [ORIG_N19[a][0] for a in ARMS]
    orig_n19_lo = [ORIG_N19[a][0] - ORIG_N19[a][1] for a in ARMS]
    orig_n19_hi = [ORIG_N19[a][2] - ORIG_N19[a][0] for a in ARMS]
    draw_panel(ax2, orig_n19_vals, (orig_n19_lo, orig_n19_hi), recal_n19, "(b) $N{=}19$",
               "Collapse rate\n(mean $\\pm$ 95% CI)", show_legend=True)

    fig.subplots_adjust(bottom=0.2, top=0.72, wspace=0.5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[m27-reframe] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
