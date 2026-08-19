#!/usr/bin/env python3
"""Figure 8: M6 cluster-size scaling (N=19), per-seed evidence --
(a) collapse rate (fraction of arm x topology cells with zero blocks in
eval) by arm, primary sample (seeds 900-911, n=12/topology) against the
independent replication sample (seeds 1000-1002, n=3/topology), pooled
across all three topologies per arm -- the same primary-vs-replication
pairing fig7_replication_forest already uses elsewhere in this paper;
(b) block precision by topology for the two arms that ever produce a
defined precision (single_agent_dqn is 100% collapsed at every cell in
both samples, so it has no precision to plot), primary sample only,
non-collapsed seeds, individual per-seed points behind the mean --
GAT-CTDE's topology-dependent secondary failure mode (perfect precision
at fully_connected, near-zero at ring/hex for some seeds) is the reason
this panel is split by topology rather than pooled.

Reuses m6_correctness_metrics.per_seed_metrics_per_gnb (imported, not
reimplemented).

Usage:
    python3 experiments/plots/paper5_fig8_m6_topology.py \
        --pilot-dir experiments/results/m6_pilot \
        --out paper5/figures/fig8_m6_topology
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m6_correctness_metrics import per_seed_metrics_per_gnb  # noqa: E402
from paper5_common import M2_ARM_STYLE, bootstrap_ci  # noqa: E402

TOPOLOGIES = ["fully_connected", "ring", "hex"]
TOPOLOGY_LABELS = {"fully_connected": "Fully-\nconnected", "ring": "Ring", "hex": "Hex"}
ARMS = ["single_agent_dqn", "gat_ctde", "independent_dqn"]
N_GNB = 19


def eval_path(pilot_dir: Path, combo: str, arm: str, seed: int) -> Path:
    base = pilot_dir / combo / arm / f"seed{seed}" / "eval"
    flat = base / "omega_log.jsonl"
    if flat.exists():
        return flat
    return base / "dqn" / "offline_eval" / "rep_0" / "omega_log.jsonl"


def collect(pilot_dir: Path, suffix: str, seeds: list) -> dict:
    """Returns {(arm, topology): [(collapsed: bool, precision_or_None), ...]}."""
    out = {}
    for topology in TOPOLOGIES:
        combo = f"n19_{topology}_capfix{suffix}"
        for arm in ARMS:
            rows = []
            for seed in seeds:
                p = eval_path(pilot_dir, combo, arm, seed)
                if not p.exists():
                    continue
                _mrpg, mmtc_b, total_b = per_seed_metrics_per_gnb(str(p), N_GNB)
                if total_b == 0:
                    rows.append((True, None))
                else:
                    rows.append((False, mmtc_b / total_b))
            out[(arm, topology)] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-dir", default="experiments/results/m6_pilot")
    ap.add_argument("--out", default="paper5/figures/fig8_m6_topology")
    args = ap.parse_args()
    pilot_dir = Path(args.pilot_dir)

    primary = collect(pilot_dir, "", list(range(900, 912)))
    replication = collect(pilot_dir, "_replication", [1000, 1001, 1002])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---- (a) collapse rate by arm, primary vs. replication, pooled over topology ----
    def collapse_rate(sample: dict, arm: str):
        rows = [r for t in TOPOLOGIES for r in sample[(arm, t)]]
        n = len(rows)
        n_collapsed = sum(1 for c, _ in rows if c)
        return n_collapsed, n

    x = np.arange(len(ARMS))
    width = 0.34
    primary_rates, primary_labels = [], []
    repl_rates, repl_labels = [], []
    for arm in ARMS:
        pc, pn = collapse_rate(primary, arm)
        rc, rn = collapse_rate(replication, arm)
        primary_rates.append(pc / pn if pn else np.nan)
        primary_labels.append(f"{pc}/{pn}")
        repl_rates.append(rc / rn if rn else np.nan)
        repl_labels.append(f"{rc}/{rn}")

    b1 = ax1.bar(x - width / 2, primary_rates, width, color="#5b5a52",
                 label="Primary (seeds 900-911, n=12/topology)")
    b2 = ax1.bar(x + width / 2, repl_rates, width, color="#c3c2b7",
                 label="Replication (seeds 1000-1002, n=3/topology)")
    for bar, lab in zip(b1, primary_labels):
        ax1.annotate(lab, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=6, xytext=(0, 2), textcoords="offset points")
    for bar, lab in zip(b2, repl_labels):
        ax1.annotate(lab, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=6, xytext=(0, 2), textcoords="offset points")

    ax1.set_xticks(x)
    ax1.set_xticklabels([M2_ARM_STYLE[a]["label"].replace(" (proposed)", "") for a in ARMS],
                         fontsize=6.5)
    ax1.set_ylabel("Collapse rate\n(fraction of cells, 0 blocks in eval)")
    ax1.set_ylim(0, 1.18)
    ax1.set_title("(a) N=19 collapse rate, pooled over topology")
    ax1.legend(loc="upper center", frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.22))

    # ---- (b) block precision by topology, primary sample, non-collapsed seeds ----
    rng = np.random.RandomState(0)
    x_cat = np.arange(len(TOPOLOGIES))
    plot_arms = ["gat_ctde", "independent_dqn"]
    offsets = {"gat_ctde": -0.14, "independent_dqn": 0.14}
    for arm in plot_arms:
        style = M2_ARM_STYLE[arm]
        means, los, his = [], [], []
        for xi, topology in zip(x_cat, TOPOLOGIES):
            vals = np.array([p for c, p in primary[(arm, topology)] if not c])
            if len(vals):
                jitter = rng.uniform(-0.09, 0.09, size=len(vals))
                ax2.scatter(np.full(len(vals), xi + offsets[arm]) + jitter, vals,
                            s=12, color=style["color"], alpha=0.4, linewidth=0, zorder=2)
                mean = vals.mean()
                lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (mean, mean)
                means.append(mean)
                los.append(mean - lo)
                his.append(hi - mean)
            else:
                means.append(np.nan)
                los.append(0)
                his.append(0)
        ax2.errorbar(x_cat + offsets[arm], means, yerr=[los, his], color=style["color"],
                     marker=style["marker"], markersize=5.5, linewidth=1.4, capsize=3,
                     linestyle="none", zorder=4, label=style["label"].replace(" (proposed)", ""))

    ax2.set_xlim(-0.5, len(TOPOLOGIES) - 0.5)
    ax2.set_xticks(x_cat)
    ax2.set_xticklabels([TOPOLOGY_LABELS[t] for t in TOPOLOGIES])
    ax2.set_ylabel("Block precision\n(non-collapsed seeds only)")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_title("(b) Precision by topology, primary sample")
    ax2.legend(loc="lower left", frameon=False, fontsize=6.5)

    fig.suptitle("M6 (N=19): collapse rate and precision, primary vs. replication", y=1.03, fontsize=9)
    fig.subplots_adjust(bottom=0.30, wspace=0.35)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig8] wrote {out_path}.pdf / .png")
    for arm in ARMS:
        pc, pn = collapse_rate(primary, arm)
        rc, rn = collapse_rate(replication, arm)
        print(f"  {arm}: primary collapse={pc}/{pn} ({pc/pn:.2f})  replication={rc}/{rn} ({rc/rn:.2f})")
    for arm in plot_arms:
        for topology in TOPOLOGIES:
            vals = [p for c, p in primary[(arm, topology)] if not c]
            print(f"  primary {arm}/{topology}: n_defined={len(vals)} "
                  f"precisions={[round(v, 3) for v in vals]}")


if __name__ == "__main__":
    main()
