#!/usr/bin/env python3
"""Figure 8: M6 cluster-size scaling (N=19), per-seed evidence --
(a) collapse rate (fraction of arm x topology cells with zero blocks in
eval) by arm, three independent seed samples (primary 900-911,
replication 1000-1002, and an M7 extension 2000-2048 aimed specifically
at narrowing GAT-CTDE's own collapse-rate estimate -- gat_ctde only,
the other two arms were not re-run since their profiles were already
well-supported), pooled across all three topologies per arm -- the same
multi-sample-forest logic fig7_replication_forest already uses
elsewhere in this paper. GAT-CTDE's bar group also carries the
seed-level bootstrap 95% CI on the 3-sample combined estimate (computed
per independent seed, not per pooled cell, since collapse status
correlates within a seed across its three topologies);
(b) block precision by topology for the two arms that ever produce a
defined precision (single_agent_dqn is 100% collapsed at every cell in
every sample, so it has no precision to plot), primary sample only,
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
    extension = collect(pilot_dir, "", list(range(2000, 2049)))  # five batches, same method, unified

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---- (a) collapse rate by arm, 3 samples, pooled over topology ----
    def collapse_rate(sample: dict, arm: str):
        rows = [r for t in TOPOLOGIES for r in sample[(arm, t)]]
        n = len(rows)
        n_collapsed = sum(1 for c, _ in rows if c)
        return n_collapsed, n

    x = np.arange(len(ARMS))
    width = 0.26
    sample_defs = [("Primary\n(900-911)", primary, "#5b5a52", -width),
                   ("Replication\n(1000-1002)", replication, "#c3c2b7", 0.0),
                   ("Extension\n(2000-2048,\ngat_ctde only)", extension, "#8fa6c9", width)]
    for label, sample, color, offset in sample_defs:
        rates, labels = [], []
        for arm in ARMS:
            c, n = collapse_rate(sample, arm)
            rates.append(c / n if n else np.nan)
            labels.append(f"{c}/{n}" if n else "")
        bars = ax1.bar(x + offset, rates, width, color=color, label=label)
        for bar, lab in zip(bars, labels):
            if lab:
                ax1.annotate(lab, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha="center", va="bottom", fontsize=5.5, xytext=(0, 2),
                             textcoords="offset points")

    # Combined 3-sample, seed-level bootstrap CI for gat_ctde specifically
    # (the arm the extension sample targeted) -- each independent seed
    # contributing its own collapse fraction across 3 topologies, not the
    # pooled cells (collapse status correlates within a seed across
    # topologies, so pooling cells understates the real uncertainty).
    gat_idx = ARMS.index("gat_ctde")
    seed_fracs = []
    for sample in (primary, replication, extension):
        by_seed = {}
        for t in TOPOLOGIES:
            for i, (c, _p) in enumerate(sample[("gat_ctde", t)]):
                by_seed.setdefault(i, []).append(c)
        seed_fracs.extend(np.mean(v) for v in by_seed.values())
    seed_fracs = np.array(seed_fracs)
    combined_mean = seed_fracs.mean()
    rng = np.random.RandomState(0)
    boots = [rng.choice(seed_fracs, size=len(seed_fracs), replace=True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    ax1.errorbar([gat_idx], [combined_mean], yerr=[[combined_mean - ci_lo], [ci_hi - combined_mean]],
                 color="#0b0b0b", marker="D", markersize=5, linewidth=1.4, capsize=4, zorder=5,
                 label=f"Combined, n={len(seed_fracs)} seeds\n({combined_mean:.0%} [{ci_lo:.0%}, {ci_hi:.0%}])")

    ax1.set_xticks(x)
    ax1.set_xticklabels([M2_ARM_STYLE[a]["label"].replace(" (proposed)", "") for a in ARMS],
                         fontsize=6.5)
    ax1.set_ylabel("Collapse rate\n(fraction of cells, 0 blocks in eval)")
    ax1.set_ylim(0, 1.30)
    ax1.set_title("(a) N=19 collapse rate, 3 samples")
    ax1.legend(loc="upper center", frameon=False, fontsize=5.3, ncol=2, bbox_to_anchor=(0.5, -0.24))

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

    fig.suptitle("M6/M7 (N=19): collapse rate and precision across 3 seed samples", y=1.03, fontsize=9)
    fig.subplots_adjust(bottom=0.34, wspace=0.35)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig8] wrote {out_path}.pdf / .png")
    for arm in ARMS:
        pc, pn = collapse_rate(primary, arm)
        rc, rn = collapse_rate(replication, arm)
        ec, en = collapse_rate(extension, arm)
        ext_part = f"extension={ec}/{en} ({ec/en:.2f})" if en else "extension=n/a"
        print(f"  {arm}: primary={pc}/{pn} ({pc/max(pn,1):.2f})  "
              f"replication={rc}/{rn} ({rc/max(rn,1):.2f})  {ext_part}")
    print(f"  gat_ctde combined ({len(seed_fracs)} seeds): mean={combined_mean:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}]")
    for arm in plot_arms:
        for topology in TOPOLOGIES:
            vals = [p for c, p in primary[(arm, topology)] if not c]
            print(f"  primary {arm}/{topology}: n_defined={len(vals)} "
                  f"precisions={[round(v, 3) for v in vals]}")


if __name__ == "__main__":
    main()
