#!/usr/bin/env python3
"""Figure 7: cross-sample forest plot -- the paper's headline paired
comparisons (M2 reward edge, M3 federation cost, M4 dropout/churn cost),
each computed independently on the committed seed sample (900-929/
900-909) and the fresh, disjoint replication sample (1000-1029/1000-1009,
docs/PAPER5_REPLICATION_FINDINGS.md), plotted on the same axis so a
reader can see at a glance which effects hold up across samples and
which do not -- most directly, that independent DQN's churn cost flips
from indistinguishable-from-zero (committed) to significant and LARGER
than the coordination-dependent arms (fresh), the retraction documented
in Section~\ref{sec:results-m4}D and docs/PAPER5_M4_disruption.md's
RETRACTED note. No number here is hand-typed from those documents --
every point and CI is recomputed directly from each sample's own raw
omega logs via the same bootstrap_ci/per_seed_metrics machinery every
other paper5 figure uses, so this figure cannot silently drift from the
underlying data the way a hand-copied summary table could.

Two panels, different metrics (not directly comparable on one axis, so
not overlaid):
  (a) mean_reward_per_step paired diffs -- M2's two headline architecture
      comparisons plus M3's federation cost.
  (b) mean_reward_per_pending_request cost (baseline - disrupted,
      Section~\ref{sec:results-m4}'s volume-normalized primary metric) --
      dropout and churn at the most severe (60%-window) condition, the
      four rows spelled out in Section~\ref{sec:results-m4}D.

Usage:
    python3 experiments/plots/paper5_fig7_replication_forest.py \
        --out paper5/figures/fig7_replication_forest
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m2_correctness_metrics import bootstrap_ci, per_seed_metrics  # noqa: E402
from m4_correctness_metrics import baseline_eval_path, disrupted_eval_path, per_seed_metrics_normalized  # noqa: E402
from paper5_common import STATUS_COLORS, eval_omega_path, load_m2_campaign  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
COMMITTED = {
    "m2_results": ROOT / "experiments/results/m2_campaign/campaign_results.json",
    "m2_dir": ROOT / "experiments/results/m2_campaign",
    "m3_results": ROOT / "experiments/results/m3_campaign/campaign_results.json",
    "m3_dir": ROOT / "experiments/results/m3_campaign",
    "m4_results": ROOT / "experiments/results/m4_campaign/campaign_results.json",
    "m4_dir": ROOT / "experiments/results/m4_campaign",
}
FRESH = {
    "m2_results": ROOT / "experiments/results/fresh_seed_retrain/m2_campaign/campaign_results.json",
    "m2_dir": ROOT / "experiments/results/fresh_seed_retrain/m2_campaign",
    "m3_results": ROOT / "experiments/results/fresh_seed_retrain/m3_campaign/campaign_results.json",
    "m3_dir": ROOT / "experiments/results/fresh_seed_retrain/m3_campaign",
    "m4_results": ROOT / "experiments/results/fresh_seed_retrain/m4_campaign/campaign_results.json",
    "m4_dir": ROOT / "experiments/results/fresh_seed_retrain/m4_campaign",
}


def paired_stat(a_vals, b_vals):
    """a - b, mean + 95% bootstrap CI + Wilcoxon p (a vs b)."""
    a, b = np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)
    diff = a - b
    lo, hi = bootstrap_ci(diff)
    if np.any(diff != 0):
        _, p = stats.wilcoxon(a, b)
    else:
        p = float("nan")
    return float(diff.mean()), float(lo), float(hi), float(p)


def m2_reward_diff(sample, arm_a, arm_b):
    all_seeds, _ = load_m2_campaign(sample["m2_results"])
    a_vals = [per_seed_metrics(str(eval_omega_path(sample["m2_dir"], arm_a, s)))[0] for s in all_seeds]
    b_vals = [per_seed_metrics(str(eval_omega_path(sample["m2_dir"], arm_b, s)))[0] for s in all_seeds]
    return paired_stat(a_vals, b_vals)


def m3_federation_cost(sample):
    with open(sample["m3_results"]) as fh:
        m3 = json.load(fh)
    seeds = m3["seeds"]
    centralized = [per_seed_metrics(str(sample["m2_dir"] / "gat_ctde" / f"seed{s}" / "eval" / "omega_log.jsonl"))[0]
                   for s in seeds]
    federated = [per_seed_metrics(str(sample["m3_dir"] / "fl_gat_ctde_sigma0.0" / f"seed{s}" / "eval" / "omega_log.jsonl"))[0]
                 for s in seeds]
    return paired_stat(centralized, federated)


def m4_cost(sample, arm, kind, severity):
    with open(sample["m4_results"]) as fh:
        data = json.load(fh)
    cells = data["results"]
    seeds = sorted({c["seed"] for c in cells.values()
                     if c["arm"] == arm and c["kind"] == kind and c["severity"] == severity})
    severity_label = f"{kind}_sev{severity}"
    baseline_vals, disrupted_vals = [], []
    for s in seeds:
        d_path = disrupted_eval_path(str(sample["m4_dir"]), arm, severity_label, s)
        b_path = baseline_eval_path(arm, s, str(sample["m2_dir"]), str(sample["m3_dir"]))
        if not Path(d_path).exists() or not Path(b_path).exists():
            continue
        d_norm, _, _ = per_seed_metrics_normalized(d_path)
        b_norm, _, _ = per_seed_metrics_normalized(b_path)
        baseline_vals.append(b_norm)
        disrupted_vals.append(d_norm)
    return paired_stat(baseline_vals, disrupted_vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper5/figures/fig7_replication_forest")
    args = ap.parse_args()

    panel_a_rows = [
        ("GAT-CTDE $-$ Independent DQN", lambda s: m2_reward_diff(s, "gat_ctde", "independent_dqn")),
        ("GAT-CTDE $-$ Single-agent DQN", lambda s: m2_reward_diff(s, "gat_ctde", "single_agent_dqn")),
        ("Federation cost (centralized $-$ federated)", m3_federation_cost),
    ]
    panel_b_rows = [
        ("Dropout cost, 60% window: GAT-CTDE", lambda s: m4_cost(s, "gat_ctde", "dropout", 3)),
        ("Churn cost, 60% window: GAT-CTDE", lambda s: m4_cost(s, "gat_ctde", "churn", 3)),
        ("Churn cost, 60% window: Federated", lambda s: m4_cost(s, "fl_gat_ctde_sigma0.0", "churn", 3)),
        ("Churn cost, 60% window: Independent DQN", lambda s: m4_cost(s, "independent_dqn", "churn", 3)),
    ]

    def compute(rows):
        out = []
        for label, fn in rows:
            c = fn(COMMITTED)
            f = fn(FRESH)
            out.append((label, c, f))
            print(f"  {label}")
            print(f"    committed: mean={c[0]:.4f} [{c[1]:.4f},{c[2]:.4f}] p={c[3]:.4f}")
            print(f"    fresh:     mean={f[0]:.4f} [{f[1]:.4f},{f[2]:.4f}] p={f[3]:.4f}")
        return out

    print("=== panel (a): mean reward per step ===")
    a_data = compute(panel_a_rows)
    print("=== panel (b): mean reward per pending request (M4 cost) ===")
    b_data = compute(panel_b_rows)

    committed_color, fresh_color = "#2a78d6", "#eb6834"
    flip_row_label = "Churn cost, 60% window: Independent DQN"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.9),
                                    gridspec_kw={"width_ratios": [len(a_data), len(b_data)]})

    def draw_panel(ax, data, xlabel, title):
        n = len(data)
        for i, (label, c, f) in enumerate(data):
            y = n - 1 - i
            highlight = label == flip_row_label
            if highlight:
                ax.axhspan(y - 0.42, y + 0.42, color="#fdeeee", zorder=0)
            ax.errorbar(c[0], y + 0.14, xerr=[[c[0] - c[1]], [c[2] - c[0]]],
                        fmt="o", color=committed_color, markersize=5.5, capsize=3,
                        linewidth=1.2, zorder=3)
            ax.errorbar(f[0], y - 0.14, xerr=[[f[0] - f[1]], [f[2] - f[0]]],
                        fmt="D", color=fresh_color, markersize=5, capsize=3,
                        linewidth=1.2, zorder=3)
        ax.axvline(0, color="#898781", linewidth=0.8, linestyle="-", zorder=1)
        ax.set_yticks(range(n))
        ax.set_yticklabels([data[n - 1 - i][0] for i in range(n)], fontsize=6.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylim(-0.6, n - 0.4)

    draw_panel(ax1, a_data, "$\\Delta$ mean reward per step",
               "(a) M2/M3 headline reward comparisons")
    draw_panel(ax2, b_data, "Disruption cost (baseline $-$ disrupted, normalized)",
               "(b) M4 disruption cost, most severe (60%) window")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=committed_color, linestyle="none", markersize=6,
               label="Committed sample (seeds 900–929/909)"),
        Line2D([0], [0], marker="D", color=fresh_color, linestyle="none", markersize=5.5,
               label="Independent replication (seeds 1000–1029/1009)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)

    fig.suptitle("Cross-sample replication: does each effect hold up on independent seeds?",
                  y=1.06, fontsize=9)
    fig.subplots_adjust(wspace=1.35, bottom=0.28, top=0.82)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig7] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
