#!/usr/bin/env python3
"""Figure 5: M4 disruption-resilience campaign -- three panels sharing a
severity x-axis: (a) gNB dropout's normalized-reward cost vs. severity,
all four arms; (b) agent churn's cost, the three genuinely multi-agent
arms (single_agent_dqn has no separable agent to churn); (c) demand
spike's block_precision vs. severity, all four arms, shown flat/near-
ceiling to visually contrast with (a)/(b)'s accelerating curves -- the
paper's own point that spike is a "genuinely different, cleaner story"
than dropout/churn (no threshold on the metric that matters).

Uses the PRIMARY, volume-normalized reward metric for (a)/(b)
(m4_correctness_metrics.per_seed_metrics_normalized) for the same reason
docs/PAPER5_M4_disruption.md gives: raw mean_reward_per_step is not
volume-comparable once a disruption changes request volume (spike does;
dropout/churn don't, but the normalized metric is used uniformly across
all three so the reader compares like with like panel to panel).

Usage:
    python3 experiments/plots/paper5_fig5_m4_disruption.py \
        --m4-results experiments/results/m4_campaign/campaign_results.json \
        --m4-campaign-dir experiments/results/m4_campaign \
        --m2-campaign-dir experiments/results/m2_campaign \
        --m3-campaign-dir experiments/results/m3_campaign \
        --out Papers_4-5/Paper_5/IEEE_Access/figures/fig5_m4_disruption
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m4_correctness_metrics import baseline_eval_path, disrupted_eval_path, per_seed_metrics_normalized  # noqa: E402
from m2_correctness_metrics import per_seed_metrics  # noqa: E402
from paper5_common import M4_ARM_ORDER, M4_ARM_STYLE, bootstrap_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4-results", default="experiments/results/m4_campaign/campaign_results.json")
    ap.add_argument("--m4-campaign-dir", default="experiments/results/m4_campaign")
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--m3-campaign-dir", default="experiments/results/m3_campaign")
    ap.add_argument("--out", default="Papers_4-5/Paper_5/IEEE_Access/figures/fig5_m4_disruption")
    ap.add_argument("--no-titles", action="store_true",
                     help="Bare (a)/(b)/(c) panel tags, no descriptive words, no suptitle "
                          "(WPC copy: caption carries the description instead).")
    args = ap.parse_args()

    with open(args.m4_results) as fh:
        data = json.load(fh)
    cells = data["results"]
    conditions = {}
    for key, cell in cells.items():
        conditions.setdefault((cell["arm"], cell["kind"], cell["severity"]), []).append(cell["seed"])

    def cost_series(kind, arms):
        """Returns {arm: (means[3], los[3], his[3])} for severities 1/2/3."""
        out = {}
        for arm in arms:
            means, los, his = [], [], []
            for severity in (1, 2, 3):
                seeds = sorted(conditions.get((arm, kind, severity), []))
                if not seeds:
                    means.append(np.nan); los.append(0); his.append(0)
                    continue
                diffs = []
                for seed in seeds:
                    severity_label = f"{kind}_sev{severity}"
                    d_path = disrupted_eval_path(args.m4_campaign_dir, arm, severity_label, seed)
                    b_path = baseline_eval_path(arm, seed, args.m2_campaign_dir, args.m3_campaign_dir)
                    if not Path(d_path).exists() or not Path(b_path).exists():
                        continue
                    d_norm, _, _ = per_seed_metrics_normalized(d_path)
                    b_norm, _, _ = per_seed_metrics_normalized(b_path)
                    diffs.append(b_norm - d_norm)
                v = np.array(diffs)
                lo, hi = bootstrap_ci(v)
                means.append(v.mean()); los.append(v.mean() - lo); his.append(hi - v.mean())
            out[arm] = (means, los, his)
        return out

    def precision_series(kind, arms):
        out = {}
        for arm in arms:
            means, los, his = [], [], []
            for severity in (1, 2, 3):
                seeds = sorted(conditions.get((arm, kind, severity), []))
                severity_label = f"{kind}_sev{severity}"
                precisions = []
                for seed in seeds:
                    d_path = disrupted_eval_path(args.m4_campaign_dir, arm, severity_label, seed)
                    if not Path(d_path).exists():
                        continue
                    _, mmtc_b, total_b = per_seed_metrics(str(d_path))
                    if total_b > 0:
                        precisions.append(mmtc_b / total_b)
                if not precisions:
                    means.append(np.nan); los.append(0); his.append(0)
                    continue
                v = np.array(precisions)
                lo, hi = bootstrap_ci(v)
                means.append(v.mean()); los.append(v.mean() - lo); his.append(hi - v.mean())
            out[arm] = (means, los, his)
        return out

    dropout = cost_series("dropout", M4_ARM_ORDER)
    churn_arms = [a for a in M4_ARM_ORDER if a != "single_agent_dqn"]
    churn = cost_series("churn", churn_arms)
    spike_precision = precision_series("spike", M4_ARM_ORDER)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.2, 2.7))
    x = np.array([1, 2, 3])

    for arm in M4_ARM_ORDER:
        style = M4_ARM_STYLE[arm]
        means, los, his = dropout[arm]
        ax1.errorbar(x, means, yerr=[los, his], color=style["color"], marker=style["marker"],
                     markersize=5, linewidth=1.4, capsize=3, label=style["label"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(["10%", "30%", "60%"])
    ax1.set_xlabel("Dropout window (% of episode)")
    ax1.set_ylabel("Normalised reward cost\n(baseline $-$ disrupted)")
    ax1.set_title("(a)" if args.no_titles else "(a) gNB dropout", loc="left")
    ax1.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=0)

    for arm in churn_arms:
        style = M4_ARM_STYLE[arm]
        means, los, his = churn[arm]
        ax2.errorbar(x, means, yerr=[los, his], color=style["color"], marker=style["marker"],
                     markersize=5, linewidth=1.4, capsize=3, label=style["label"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["10%", "30%", "60%"])
    ax2.set_xlabel("Churn window (% of episode)")
    ax2.set_ylabel("Normalised reward cost\n(baseline $-$ disrupted)")
    ax2.set_title("(b)" if args.no_titles else "(b) Agent churn", loc="left")
    ax2.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=0)

    for arm in M4_ARM_ORDER:
        style = M4_ARM_STYLE[arm]
        means, los, his = spike_precision[arm]
        ax3.errorbar(x, means, yerr=[los, his], color=style["color"], marker=style["marker"],
                     markersize=5, linewidth=1.4, capsize=3, label=style["label"])
    ax3.set_xticks(x)
    ax3.set_xticklabels(["2x", "4x", "8x"])
    ax3.set_xlabel("Spike multiplier")
    ax3.set_ylabel("Block precision\n(fraction targeting mMTC)")
    ax3.set_ylim(-0.05, 1.08)
    ax3.set_title("(c)" if args.no_titles else "(c) Demand spike", loc="left")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)

    if not args.no_titles:
        fig.suptitle("M4: disruption cost vs. severity, 10 seeds/arm", y=1.05, fontsize=9)
    fig.subplots_adjust(bottom=0.05, wspace=0.45)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig5] wrote {out_path}.pdf / .png")
    for panel_name, series in [("dropout", dropout), ("churn", churn)]:
        for arm, (means, los, his) in series.items():
            print(f"  {panel_name}/{arm}: sev1/2/3 cost = {means}")


if __name__ == "__main__":
    main()
