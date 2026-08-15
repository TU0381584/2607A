#!/usr/bin/env python3
"""Figure 3: M2 30-seed campaign, two panels sharing one arm axis --
(a) sla_compliance_all_slices (paper #4's own established metric) and
(b) mean_reward_per_step (the correctness-aware metric added after
finding (a) structurally rewards the "accept everything" collapse over
genuinely-correct differentiated shedding -- docs/PAPER5_M2_gat_ctde.md
section 12). Both panels shown together deliberately: the paper's own
honest finding is that they disagree, so plotting only one would hide
that disagreement from the reader.

Reuses m2_correctness_metrics.per_seed_metrics (imported, not
reimplemented) for mean_reward_per_step, and the same bootstrap-CI
methodology as m2_campaign_analysis.py.

Usage:
    python3 experiments/plots/paper5_fig3_m2_campaign.py \
        --results experiments/results/m2_campaign/campaign_results.json \
        --campaign-dir experiments/results/m2_campaign \
        --out paper5/figures/fig3_m2_campaign
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m2_correctness_metrics import per_seed_metrics  # noqa: E402
from paper5_common import M2_ARM_ORDER, M2_ARM_STYLE, bootstrap_ci, eval_omega_path, load_m2_campaign  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results/m2_campaign/campaign_results.json")
    ap.add_argument("--campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--out", default="paper5/figures/fig3_m2_campaign")
    args = ap.parse_args()

    all_seeds, results = load_m2_campaign(args.results)

    compliance_by_arm, reward_by_arm = {}, {}
    for arm in M2_ARM_ORDER:
        by_seed = results[arm]
        compliance_by_arm[arm] = [by_seed[str(s)]["sla_compliance_all_slices"] for s in all_seeds if str(s) in by_seed]
        rewards = []
        for s in all_seeds:
            path = eval_omega_path(args.campaign_dir, arm, s)
            if path.exists():
                mrps, _, _ = per_seed_metrics(str(path))
                rewards.append(mrps)
        reward_by_arm[arm] = rewards

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    x = np.arange(len(M2_ARM_ORDER))
    labels = [M2_ARM_STYLE[a]["label"] for a in M2_ARM_ORDER]
    colors = [M2_ARM_STYLE[a]["color"] for a in M2_ARM_ORDER]

    for ax, data_by_arm, ylabel, title, ylim in [
        (ax1, compliance_by_arm, "SLA compliance\n(all slices, fraction)", "(a) Compliance (n=30/arm)", (0, 0.6)),
        (ax2, reward_by_arm, "Mean reward\nper step", "(b) Reward (n=30/arm)", None),
    ]:
        means, los, his = [], [], []
        for arm in M2_ARM_ORDER:
            v = np.array(data_by_arm[arm])
            lo, hi = bootstrap_ci(v)
            means.append(v.mean())
            los.append(v.mean() - lo)
            his.append(hi - v.mean())
        ax.bar(x, means, yerr=[los, his], color=colors, width=0.6, capsize=3,
               edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)

    fig.suptitle("M2 campaign: compliance vs. correctness-aware reward, 30 seeds/arm", y=1.04, fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[paper5:fig3] wrote {out_path}.pdf / .png")
    for arm in M2_ARM_ORDER:
        c = np.array(compliance_by_arm[arm])
        r = np.array(reward_by_arm[arm])
        print(f"  {arm}: compliance mean={c.mean():.3f} (n={len(c)}), reward mean={r.mean():.3f} (n={len(r)})")


if __name__ == "__main__":
    main()
