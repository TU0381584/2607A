#!/usr/bin/env python3
"""Live UE-count fix/generalisation comparison: real live campaigns,
same seed-900 checkpoint lineage, real per-slice UDP traffic throughout.
Default 3 conditions -- (1) the original checkpoint at 3 UEs (healthy
baseline), (2) the same original checkpoint at 6 UEs (complete collapse,
already reported in Fig.~\ref{fig:live-3v6}), (3) a checkpoint retrained
against a recalibrated offline simulator (RealisticServedKpmSource, M34)
at 6 UEs -- with three more optional conditions (recalibrated @ 3 UEs,
and both checkpoints @ an M37 held-out UE count) for M38's full 2x3
factorial once that live data exists (`docs/PAPER5_M37_M38_scoping.md`).
Reads every number directly from each run's own raw omega_log.jsonl
(evidence.rollup == true, one per episode) -- not retyped. 95% bootstrap
percentile CIs (10,000 resamples, this project's standard) over the real
per-episode values, reusing m2_correctness_metrics.bootstrap_ci.

Two panels:
(a) Blocks per episode, all episodes per condition, mean +/- range
    with individual episodes jittered behind -- the original-6UE
    condition's flat line at zero IS the collapse finding, not a
    plotting choice.
(b) Mean reward per step, mean + 95% CI with individual episodes
    jittered behind.

Usage (default 3-condition Fig. 8, unchanged from before):
    python3 experiments/plots/paper5_fig_live_recalibrated_fix.py \
        --orig-3ue-jsonl experiments/results/live/m31_highconf/3ue_20ep_omega_log.jsonl \
        --orig-6ue-jsonl experiments/results/live/m31_highconf/6ue_20ep_omega_log.jsonl \
        --fixed-6ue-jsonl experiments/results/live/m34_realistic_retrain_check/6ue_20ep_omega_log.jsonl \
        --out Papers_4-5/Paper_5/WPC/figures/fig8_live_recalibrated_fix

Usage (M38's full factorial, once the extra live data exists):
    python3 experiments/plots/paper5_fig_live_recalibrated_fix.py \
        --orig-3ue-jsonl ... --orig-6ue-jsonl ... --fixed-6ue-jsonl ... \
        --fixed-3ue-jsonl experiments/results/m38_live/recal_3ue_omega_log.jsonl \
        --orig-heldout-jsonl experiments/results/m38_live/orig_heldout_omega_log.jsonl \
        --fixed-heldout-jsonl experiments/results/m38_live/recal_heldout_omega_log.jsonl \
        --heldout-label "5 UEs" \
        --out Papers_4-5/Paper_5/WPC/figures/fig9_m38_full_factorial
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paper5_common import IEEE_COLUMN_WIDTH_IN  # noqa: E402,F401
from m2_correctness_metrics import bootstrap_ci  # noqa: E402

BASE_COND_STYLE = {
    "Original,\n3 UEs": {"color": "#2a78d6", "marker": "o"},
    "Original,\n6 UEs": {"color": "#eb6834", "marker": "^"},
    "Recalibrated,\n6 UEs": {"color": "#1a9e8f", "marker": "s"},
    "Recalibrated,\n3 UEs": {"color": "#5fc9ba", "marker": "D"},
}


def load_episode_rollups(path: str):
    rollups = []
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", {})
            if ev.get("rollup"):
                rollups.append(ev)
    return rollups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-3ue-jsonl", default="experiments/results/live/m31_highconf/3ue_20ep_omega_log.jsonl")
    ap.add_argument("--orig-6ue-jsonl", default="experiments/results/live/m31_highconf/6ue_20ep_omega_log.jsonl")
    ap.add_argument("--fixed-6ue-jsonl",
                     default="experiments/results/live/m34_realistic_retrain_check/6ue_20ep_omega_log.jsonl")
    ap.add_argument("--fixed-3ue-jsonl", default=None, help="M38 gap: recalibrated checkpoint @ 3 UEs")
    ap.add_argument("--orig-heldout-jsonl", default=None, help="M38 gap: original checkpoint @ M37 held-out UE count")
    ap.add_argument("--fixed-heldout-jsonl", default=None,
                     help="M38 gap: recalibrated checkpoint @ M37 held-out UE count")
    ap.add_argument("--heldout-label", default="held-out UEs", help='e.g. "5 UEs" -- used in the two heldout labels')
    ap.add_argument("--out", default="Papers_4-5/Paper_5/WPC/figures/fig8_live_recalibrated_fix")
    args = ap.parse_args()

    cond_style = dict(BASE_COND_STYLE)
    orig_heldout_label = f"Original,\n{args.heldout_label}"
    fixed_heldout_label = f"Recalibrated,\n{args.heldout_label}"
    cond_style[orig_heldout_label] = {"color": "#b3462c", "marker": "^"}
    cond_style[fixed_heldout_label] = {"color": "#0f6b60", "marker": "s"}

    runs = {
        "Original,\n3 UEs": load_episode_rollups(args.orig_3ue_jsonl),
        "Original,\n6 UEs": load_episode_rollups(args.orig_6ue_jsonl),
        "Recalibrated,\n6 UEs": load_episode_rollups(args.fixed_6ue_jsonl),
    }
    if args.fixed_3ue_jsonl:
        runs["Recalibrated,\n3 UEs"] = load_episode_rollups(args.fixed_3ue_jsonl)
    if args.orig_heldout_jsonl:
        runs[orig_heldout_label] = load_episode_rollups(args.orig_heldout_jsonl)
    if args.fixed_heldout_jsonl:
        runs[fixed_heldout_label] = load_episode_rollups(args.fixed_heldout_jsonl)
    labels = list(runs.keys())
    for label, eps in runs.items():
        rewards = np.array([e["episode_mean_reward"] for e in eps])
        lo, hi = bootstrap_ci(rewards)
        blocks = [sum(e["episode_block_by_slice"].values()) for e in eps]
        n_zero = sum(1 for b in blocks if b == 0)
        n_mmtc_only = sum(1 for e in eps if sum(e["episode_block_by_slice"].values()) > 0
                           and set(e["episode_block_by_slice"].keys()) == {"mmtc"})
        print(f"[live-fix] {label.strip()}: n={len(eps)} reward mean={rewards.mean():.3f} "
              f"CI=[{lo:.3f},{hi:.3f}] zero-block={n_zero}/{len(eps)} mmtc-only={n_mmtc_only}/{len(eps)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_COLUMN_WIDTH_IN * (1.9 if len(labels) <= 3 else 2.8),
                                                   IEEE_COLUMN_WIDTH_IN * 0.95))

    xb = np.arange(len(labels))

    # ---- (a) blocks per episode, mean +/- range, episodes jittered behind ----
    for i, label in enumerate(labels):
        style = cond_style[label]
        blocks = np.array([sum(e["episode_block_by_slice"].values()) for e in runs[label]])
        jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(blocks))
        ax1.scatter(np.full(len(blocks), xb[i]) + jitter, blocks, color=style["color"],
                    alpha=0.35, s=12, zorder=2, marker=style["marker"])
        mean = blocks.mean()
        ax1.bar(xb[i], mean, width=0.5, color=style["color"], alpha=0.85, edgecolor="white",
                linewidth=0.5, zorder=1)
        ax1.errorbar(xb[i], mean, yerr=[[mean - blocks.min()], [blocks.max() - mean]], fmt="none",
                     ecolor="#0b0b0b", capsize=3, linewidth=1.2, zorder=5)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Blocks per episode\n(mean, range, n=20 episodes)")
    ax1.set_title("(a)", loc="left")

    # ---- (b) mean reward per step, mean + 95% CI, episodes jittered behind ----
    for i, label in enumerate(labels):
        style = cond_style[label]
        rewards = np.array([e["episode_mean_reward"] for e in runs[label]])
        lo, hi = bootstrap_ci(rewards)
        mean = rewards.mean()
        jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(rewards))
        ax2.scatter(np.full(len(rewards), xb[i]) + jitter, rewards, color=style["color"],
                    alpha=0.35, s=12, zorder=2, marker=style["marker"])
        ax2.bar(xb[i], mean, width=0.5, color=style["color"], alpha=0.85, edgecolor="white",
                linewidth=0.5, zorder=1)
        ax2.errorbar(xb[i], mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                     ecolor="#0b0b0b", capsize=3, linewidth=1.2, zorder=5)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Mean reward per step\n(mean $\\pm$ 95% CI, n=20 episodes)")
    ax2.set_title("(b)", loc="left")

    fig.subplots_adjust(wspace=0.55, bottom=0.22)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[live-fix] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
