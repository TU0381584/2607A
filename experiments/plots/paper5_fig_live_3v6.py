#!/usr/bin/env python3
"""Live single-gNB load comparison: same seed-900 single-agent-DQN
checkpoint, 3 UEs vs 6 UEs, both against the real OAI rig with real
per-slice UDP traffic, 20 episodes each (high-confidence campaign,
not the earlier 2-episode pilot). Reads each run's own rollup records
directly from its raw omega_log.jsonl (evidence.rollup == true, one per
episode) -- not retyped. 95% bootstrap percentile CIs (10,000 resamples,
this project's standard), computed across the real 20 episodes per
condition, reusing m2_correctness_metrics.bootstrap_ci rather than
reimplementing it.

Two panels:
(a) SLA margin, percent change from 3 UEs to 6 UEs, one bar per slice,
    with a bootstrap 95% CI on the underlying per-episode difference.
(b) Mean reward per step, 3 UEs vs 6 UEs, mean + 95% CI, individual
    episodes shown behind it.

Usage:
    python3 experiments/plots/paper5_fig_live_3v6.py \
        --run3-jsonl experiments/results/live/m31_highconf/3ue_20ep_omega_log.jsonl \
        --run6-jsonl experiments/results/live/m31_highconf/6ue_20ep_omega_log.jsonl \
        --out paper5_wpc/figures/fig_live_3v6
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from paper5_common import IEEE_COLUMN_WIDTH_IN, STATUS_COLORS  # noqa: E402,F401
from m2_correctness_metrics import bootstrap_ci  # noqa: E402

SLICES = ["urllc", "embb", "mmtc"]
COND_STYLE = {
    "3 UEs": {"color": "#2a78d6", "marker": "o"},
    "6 UEs": {"color": "#eb6834", "marker": "^"},
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
    ap.add_argument("--run3-jsonl", default="experiments/results/live/m31_highconf/3ue_20ep_omega_log.jsonl")
    ap.add_argument("--run6-jsonl", default="experiments/results/live/m31_highconf/6ue_20ep_omega_log.jsonl")
    ap.add_argument("--out", default="Paper5/WPC/figures/fig_live_3v6")
    args = ap.parse_args()

    runs = {
        "3 UEs": load_episode_rollups(args.run3_jsonl),
        "6 UEs": load_episode_rollups(args.run6_jsonl),
    }
    for label, eps in runs.items():
        print(f"[live-3v6] {label}: {len(eps)} episodes loaded from raw log")
        rewards = np.array([e["episode_mean_reward"] for e in eps])
        lo, hi = bootstrap_ci(rewards)
        print(f"  reward: mean={rewards.mean():.3f} 95% CI [{lo:.3f}, {hi:.3f}]")
        blocks_total = [sum(e["episode_block_by_slice"].values()) for e in eps]
        n_zero = sum(1 for b in blocks_total if b == 0)
        n_mmtc_only = sum(1 for e in eps if sum(e["episode_block_by_slice"].values()) > 0
                           and set(e["episode_block_by_slice"].keys()) == {"mmtc"})
        print(f"  blocks: {n_zero}/{len(eps)} episodes had zero blocks, "
              f"{n_mmtc_only}/{len(eps)} were mmtc-only")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_COLUMN_WIDTH_IN * 1.7, IEEE_COLUMN_WIDTH_IN * 0.9))

    # ---- (a) SLA margin, percent change 3 UEs -> 6 UEs, one bar per slice, with bootstrap CI ----
    x = np.arange(len(SLICES))
    pct_change, pct_lo, pct_hi = [], [], []
    for s in SLICES:
        v3 = np.array([e["episode_sla_margin_by_slice"][s] for e in runs["3 UEs"]])
        v6 = np.array([e["episode_sla_margin_by_slice"][s] for e in runs["6 UEs"]])
        mean3, mean6 = v3.mean(), v6.mean()
        pct = 100.0 * (abs(mean6) - abs(mean3)) / abs(mean3)
        pct_change.append(pct)
        # bootstrap the % change directly: resample each condition's episodes independently
        rng = np.random.RandomState(0)
        boot = []
        for _ in range(10000):
            b3 = rng.choice(v3, size=len(v3), replace=True).mean()
            b6 = rng.choice(v6, size=len(v6), replace=True).mean()
            boot.append(100.0 * (abs(b6) - abs(b3)) / abs(b3))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pct_lo.append(pct - lo)
        pct_hi.append(hi - pct)

    colors = [STATUS_COLORS["good"] if v <= 0 else STATUS_COLORS["critical"] for v in pct_change]
    ax1.bar(x, pct_change, width=0.5, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
            yerr=[pct_lo, pct_hi], capsize=3, ecolor="#0b0b0b", error_kw={"linewidth": 1.0})
    for xi, v in zip(x, pct_change):
        ax1.annotate(f"{v:+.0f}%", (xi, v), xytext=(0, 12 if v >= 0 else -14),
                     textcoords="offset points", ha="center", fontsize=7)
    ax1.axhline(0, color="#898781", linewidth=0.8, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(SLICES)
    ax1.set_ylabel("SLA margin change,\n3 UEs $\\to$ 6 UEs (%, 95% CI)")
    ax1.set_title("(a)", loc="left")

    # ---- (b) mean reward per step, 3 UEs vs 6 UEs, mean + 95% CI, episodes behind ----
    labels = list(runs.keys())
    xb = np.arange(len(labels))
    for i, label in enumerate(labels):
        style = COND_STYLE[label]
        rewards = np.array([e["episode_mean_reward"] for e in runs[label]])
        lo, hi = bootstrap_ci(rewards)
        mean = rewards.mean()
        jitter = np.random.RandomState(0).uniform(-0.12, 0.12, size=len(rewards))
        ax2.scatter(np.full(len(rewards), xb[i]) + jitter, rewards, color=style["color"],
                    alpha=0.35, s=12, zorder=2, marker=style["marker"])
        ax2.bar(xb[i], mean, width=0.5, color=style["color"], alpha=0.85, edgecolor="white", linewidth=0.5, zorder=1)
        ax2.errorbar(xb[i], mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                     ecolor="#0b0b0b", capsize=3, linewidth=1.2, zorder=5)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Mean reward per step\n(mean $\\pm$ 95% CI, n=20 episodes)")
    ax2.set_title("(b)", loc="left")

    fig.subplots_adjust(wspace=0.55, bottom=0.18)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    print(f"[live-3v6] wrote {out_path}.pdf / .png")
    for s, v, lo_e, hi_e in zip(SLICES, pct_change, pct_lo, pct_hi):
        print(f"  {s}: change={v:+.1f}% [{v-lo_e:+.1f}%, {v+hi_e:+.1f}%]")


if __name__ == "__main__":
    main()
