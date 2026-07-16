#!/usr/bin/env python3
"""Figure 7: QoE-reward decomposition (alpha*MOS / beta*cost / gamma*SLA_viol
components, eq.9) for the qoe arms (dqn_qoe, a2c_qoe) over training episodes
-- reads the OFFLINE training omega logs (where these components are the
actual optimized reward, not passive diagnostics) since that's where the
learning trajectory of each component is meaningful.

Usage:
    python3 experiments/plots/fig7_qoe_decomposition.py \
        --offline-root experiments/results/offline --seeds 256 257 258 \
        --out experiments/plots/out/fig7_qoe_decomposition
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, read_omega_log  # noqa: E402

QOE_ARMS = {"dqn_qoe": "dqn", "a2c_qoe": "a2c"}
# eq.9 weights, from experiments/configs/saclb_offline_campaign.yaml's qoe.reward section
ALPHA, BETA, GAMMA = 1.0, 0.2, 0.5


def load_components(omega_path: Path) -> dict:
    """Per-episode mean of alpha*mean_mos, beta*cost, gamma*sla_viol,
    aggregated from step-level rows (rollups don't carry these fields)."""
    by_episode = {}
    for row in read_omega_log(omega_path):
        if row.step < 1:
            continue
        mos = row.evidence.get("mean_mos")
        cost = row.evidence.get("cost")
        viol = row.evidence.get("sla_viol")
        if mos is None or cost is None or viol is None:
            continue
        by_episode.setdefault(row.episode, {"mos": [], "cost": [], "viol": []})
        by_episode[row.episode]["mos"].append(mos)
        by_episode[row.episode]["cost"].append(cost)
        by_episode[row.episode]["viol"].append(viol)

    n = max(by_episode) if by_episode else 0
    out = {"alpha_mos": np.full(n, np.nan), "beta_cost": np.full(n, np.nan), "gamma_viol": np.full(n, np.nan)}
    for ep, vals in by_episode.items():
        out["alpha_mos"][ep - 1] = ALPHA * np.mean(vals["mos"])
        out["beta_cost"][ep - 1] = BETA * np.mean(vals["cost"])
        out["gamma_viol"][ep - 1] = GAMMA * np.mean(vals["viol"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline-root", default="experiments/results/offline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[256, 257, 258])
    ap.add_argument("--out", default="experiments/plots/out/fig7_qoe_decomposition")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(QOE_ARMS), 1, sharex=True, figsize=(3.5, 4.5))
    component_style = {
        "alpha_mos": {"color": "#2a78d6", "label": r"$\alpha \cdot$MOS"},
        "beta_cost": {"color": "#eda100", "label": r"$\beta \cdot$cost"},
        "gamma_viol": {"color": "#e34948", "label": r"$\gamma \cdot$SLA_viol"},
    }

    for ax, (arm, algo) in zip(axes, QOE_ARMS.items()):
        per_seed = []
        for seed in args.seeds:
            omega_path = Path(args.offline_root) / "qoe" / f"seed{seed}" / algo / "offline_closed_loop" / "rep_0" / "omega_log.jsonl"
            if omega_path.exists():
                per_seed.append(load_components(omega_path))

        if not per_seed:
            print(f"[fig7] WARNING: no data for {arm}", file=sys.stderr)
            continue

        min_len = min(len(c["alpha_mos"]) for c in per_seed)
        for comp in ["alpha_mos", "beta_cost", "gamma_viol"]:
            stacked = np.stack([c[comp][:min_len] for c in per_seed])
            mean = np.nanmean(stacked, axis=0)
            episodes = np.arange(1, min_len + 1)
            style = component_style[comp]
            ax.plot(episodes, mean, color=style["color"], label=style["label"], linewidth=1.0)

        ax.set_title(ARM_STYLE[arm]["label"], fontsize=8)
        ax.set_ylabel("Component value")

    axes[-1].set_xlabel("Training episode")
    axes[0].legend(loc="best", frameon=False, fontsize=6)
    fig.suptitle("QoE-reward (eq.9) decomposition", fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[fig7] wrote {out_path}.pdf / .png")


if __name__ == "__main__":
    main()
