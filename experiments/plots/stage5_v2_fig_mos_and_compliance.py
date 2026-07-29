#!/usr/bin/env python3
"""Stage 5 v2 trial figures: DQN vs baseline, under the corrected MOS/SLA
calibration (docs/STAGE5_recalibration.md). Mirrors fig6_inferred_mos.py's
exact style/data-reading conventions but restricted to the arms this
trial actually has data for (baseline, dqn_sla, dqn_qoe -- static_at_cap
omitted here since the user asked for "DQN vs baseline" specifically),
with an explicit n=2-episodes/seed=950/directional-only annotation baked
into the title -- this is NOT the statistically powered re-run.

Usage:
    python3 experiments/plots/stage5_v2_fig_mos_and_compliance.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STYLE, SLICE_ORDER, SLICE_STYLE, arm_run_dir, read_omega_log  # noqa: E402

LIVE_ROOT = "experiments/results/live_campaign_v2"
SEEDS = [950, 951, 952]
ARMS = ["baseline", "dqn_sla", "dqn_qoe", "static_at_cap"]
ARM_REWARD_MODE = {"baseline": "sla", "dqn_sla": "sla", "dqn_qoe": "qoe", "static_at_cap": "sla"}
OUT = "experiments/plots/out/stage5_v2_campaign_fig_mos_and_compliance"


def pooled_mos_and_compliance(omega_paths) -> tuple:
    mos_vals = {s: [] for s in SLICE_ORDER}
    compliant = {s: 0 for s in SLICE_ORDER}
    total = {s: 0 for s in SLICE_ORDER}
    for omega_path in omega_paths:
        if not omega_path.exists():
            continue
        for row in read_omega_log(omega_path):
            if row.step < 1:
                continue
            mos_by_slice = row.evidence.get("mos_by_slice", {})
            for s in SLICE_ORDER:
                if s in mos_by_slice:
                    mos_vals[s].append(mos_by_slice[s])
            c = row.evidence.get("per_slice_compliant", {})
            for s in SLICE_ORDER:
                if s in c:
                    total[s] += 1
                    if c[s]:
                        compliant[s] += 1
    mos = {s: (float(np.mean(mos_vals[s])) if mos_vals[s] else float("nan")) for s in SLICE_ORDER}
    compliance = {s: (100.0 * compliant[s] / total[s] if total[s] else float("nan")) for s in SLICE_ORDER}
    return mos, compliance


def main() -> None:
    mos_by_arm, compliance_by_arm = {}, {}
    for arm in ARMS:
        mode = ARM_REWARD_MODE[arm]
        omega_paths = [arm_run_dir(LIVE_ROOT, arm, mode, seed) / "omega_log.jsonl" for seed in SEEDS]
        mos_by_arm[arm], compliance_by_arm[arm] = pooled_mos_and_compliance(omega_paths)

    fig, (ax_c, ax_m) = plt.subplots(2, 1, figsize=(3.5, 3.5 * 1.5))
    n_slices = len(SLICE_ORDER)
    bar_width = 0.8 / n_slices
    x = np.arange(len(ARMS))

    for i, slice_id in enumerate(SLICE_ORDER):
        style = SLICE_STYLE[slice_id]
        offset = (i - (n_slices - 1) / 2) * bar_width
        c_vals = [compliance_by_arm[arm][slice_id] for arm in ARMS]
        ax_c.bar(x + offset, c_vals, bar_width, color=style["color"], hatch=style["hatch"],
                 label=style["label"], edgecolor="white", linewidth=0.3)
        m_vals = [mos_by_arm[arm][slice_id] for arm in ARMS]
        ax_m.bar(x + offset, m_vals, bar_width, color=style["color"], hatch=style["hatch"],
                 label=style["label"], edgecolor="white", linewidth=0.3)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([ARM_STYLE[a]["label"] for a in ARMS], rotation=20, ha="right")
    ax_c.set_ylabel("Per-step SLA\ncompliance (%)")
    ax_c.set_ylim(0, 105)
    ax_c.set_title("Stage 5 v2 campaign (n=6 episodes/arm, 3 seeds x 2 episodes)", fontsize=7)
    ax_c.legend(loc="upper left", frameon=False, ncol=1, fontsize=5.5)

    ax_m.set_xticks(x)
    ax_m.set_xticklabels([ARM_STYLE[a]["label"] for a in ARMS], rotation=20, ha="right")
    ax_m.set_ylabel("Inferred MOS (1-5)\ncorrected calibration")
    ax_m.set_ylim(1, 5)

    fig.tight_layout()
    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    print(f"wrote {out_path}.pdf / .png")
    print("compliance:", compliance_by_arm)
    print("mos:", mos_by_arm)


if __name__ == "__main__":
    main()
