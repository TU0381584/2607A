#!/usr/bin/env python3
"""Stage 5 v2 campaign metrics: 4 arms, fully powered at n=21 episodes/arm
each (experiments/results/live_campaign_v2), under the corrected MOS/SLA
calibration + retrained checkpoints (docs/STAGE5_recalibration.md).
Reuses metrics_stage2.py's exact helper functions (weighted_u,
percentile_stats, fisher_exact_vs_baseline) -- nothing recomputed with
different logic than the rest of this project's metrics pipeline.

Reaches n=21/arm via the "top-up" pass (experiments/scripts/
run_stage5_v2_topup.sh, 2026-07-30): all 4 arms now share the SAME
seed set -- 950/951/952 at the 3h campaign's 2-episode depth, plus
953/954/955 at the full 5-episode protocol -- 6 + 15 = 21 each. This is
the deferred "full 3-seed x 5-episode x 4-arm" live re-evaluation user's
own words: "we'll do the 6 hour one later" (docs/STAGE5_recalibration.md
section 6).

Usage:
    python3 experiments/scripts/metrics_stage5_v2.py --out docs/stage5_v2_campaign_metrics_raw.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_stage2 import (  # noqa: E402
    SLICE_ORDER, LIVE_PRIORITY_WEIGHT, LIVE_VIOLATION_PENALTY,
    weighted_u, percentile_stats, fisher_exact_vs_baseline, _read_omega,
)

ARM_REWARD_MODE = {"baseline": "sla", "dqn_sla": "sla", "dqn_qoe": "qoe", "static_at_cap": "sla"}
# All 4 arms now share the same, consistent seed set (n=21 each) after
# the 2026-07-30 top-up pass -- see module docstring.
ARM_SEEDS = {
    "baseline": [950, 951, 952, 953, 954, 955],
    "dqn_qoe": [950, 951, 952, 953, 954, 955],
    "dqn_sla": [950, 951, 952, 953, 954, 955],
    "static_at_cap": [950, 951, 952, 953, 954, 955],
}


def arm_metrics(live_root: Path, arm: str, mode: str, seeds) -> dict:
    margins = {s: [] for s in SLICE_ORDER}
    compliant_steps = {s: 0 for s in SLICE_ORDER}
    total_steps = {s: 0 for s in SLICE_ORDER}
    mos_vals = {s: [] for s in SLICE_ORDER}
    episode_fully_compliant = 0
    episode_total = 0

    for seed in seeds:
        path = live_root / arm / mode / f"rep_seed{seed}" / "omega_log.jsonl"
        if not path.exists():
            print(f"[WARN] missing {path}", file=sys.stderr)
            continue
        for row in _read_omega(path):
            ev = row.get("evidence", {})
            if row.get("step", -1) == -1:
                by_slice = ev.get("episode_sla_compliance_by_slice")
                if by_slice:
                    episode_total += 1
                    if all(by_slice.get(s, 0.0) >= 0.99995 for s in SLICE_ORDER):
                        episode_fully_compliant += 1
                continue
            m = ev.get("per_slice_sla_margin", {})
            c = ev.get("per_slice_compliant", {})
            mbs = ev.get("mos_by_slice", {})
            for s in SLICE_ORDER:
                if s in m:
                    margins[s].append(m[s])
                if s in c:
                    total_steps[s] += 1
                    if c[s]:
                        compliant_steps[s] += 1
                if s in mbs:
                    mos_vals[s].append(mbs[s])

    compliance_pct = {s: 100.0 * compliant_steps[s] / max(1, total_steps[s]) for s in SLICE_ORDER}
    mean_mos = {s: (sum(mos_vals[s]) / len(mos_vals[s]) if mos_vals[s] else float("nan")) for s in SLICE_ORDER}
    return {
        "seeds_used": seeds,
        "compliance_pct": compliance_pct,
        "u_priority_weight": weighted_u(compliance_pct, LIVE_PRIORITY_WEIGHT),
        "u_violation_penalty": weighted_u(compliance_pct, LIVE_VIOLATION_PENALTY),
        "mean_mos_by_slice": mean_mos,
        "severity": {s: percentile_stats(margins[s]) for s in SLICE_ORDER},
        "episodes_fully_compliant": episode_fully_compliant,
        "episodes_total": episode_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", default="/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2")
    ap.add_argument("--out", default="docs/stage5_v2_campaign_metrics_raw.json")
    args = ap.parse_args()

    live_root = Path(args.live_root)
    out = {}
    for arm, mode in ARM_REWARD_MODE.items():
        out[arm] = arm_metrics(live_root, arm, mode, ARM_SEEDS[arm])

    base = out["baseline"]
    for arm in ("dqn_sla", "dqn_qoe", "static_at_cap"):
        out[arm]["fisher_vs_baseline"] = fisher_exact_vs_baseline(
            out[arm]["episodes_fully_compliant"], out[arm]["episodes_total"],
            base["episodes_fully_compliant"], base["episodes_total"],
        )

    # The specific question this validation round exists to answer:
    # does DQN's Stage-3 collapse-avoidance edge over static_at_cap
    # (0/25 vs 4/15 collapsed, p=0.0149 under the OLD calibration) still
    # hold under v2?
    out["dqn_sla"]["fisher_vs_static_at_cap"] = fisher_exact_vs_baseline(
        out["dqn_sla"]["episodes_fully_compliant"], out["dqn_sla"]["episodes_total"],
        out["static_at_cap"]["episodes_fully_compliant"], out["static_at_cap"]["episodes_total"],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    for arm, r in out.items():
        print(f"{arm}: compliance={r['compliance_pct']} U={r['u_priority_weight']:.1f} "
              f"mos={r['mean_mos_by_slice']} episodes_fully_compliant={r['episodes_fully_compliant']}/{r['episodes_total']}",
              file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
