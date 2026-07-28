#!/usr/bin/env python3
"""Stage 3 metrics: the static_at_cap oracle arm vs. the existing live-
campaign arms. Reuses metrics_stage2.py's exact helper functions
(weighted_u, percentile_stats, fisher_exact_vs_baseline) and its exact
omega-log-reading logic, applied to the new arm's logs
(experiments/results/live_campaign/static_at_cap/sla/rep_seed{950,951}/omega_log.jsonl).
baseline/dqn_sla/dqn_qoe numbers are NOT recomputed here -- reused as-is
from docs/stage2_metrics_raw.json (already verified in Stage 2), per the
"never invent a number" rule: nothing here re-derives a figure that
already has a citable source.

static_at_cap was run for only 2 seeds (950, 951), not 3 -- a deliberate
scope cut to fit a ~1h rig budget (see run_phase3_static_at_cap.sh's
header). Its episode count (n=10) is therefore smaller than the other
arms' (n=15); this script does not paper over that difference.

Usage:
    python3 experiments/scripts/metrics_stage3.py --out docs/stage3_metrics_raw.json
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

STATIC_AT_CAP_SEEDS = [950, 951, 952]
DQN_SLA_REVERIFY_SEEDS = [953, 954]


def _arm_metrics(live_root: Path, arm: str, seeds) -> dict:
    margins = {s: [] for s in SLICE_ORDER}
    compliant_steps = {s: 0 for s in SLICE_ORDER}
    total_steps = {s: 0 for s in SLICE_ORDER}
    episode_fully_compliant = 0
    episode_total = 0
    episode_compliance_rows = []  # per-episode per-slice compliance, for the bimodality check

    for seed in seeds:
        path = live_root / arm / "sla" / f"rep_seed{seed}" / "omega_log.jsonl"
        if not path.exists():
            print(f"[WARN] missing {path}", file=sys.stderr)
            continue
        for row in _read_omega(path):
            ev = row.get("evidence", {})
            if row.get("step", -1) == -1:
                by_slice = ev.get("episode_sla_compliance_by_slice")
                if by_slice:
                    episode_total += 1
                    episode_compliance_rows.append({s: by_slice.get(s, 0.0) for s in SLICE_ORDER})
                    if all(by_slice.get(s, 0.0) >= 0.99995 for s in SLICE_ORDER):
                        episode_fully_compliant += 1
                continue
            m = ev.get("per_slice_sla_margin", {})
            c = ev.get("per_slice_compliant", {})
            for s in SLICE_ORDER:
                if s in m:
                    margins[s].append(m[s])
                if s in c:
                    total_steps[s] += 1
                    if c[s]:
                        compliant_steps[s] += 1

    compliance_pct = {s: 100.0 * compliant_steps[s] / max(1, total_steps[s]) for s in SLICE_ORDER}
    return {
        "seeds_used": seeds,
        "compliance_pct": compliance_pct,
        "u_priority_weight": weighted_u(compliance_pct, LIVE_PRIORITY_WEIGHT),
        "u_violation_penalty": weighted_u(compliance_pct, LIVE_VIOLATION_PENALTY),
        "severity": {s: percentile_stats(margins[s]) for s in SLICE_ORDER},
        "episodes_fully_compliant": episode_fully_compliant,
        "episodes_total": episode_total,
        "episode_compliance_rows": episode_compliance_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-root", default="/home/kmanojp/oranslice_rig/experiments/results/live_campaign")
    ap.add_argument("--stage2-raw", default="/home/kmanojp/oranslice_rig/docs/stage2_metrics_raw.json")
    ap.add_argument("--out", default="docs/stage3_metrics_raw.json")
    args = ap.parse_args()

    stage2 = json.load(open(args.stage2_raw))["live_campaign"]
    live_root = Path(args.live_root)
    static_at_cap = _arm_metrics(live_root, "static_at_cap", STATIC_AT_CAP_SEEDS)
    dqn_reverify = _arm_metrics(live_root, "dqn_sla_reverify", DQN_SLA_REVERIFY_SEEDS)

    # Combined DQN(SLA) record: original campaign (seeds 950/951/952, n=15)
    # + reverification on 2 BRAND-NEW seeds never used before (953/954,
    # n=10) -- 25 episodes across 5 distinct seeds total.
    dqn_combined_fully_compliant = stage2["dqn_sla"]["episodes_fully_compliant"] + dqn_reverify["episodes_fully_compliant"]
    dqn_combined_total = stage2["dqn_sla"]["episodes_total"] + dqn_reverify["episodes_total"]

    static_at_cap["fisher_vs_baseline"] = fisher_exact_vs_baseline(
        static_at_cap["episodes_fully_compliant"], static_at_cap["episodes_total"],
        stage2["baseline"]["episodes_fully_compliant"], stage2["baseline"]["episodes_total"],
    )
    static_at_cap["fisher_vs_dqn_sla_original"] = fisher_exact_vs_baseline(
        static_at_cap["episodes_fully_compliant"], static_at_cap["episodes_total"],
        stage2["dqn_sla"]["episodes_fully_compliant"], stage2["dqn_sla"]["episodes_total"],
    )
    static_at_cap["fisher_vs_dqn_sla_combined"] = fisher_exact_vs_baseline(
        static_at_cap["episodes_fully_compliant"], static_at_cap["episodes_total"],
        dqn_combined_fully_compliant, dqn_combined_total,
    )
    dqn_reverify["fisher_vs_baseline"] = fisher_exact_vs_baseline(
        dqn_reverify["episodes_fully_compliant"], dqn_reverify["episodes_total"],
        stage2["baseline"]["episodes_fully_compliant"], stage2["baseline"]["episodes_total"],
    )

    result = {
        "static_at_cap": static_at_cap,
        "dqn_sla_reverify_fresh_seeds": dqn_reverify,
        "dqn_sla_combined": {
            "seeds": stage2["dqn_sla"].get("seeds_used", [950, 951, 952]) + DQN_SLA_REVERIFY_SEEDS,
            "episodes_fully_compliant": dqn_combined_fully_compliant,
            "episodes_total": dqn_combined_total,
        },
        "baseline_reused_from_stage2": stage2["baseline"],
        "dqn_sla_reused_from_stage2": stage2["dqn_sla"],
        "dqn_qoe_reused_from_stage2": stage2["dqn_qoe"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
