#!/usr/bin/env python3
"""M1, Step 1: extract per-step live traces (per-slice SLA margin, commanded
ceiling as the closest live-observable proxy for served ratio) for the six
dqn_sla single-gNB checkpoints (training seeds 256-261) live-evaluated in
Stage 11 (docs/STAGE11_checkpoint_sensitivity.md).

Live omega_log.jsonl schema (confirmed by direct inspection this session)
has no literal "backlog occupancy" field -- that is purely an offline-
simulator internal state variable (ClosedLoopKpmSource._backlog). The live
system's closest per-step signal is `per_slice_sla_margin` (already a
function of the live backlog-proxy KPM, dl_mac_buffer_occupation or its
dl_errors+dl_bler fallback -- see qoe_mapper.py/reward.py docstrings) and
`ceilings[gnb:slice].max_ratio` (the commanded ceiling, a proxy for what the
slice COULD serve, not a literal served-byte-count). This script extracts
both rather than inventing a live "backlog" number that does not exist in
any log.

Known outlier filter: docs/STAGE13_recalibration_attempt.md documents that
~3.5-8.5% of live per_slice_sla_margin readings across SLA-reward arms show
extreme, monotonically-growing-then-plateauing values (e.g. exactly
-1002377.5) traced to a real RLC max-RETX-style physical failure event, not
a sensor artifact, and NOT part of normal traffic dynamics this calibration
targets. Filtered here via a documented, motivated threshold (abs(margin) >
50 -- normal margins are O(1) per Stage 13's own reported range of
-0.60..+0.75), not a silently-invented magic number.

Checkpoint 256's live data is the full live_campaign_v2/dqn_sla arm
(seeds 950-977, the same data paper #4's Table I uses). Checkpoints
257-261's live data is experiments/results/live_checkpoint_sensitivity/
(seeds 950-955, Stage 11's 21-episode protocol).

Usage:
    python3 experiments/scripts/m1_extract_live_traces.py \
        --out experiments/results/m1_recalibration/live_traces.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/kmanojp/oranslice_rig")

# Established, already-audited per-checkpoint live compliance (Stage 11,
# docs/STAGE11_checkpoint_sensitivity.md's own result table; Stage 14's
# fabrication sweep found zero issues with these numbers). Reused directly
# rather than re-derived, to avoid introducing a new, undocumented number
# for a figure that already has a trusted source.
LIVE_COMPLIANCE_PCT = {
    256: 44 / 46 * 100.0,   # 95.7%, n=46 (already-reported protocol, predates the n=128 campaign)
    257: 13 / 21 * 100.0,   # 61.9%
    258: 21 / 21 * 100.0,   # 100%
    259: 21 / 21 * 100.0,   # 100%
    260: 19 / 21 * 100.0,   # 90.5%
    261: 21 / 21 * 100.0,   # 100%
}

SLICE_IDS = ["embb", "urllc", "mmtc"]
MARGIN_OUTLIER_ABS_THRESHOLD = 50.0  # normal range per Stage 13: -0.60..+0.75


def live_log_paths(train_seed: int):
    if train_seed == 256:
        base = REPO_ROOT / "experiments/results/live_campaign_v2/dqn_sla/sla"
    else:
        base = REPO_ROOT / f"experiments/results/live_checkpoint_sensitivity/dqn_sla_seed{train_seed}/sla"
    return sorted(base.glob("rep_seed*/omega_log.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/results/m1_recalibration/live_traces.json")
    args = ap.parse_args()

    per_checkpoint = {}
    pooled_margins = defaultdict(list)   # slice_id -> [margin, ...] across all 6 checkpoints
    pooled_ceilings = defaultdict(list)  # slice_id -> [max_ratio, ...] across all 6 checkpoints
    n_outliers_dropped = 0
    n_total_readings = 0

    for train_seed in [256, 257, 258, 259, 260, 261]:
        paths = live_log_paths(train_seed)
        if not paths:
            print(f"[m1-extract] WARNING: no live logs found for seed {train_seed}")
            continue
        margins = defaultdict(list)
        ceilings = defaultdict(list)
        n_steps = 0
        for p in paths:
            with open(p) as fh:
                for line in fh:
                    rec = json.loads(line)
                    ev = rec.get("evidence", {})
                    m = ev.get("per_slice_sla_margin")
                    c = ev.get("ceilings")
                    if m is None:
                        continue  # rollup/summary rows carry episode_* fields instead, not per-step
                    n_steps += 1
                    for slice_id in SLICE_IDS:
                        if slice_id not in m:
                            continue
                        n_total_readings += 1
                        val = m[slice_id]
                        if abs(val) > MARGIN_OUTLIER_ABS_THRESHOLD:
                            n_outliers_dropped += 1
                            continue
                        margins[slice_id].append(val)
                        pooled_margins[slice_id].append(val)
                    if c:
                        for key, ratios in c.items():
                            # key format "gnb-0:embb"
                            _, slice_id = key.split(":", 1)
                            if slice_id in SLICE_IDS and "max_ratio" in ratios:
                                ceilings[slice_id].append(ratios["max_ratio"])
                                pooled_ceilings[slice_id].append(ratios["max_ratio"])
        per_checkpoint[train_seed] = {
            "n_step_rows": n_steps,
            "n_files": len(paths),
            "margin_by_slice": {s: v for s, v in margins.items()},
            "ceiling_by_slice": {s: v for s, v in ceilings.items()},
            "live_compliance_pct": LIVE_COMPLIANCE_PCT[train_seed],
        }
        print(f"[m1-extract] seed={train_seed}: {len(paths)} files, {n_steps} step-rows, "
              f"live_compliance={LIVE_COMPLIANCE_PCT[train_seed]:.1f}%")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "per_checkpoint": per_checkpoint,
            "pooled_margin_by_slice": {s: v for s, v in pooled_margins.items()},
            "pooled_ceiling_by_slice": {s: v for s, v in pooled_ceilings.items()},
            "n_outliers_dropped": n_outliers_dropped,
            "n_total_readings": n_total_readings,
            "outlier_threshold": MARGIN_OUTLIER_ABS_THRESHOLD,
        }, fh, indent=2)
    print(f"[m1-extract] wrote {out_path}")
    print(f"[m1-extract] dropped {n_outliers_dropped}/{n_total_readings} "
          f"({100.0 * n_outliers_dropped / max(n_total_readings, 1):.2f}%) as hardware-failure outliers "
          f"(threshold |margin| > {MARGIN_OUTLIER_ABS_THRESHOLD})")


if __name__ == "__main__":
    main()
