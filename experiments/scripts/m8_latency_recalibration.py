#!/usr/bin/env python3
"""M8 calibration attempt: replace the raw-byte/Lmax margin ratio (which
produced margin values of -1,002,377.5 for a real, physically-plausible
~10MB backlog -- 5-6 orders of magnitude beyond anything the offline
Normal(5,2)-calibrated Lmax=10 was ever meant to represent, per
docs/PAPER5_M8_live_anchor.md's Part 2) with a physically-grounded
alternative: convert the backlog byte count to an estimated queueing
delay via Little's Law (already implemented, unused for this purpose,
in qoe_oran_framework/calibration/units.py's backlog_bytes_to_latency_s
-- built for QoE-mapper calibration, reused here rather than
reimplemented) and normalise against each slice's OWN already-calibrated
latency_budget_ms, instead of an arbitrary byte count.

Does NOT modify reward.py or kpm_adapter.py (both frozen) -- this is a
new, project-owned post-hoc reanalysis of already-logged data, the same
pattern m6_correctness_metrics.py/m2_correctness_metrics.py already
establish for correctness-aware reanalysis without touching frozen
reward computation.

Back-derives the raw backlog byte count from the ALREADY-LOGGED
per_slice_sla_margin via margin = 1 - raw_queue_len_norm =
1 - queue_raw/Lmax (reward.py's own formula, read directly, not
guessed) -- valid whenever queue_margin dominates loss_margin, which is
guaranteed here since loss_margin is bounds-checked to be far smaller
in magnitude (loss_proxy and loss_budget_pct are both O(1) quantities;
see docs/PAPER5_M8_live_anchor.md Part 2 for the bound). Uses a
representative service throughput per slice from CAMPAIGN_LOG.md's own
empirical PRB-serving measurement (embb: ceiling=4 empirically serves
~15-22 real PRB -- this run's ceiling stayed at 4 throughout, confirmed
directly from the omega log's own "ceilings" field, not assumed) rather
than a per-step reconstruction, since per-step avg_prbs_dl was not
logged by this run -- this approximation is stated plainly, not hidden.

Usage:
    python3 experiments/scripts/m8_latency_recalibration.py \
        --omega-jsonl experiments/results/m8_live_anchor/live_eval/seed900/omega_log.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "framework"))
from qoe_oran_framework.calibration.units import backlog_bytes_to_latency_s  # noqa: E402

LMAX = 10.0  # saclb_live.yaml's current Lmax, the divisor reward.py actually used

# CAMPAIGN_LOG.md's own empirically-measured served-PRB range at
# max_ratio=4 (embb's ceiling throughout this run) -- NOT the configured
# 4 Mbps offered rate, since Little's Law needs the DRAIN (service) rate,
# not the offered one. Range, not a single guess, reported as such.
EMBB_SERVED_PRB_LOW, EMBB_SERVED_PRB_HIGH = 15.0, 22.0
PRB_TO_KBPS = 100.0  # calibration/units.py's own documented approximation
EMBB_LATENCY_BUDGET_S = 45.0 / 1000.0  # saclb_live.yaml's embb latency_budget_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omega-jsonl", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.omega_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            ev = rec.get("evidence", {})
            if "per_slice_sla_margin" in ev:
                rows.append(ev["per_slice_sla_margin"]["embb"])

    print(f"[m8-recal] {len(rows)} steps loaded")
    print(f"[m8-recal] Lmax-based (current) embb margin: first={rows[0]:.1f} "
          f"min={min(rows):.1f} last={rows[-1]:.1f}")

    lo_kbps = EMBB_SERVED_PRB_LOW * PRB_TO_KBPS
    hi_kbps = EMBB_SERVED_PRB_HIGH * PRB_TO_KBPS

    print(f"\n[m8-recal] latency-normalised recalibration "
          f"(service rate {lo_kbps:.0f}-{hi_kbps:.0f} kbps, "
          f"budget {EMBB_LATENCY_BUDGET_S*1000:.0f}ms):")
    print(f"{'step':>5} {'margin(Lmax)':>14} {'backlog_bytes':>14} "
          f"{'latency_s(lo)':>13} {'margin(lo)':>11} {'latency_s(hi)':>13} {'margin(hi)':>11}")
    for i in [0, 1, 2, 3, 4, 5, len(rows) // 2, len(rows) - 1]:
        margin_lmax = rows[i]
        raw_queue_len_norm = 1.0 - margin_lmax
        backlog_bytes = raw_queue_len_norm * LMAX
        lat_lo = float(backlog_bytes_to_latency_s(backlog_bytes, lo_kbps))
        lat_hi = float(backlog_bytes_to_latency_s(backlog_bytes, hi_kbps))
        margin_lo = 1.0 - lat_lo / EMBB_LATENCY_BUDGET_S
        margin_hi = 1.0 - lat_hi / EMBB_LATENCY_BUDGET_S
        print(f"{i:>5} {margin_lmax:>14.1f} {backlog_bytes:>14.1f} "
              f"{lat_lo:>13.4f} {margin_lo:>11.1f} {lat_hi:>13.4f} {margin_hi:>11.1f}")

    # Healthy-reference sanity check: live_campaign_v2's own steady-state
    # embb margin (0.7 under the current Lmax scheme, confirmed genuine --
    # not the dl_errors+dl_bler fallback, checked directly against that
    # run's own "limitation" field) -- what does the SAME backlog read as
    # under the recalibrated scheme?
    healthy_margin_lmax = 0.7
    healthy_backlog = (1.0 - healthy_margin_lmax) * LMAX
    healthy_lat_lo = float(backlog_bytes_to_latency_s(healthy_backlog, lo_kbps))
    healthy_margin_lo = 1.0 - healthy_lat_lo / EMBB_LATENCY_BUDGET_S
    print(f"\n[m8-recal] live_campaign_v2's healthy reference (backlog={healthy_backlog:.1f} bytes, "
          f"Lmax-margin=0.700): recalibrated margin = {healthy_margin_lo:.4f}")


if __name__ == "__main__":
    main()
