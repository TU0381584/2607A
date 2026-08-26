#!/usr/bin/env python3
"""M34: empirical ratio-to-PRB sweep, one slice at a time, under real
traffic. Generalizes phase1_contention_gate.py's single pin/restore cycle
into a multi-point sweep across the ACTUAL operating range a trained
policy can issue (saclb_live.yaml's min_ratio_floor=1 through each
slice's own max_ratio_cap), to build a real ratio->served-PRB mapping
for recalibrating the offline training simulator (ClosedLoopKpmSource
currently assumes 1:1, already shown false in CAMPAIGN_LOG.md: max_ratio=4
empirically served ~15-22 real PRB for embb, not ~4).

Not part of the frozen qoe_oran_framework/ package -- calls only the
framework's public LiveKpmSource API (poll/send_control), same pattern
phase1_contention_gate.py already established.

Protocol per (slice, ratio) pair in the sweep:
  1. PIN: send ONE slicing_control_m with min=max=ratio.
  2. SETTLE: poll --settle-polls times (discarded) to let backlog/service
     reach steady state after the ceiling change.
  3. MEASURE: poll --measure-polls times, record avg_prbs_dl and
     dl_mac_buffer_occupation for that slice's UEs.
Every poll is logged (timestamp, rnti, nssai_sd, ratio, phase,
avg_prbs_dl, dl_mac_buffer_occupation) to a JSONL trace; a summary table
(mean/max avg_prbs_dl per ratio value) is printed at the end.

Usage:
    python3 experiments/scripts/m34_ratio_to_prb_sweep.py \
        --sst 1 --sd 16777215 --slice-label embb \
        --ratios 1 2 3 4 \
        --out experiments/results/live/m34_ratio_sweep/embb_3ue.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnb-id", default="gnb0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--sst", type=int, default=1)
    ap.add_argument("--sd", type=int, required=True)
    ap.add_argument("--slice-label", required=True)
    ap.add_argument("--ratios", type=int, nargs="+", required=True)
    ap.add_argument("--settle-polls", type=int, default=15)
    ap.add_argument("--measure-polls", type=int, default=20)
    ap.add_argument("--restore-min", type=int, default=1)
    ap.add_argument("--restore-max", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = LiveKpmSource(gnb_id=args.gnb_id, host=args.host, recv_timeout_s=10.0)
    rows = []

    def poll_phase(phase: str, ratio: int, n: int) -> None:
        for _ in range(n):
            t = time.time()
            try:
                batch = src.poll()
            except TimeoutError:
                time.sleep(args.interval_s)
                continue
            for ue in batch:
                if ue.nssai_sd != args.sd:
                    continue
                rows.append({
                    "t": t, "phase": phase, "ratio": ratio, "rnti": ue.rnti,
                    "nssai_sd": ue.nssai_sd, "avg_prbs_dl": ue.avg_prbs_dl,
                    "dl_mac_buffer_occupation": ue.dl_mac_buffer_occupation,
                })
            time.sleep(args.interval_s)

    for ratio in args.ratios:
        print(f"[m34] slice={args.slice_label} PIN ratio={ratio}", file=sys.stderr)
        src.send_control(args.gnb_id, args.sst, args.sd, ratio, ratio)
        poll_phase("settle", ratio, args.settle_polls)
        poll_phase("measure", ratio, args.measure_polls)

    print(f"[m34] RESTORE: min={args.restore_min} max={args.restore_max}", file=sys.stderr)
    src.send_control(args.gnb_id, args.sst, args.sd, args.restore_min, args.restore_max)
    src.close()

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n[m34] slice={args.slice_label} sd={args.sd} -- summary (measure phase only)")
    print(f"{'ratio':>6} {'n':>4} {'mean_prb':>10} {'max_prb':>9} {'mean_backlog':>13}")
    for ratio in args.ratios:
        measure_rows = [r for r in rows if r["phase"] == "measure" and r["ratio"] == ratio]
        prbs = [r["avg_prbs_dl"] for r in measure_rows]
        backlog = [r["dl_mac_buffer_occupation"] for r in measure_rows]
        mean_prb = sum(prbs) / len(prbs) if prbs else float("nan")
        max_prb = max(prbs) if prbs else float("nan")
        mean_backlog = sum(backlog) / len(backlog) if backlog else float("nan")
        print(f"{ratio:>6} {len(measure_rows):>4} {mean_prb:>10.2f} {max_prb:>9.2f} {mean_backlog:>13.2f}")
    print(f"\n[m34] trace written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
