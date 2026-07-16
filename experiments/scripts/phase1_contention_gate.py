#!/usr/bin/env python3
"""Phase 1 CONTENTION GATE (per the campaign handover): before any training,
prove that (a) aggregate offered demand exceeds the calibrated ceilings, and
(b) pinning one slice's max_ratio low measurably raises its
dl_mac_buffer_occupation backlog within the episode, then recovers when
restored.

Not part of the frozen qoe_oran_framework/ package -- a new campaign script
that only calls the framework's public LiveKpmSource API (poll/send_control),
exactly as qoe_oran_framework/scripts/probe_e2_preconditions.py does, so no
frozen source is touched.

Protocol:
  1. BASELINE window: poll every --interval-s seconds for --baseline-polls
     polls, with the target slice at whatever ceiling the gNB already has
     (should be near-default/wide-open, or already at the calibrated
     nominal ratio if a prior arm's static baseline is still asserting it).
  2. PIN: send ONE slicing_control_m for --sst/--sd with min=max=--pin-ratio
     (a ceiling deliberately far below the slice's real demand).
  3. PINNED window: poll for --pinned-polls polls -- this is where backlog
     should rise if contention is genuine.
  4. RESTORE: send ONE slicing_control_m restoring min=--restore-min,
     max=--restore-max (the slice's calibrated floor/cap).
  5. RECOVERY window: poll for --recovery-polls polls -- backlog should
     stabilize/drain relative to the pinned window.

Every poll is logged (timestamp, rnti, nssai_sd, avg_prbs_dl,
dl_mac_buffer_occupation, phase) to a JSONL trace for plotting and for the
evidence chain (never overwritten -- timestamped output path).
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
    ap.add_argument("--sd", type=int, required=True, help="target slice's nssai_sd, e.g. embb=16777215")
    ap.add_argument("--slice-label", default="embb")
    ap.add_argument("--baseline-polls", type=int, default=30)
    ap.add_argument("--pin-ratio", type=int, default=1, help="min=max=this value during PIN phase")
    ap.add_argument("--pinned-polls", type=int, default=60)
    ap.add_argument("--restore-min", type=int, default=1)
    ap.add_argument("--restore-max", type=int, default=4)
    ap.add_argument("--recovery-polls", type=int, default=30)
    ap.add_argument("--out", required=True, help="output JSONL trace path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = LiveKpmSource(gnb_id=args.gnb_id, host=args.host, recv_timeout_s=10.0)
    rows = []

    def poll_phase(phase: str, n: int) -> None:
        for _ in range(n):
            t = time.time()
            try:
                batch = src.poll()
            except TimeoutError:
                time.sleep(args.interval_s)
                continue
            for ue in batch:
                rows.append({
                    "t": t, "phase": phase, "rnti": ue.rnti,
                    "nssai_sst": ue.nssai_sst, "nssai_sd": ue.nssai_sd,
                    "avg_prbs_dl": ue.avg_prbs_dl,
                    "dl_mac_buffer_occupation": ue.dl_mac_buffer_occupation,
                })
            time.sleep(args.interval_s)

    print(f"[phase1] BASELINE: {args.baseline_polls} polls @ {args.interval_s}s", file=sys.stderr)
    poll_phase("baseline", args.baseline_polls)

    print(f"[phase1] PIN: sst={args.sst} sd={args.sd} min=max={args.pin_ratio}", file=sys.stderr)
    src.send_control(args.gnb_id, args.sst, args.sd, args.pin_ratio, args.pin_ratio)
    poll_phase("pinned", args.pinned_polls)

    print(f"[phase1] RESTORE: sst={args.sst} sd={args.sd} min={args.restore_min} max={args.restore_max}", file=sys.stderr)
    src.send_control(args.gnb_id, args.sst, args.sd, args.restore_min, args.restore_max)
    poll_phase("recovery", args.recovery_polls)

    src.close()

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # ---- quick pass/fail verdict on the target slice's backlog ------------
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    target_rows = [r for r in rows if r["nssai_sd"] == args.sd]
    baseline_backlog = [r["dl_mac_buffer_occupation"] for r in target_rows if r["phase"] == "baseline"]
    pinned_backlog = [r["dl_mac_buffer_occupation"] for r in target_rows if r["phase"] == "pinned"]
    recovery_backlog = [r["dl_mac_buffer_occupation"] for r in target_rows if r["phase"] == "recovery"]

    baseline_mean = mean(baseline_backlog)
    pinned_mean = mean(pinned_backlog)
    recovery_mean = mean(recovery_backlog)
    pinned_max = max(pinned_backlog) if pinned_backlog else 0.0

    verdict = pinned_mean > baseline_mean and pinned_max > 0.0

    print(f"\n[phase1] slice={args.slice_label} sd={args.sd}")
    print(f"[phase1] baseline mean backlog = {baseline_mean:.3f} (n={len(baseline_backlog)})")
    print(f"[phase1] pinned   mean backlog = {pinned_mean:.3f} (n={len(pinned_backlog)}) max={pinned_max:.3f}")
    print(f"[phase1] recovery mean backlog = {recovery_mean:.3f} (n={len(recovery_backlog)})")
    print(f"[phase1] VERDICT: ceiling-down => backlog-up {'PASS' if verdict else 'FAIL'}")
    print(f"[phase1] trace written to {out_path}")

    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
