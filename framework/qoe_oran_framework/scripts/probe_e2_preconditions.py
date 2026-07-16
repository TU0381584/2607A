#!/usr/bin/env python3
"""Live E2 precondition probe for a freshly built rig (run BEFORE any
application code, per PROJECT_HANDOFF_SUMMARY.md §5).

Automates the live half of the migration precondition checklist:

  P2  wire protocol   -- one real poll() round-trip against the running gNB's
                         UDP E2_AGENT (INDICATION_REQUEST -> one
                         INDICATION_RESPONSE). Never sends SUBSCRIPTION.
  P4  KPM population  -- polls N times and reports, per slice, the fraction of
                         polls in which each KPM field was present/nonzero
                         (re-characterization of dl_mac_buffer_occupation
                         intermittency etc. before wiring the QoE mapper).
  P3  real demand     -- summarizes observed per-UE avg_prbs_dl so PRB ratio
                         caps can be set to actually bind (handoff §4.4:
                         always poll real demand before setting ratio caps).
  P5  control path    -- OPT-IN (--send-control): sends one slicing_control_m
                         at the values you pass, then re-polls. This MUTATES
                         the live gNB slicing policy; only use values matching
                         your configured floor/ceiling.

Usage (on the rig, with core+gNB up and >=1 UE attached):
  export XAPP_OAI_PROTO_DIR=<oai_ran>/openair2/E2_AGENT/oai-oran-protolib/builds
  python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --polls 60
  # optionally, after reviewing demand output:
  python3 -m qoe_oran_framework.scripts.probe_e2_preconditions \
      --send-control --sst 1 --sd 0 --min-ratio 5 --max-ratio 50
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

from qoe_oran_framework.live_kpm_source import LiveKpmSource

# Fields re-characterized for precondition P4. Names match UeSample /
# ran_messages.proto ue_info_m (verified identical on ORANSlice main,
# OAI 2024.w28 base, 2026-07-14).
KPM_FIELDS = [
    "avg_prbs_dl",
    "dl_mac_buffer_occupation",
    "dl_total_bytes",
    "dl_errors",
    "dl_bler",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnb-id", default="gnb0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--polls", type=int, default=60,
                    help="number of poll() round-trips for P3/P4 stats")
    ap.add_argument("--interval-s", type=float, default=0.5)
    ap.add_argument("--recv-timeout-s", type=float, default=10.0)
    ap.add_argument("--send-control", action="store_true",
                    help="P5: send ONE slicing_control_m (mutates live policy)")
    ap.add_argument("--sst", type=int, default=1)
    ap.add_argument("--sd", type=int, default=0)
    ap.add_argument("--min-ratio", type=int)
    ap.add_argument("--max-ratio", type=int)
    args = ap.parse_args()

    src = LiveKpmSource(gnb_id=args.gnb_id, host=args.host,
                        recv_timeout_s=args.recv_timeout_s)

    # ---- P2: single round-trip ------------------------------------------
    print("[P2] wire protocol: sending one INDICATION_REQUEST ...")
    try:
        samples = src.poll()
    except TimeoutError as exc:
        print(f"[P2] FAIL: {exc}")
        print("     -> is the gNB up? is the E2_AGENT loop logging "
              "'Indication request message received'?")
        return 1
    print(f"[P2] PASS: got INDICATION_RESPONSE with {len(samples)} UE sample(s)")
    if not samples:
        print("     NOTE: 0 UEs attached; P3/P4 stats will be empty. "
              "Attach a UE and re-run.")

    # ---- P3 + P4: repeated polls -----------------------------------------
    present = defaultdict(lambda: defaultdict(int))   # slice -> field -> nonzero count
    seen = defaultdict(int)                           # slice -> ue-sample count
    prbs = defaultdict(list)                          # rnti -> avg_prbs_dl readings
    n_ok = 0
    for _ in range(args.polls):
        try:
            batch = src.poll()
        except TimeoutError:
            continue
        n_ok += 1
        for ue in batch:
            slc = f"sst{ue.nssai_sst}/sd{ue.nssai_sd}"
            seen[slc] += 1
            for f in KPM_FIELDS:
                v = getattr(ue, f, None)
                if v is not None and v != 0:
                    present[slc][f] += 1
            if ue.avg_prbs_dl:
                prbs[ue.rnti].append(ue.avg_prbs_dl)
        time.sleep(args.interval_s)

    print(f"\n[P4] KPM field population over {n_ok}/{args.polls} successful polls:")
    for slc in sorted(seen):
        n = seen[slc]
        rates = "  ".join(
            f"{f}={100.0 * present[slc][f] / n:5.1f}%" for f in KPM_FIELDS
        )
        print(f"     {slc} (n={n}): {rates}")
    if not seen:
        print("     (no UE samples observed)")

    print("\n[P3] real per-UE demand (avg_prbs_dl) -- set ratio caps BELOW "
          "aggregate demand or they will never bind (handoff §4.4):")
    for rnti, vals in sorted(prbs.items()):
        print(f"     rnti={rnti}: mean={sum(vals)/len(vals):.2f} PRB  "
              f"max={max(vals):.2f}  n={len(vals)}")
    if not prbs:
        print("     (no nonzero avg_prbs_dl observed -- start real traffic "
              "before trusting demand numbers)")

    # ---- P5: optional control round-trip ---------------------------------
    if args.send_control:
        if args.min_ratio is None or args.max_ratio is None:
            print("[P5] --send-control requires --min-ratio and --max-ratio")
            return 2
        print(f"\n[P5] sending slicing_control_m sst={args.sst} sd={args.sd} "
              f"min={args.min_ratio} max={args.max_ratio} (fire-and-forget, "
              "no response expected by protocol) ...")
        src.send_control(args.gnb_id, args.sst, args.sd,
                         args.min_ratio, args.max_ratio)
        time.sleep(1.0)
        after = src.poll()
        print(f"[P5] post-control poll OK ({len(after)} UE sample(s)); check "
              "gNB log for 'Control message received' + applied ratios.")

    src.close()
    print("\nDone. NEVER send SUBSCRIPTION-type messages: still assert(0!=0) "
          "in handle_subscription() on this OAI base (verified 2026-07-14).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
