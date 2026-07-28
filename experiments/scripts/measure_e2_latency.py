#!/usr/bin/env python3
"""Stage 4 (engineering instrumentation), live half: E2 control-loop
round-trip latency, measured directly against the real gNB E2 agent --
NOT derived, assumed, or copied from any other paper. Imports
LiveKpmSource directly (same class RANEnv uses in every live arm) and
times poll() (a true blocking round trip: send INDICATION_REQUEST, block
for INDICATION_RESPONSE) and send_control() (fire-and-forget per the E2
wire protocol -- see live_kpm_source.py's own docstring: CONTROL messages
get no response, applied directly to gNB_MAC_INST -- so what's measured
for send_control is call/send overhead, not a round trip; reported as
such, not mislabeled as a round trip it structurally cannot be).

Needs the live rig (Docker core + gNB, real E2 agent) but NOT traffic
generators or full episodes -- this measures wire-protocol latency, not
compliance/reward, so the campaign's synthetic/real traffic load is not
required for a valid measurement (UEs attached is enough for a
realistic, non-empty UE_LIST payload).

Usage:
    python3 experiments/scripts/measure_e2_latency.py --polls 500 --out docs/stage4_e2_latency_raw.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.live_kpm_source import LiveKpmSource  # noqa: E402

import numpy as np


def stats(latencies_ms):
    arr = np.array(latencies_ms)
    return {
        "n": len(arr), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)), "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()), "max_ms": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gnb-id", default="gnb0")
    ap.add_argument("--polls", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="docs/stage4_e2_latency_raw.json")
    args = ap.parse_args()

    kpm = LiveKpmSource(gnb_id=args.gnb_id)

    for _ in range(args.warmup):
        kpm.poll()

    poll_latencies_ms = []
    n_ues_seen = []
    for _ in range(args.polls):
        t0 = time.perf_counter()
        samples = kpm.poll()
        t1 = time.perf_counter()
        poll_latencies_ms.append((t1 - t0) * 1000.0)
        n_ues_seen.append(len(samples))

    send_control_latencies_ms = []
    for _ in range(args.polls):
        t0 = time.perf_counter()
        kpm.send_control(gnb_id=args.gnb_id, sst=1, sd=1, min_ratio=1, max_ratio=4)
        t1 = time.perf_counter()
        send_control_latencies_ms.append((t1 - t0) * 1000.0)

    out = {
        "poll_round_trip": stats(poll_latencies_ms),
        "send_control_call_overhead_NOT_a_round_trip": stats(send_control_latencies_ms),
        "mean_ues_per_poll": float(np.mean(n_ues_seen)),
    }
    print(json.dumps(out, indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
