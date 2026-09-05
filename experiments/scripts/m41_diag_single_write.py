#!/usr/bin/env python3
"""M41 diagnostic: single-write mechanism instrumentation capture.

Not an envelope-sweep condition (no contention gate, no manifest row) --
a targeted, minimal capture to correlate the new M41DBG instrumentation
(gNB-side SL_sched PRB accounting in dl_sched_unit(), UE-side RLC
retx_count ramp-up, the E2 write itself) against the established t=10.0s
RLC max-RETX failure. Reuses restart_native_stack/start_traffic/teardown
from m41_envelope_sweep.py unmodified; skips the contention gate and its
own PIN-phase backlog damage deliberately, since neither is relevant to
a single mechanism-diagnosis capture and the gate's own damage would
contaminate the clean single-write measurement window.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m41_envelope_sweep as m41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-mult", type=float, default=1.0)
    ap.add_argument("--slices", default=None)
    ap.add_argument("--duration-s", type=float, default=40.0)
    ap.add_argument("--condition-label", default="diag_single_write")
    args = ap.parse_args()

    slices = set(args.slices.split(",")) if args.slices else {"embb", "urllc", "mmtc"}

    print(f"[diag] === single-write instrumentation capture === "
          f"load_mult={args.load_mult} slices={sorted(slices)} duration_s={args.duration_s}",
          file=sys.stderr)

    if not m41.ensure_docker_core(force_fresh=True):
        print("[diag] FATAL: docker core failed", file=sys.stderr)
        return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    if not m41.restart_native_stack(ts, slices):
        print("[diag] FATAL: native stack bring-up failed", file=sys.stderr)
        m41.teardown(None, ts)
        return 1

    m41.start_traffic(args.load_mult, ts, slices)
    print("[diag] traffic launched, 30s stabilization window (no gate)...", file=sys.stderr)
    time.sleep(30)

    probe_args = argparse.Namespace(
        role="probe",
        condition_label=args.condition_label,
        config=m41.DEFAULT_CONFIG,
        checkpoint=m41.DEFAULT_CHECKPOINT,
        write_interval_s=1.0,
        write_mode="static",
        write_magnitude_cap=None,
        probe_episodes=500,
        gnb_id="gnb-0",
        seed=900,
    )
    probe_proc = m41.launch_probe(probe_args, ts)
    print(f"[diag] probe launched PID={probe_proc.pid}, write_mode=static, "
          f"capturing for {args.duration_s}s...", file=sys.stderr)

    start = time.monotonic()
    while time.monotonic() - start < args.duration_s:
        if probe_proc.poll() is not None:
            print(f"[diag] probe exited early, rc={probe_proc.returncode}", file=sys.stderr)
            break
        time.sleep(1)

    print(f"[diag] capture window done at t={time.monotonic()-start:.1f}s, tearing down...", file=sys.stderr)
    m41.teardown(probe_proc, ts)
    print(f"[diag] === DONE ts={ts} ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
