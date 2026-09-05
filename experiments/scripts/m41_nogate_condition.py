#!/usr/bin/env python3
"""M41 no-gate condition runner: for UE-count/slice-subset combinations
the standard contention gate can't validate (it's hardcoded to check
embb specifically -- phase1_contention_gate.py --slice-label embb --
so any combo without embb attached would see zero real backlog
pressure and either false-fail or be meaningless, matching the
already-documented single-UE-native-load finding earlier in this
investigation). Mirrors orchestrate()'s own post-bringup monitor loop
exactly (same check_loss/tail_new_retx logic, same 100%-loss and
max-retx stop conditions) so results are directly comparable to gated
conditions -- just skips the gate and the post-gate fresh-restart step,
neither of which is meaningful without embb in the slice set.
Writes its own manifest (nogate_manifest.csv), kept separate from the
gated envelope sweep's manifest.csv so gated and non-gated results are
never accidentally conflated.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m41_envelope_sweep as m41

NOGATE_MANIFEST = m41.RESULTS_ROOT / "nogate_manifest.csv"
NOGATE_FIELDS = [
    "condition_label", "slices", "load_mult", "write_interval_s", "write_mode",
    "duration_s_target", "ts", "bringup_ok", "survived", "onset_s", "onset_reason",
    "elapsed_s",
]


def write_nogate_row(row: dict) -> None:
    NOGATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    exists = NOGATE_MANIFEST.exists()
    with open(NOGATE_MANIFEST, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=NOGATE_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({f: row.get(f, "") for f in NOGATE_FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True, help="comma-separated subset of embb,mmtc,urllc")
    ap.add_argument("--load-mult", type=float, default=1.0)
    ap.add_argument("--write-interval-s", type=float, default=1.0)
    ap.add_argument("--write-mode", choices=["normal", "static"], default="normal")
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--condition-label", required=True)
    args = ap.parse_args()

    slices = set(args.slices.split(","))
    ts = time.strftime("%Y%m%d_%H%M%S")
    row = {
        "condition_label": args.condition_label, "slices": ",".join(sorted(slices)),
        "load_mult": args.load_mult, "write_interval_s": args.write_interval_s,
        "write_mode": args.write_mode, "duration_s_target": args.duration_s, "ts": ts,
        "bringup_ok": None, "survived": None, "onset_s": "", "onset_reason": "", "elapsed_s": "",
    }

    print(f"[nogate] === condition {args.condition_label} === slices={sorted(slices)} "
          f"load_mult={args.load_mult} write_interval_s={args.write_interval_s} "
          f"write_mode={args.write_mode} duration_s={args.duration_s}", file=sys.stderr)

    if not m41.ensure_docker_core(force_fresh=True):
        row.update(bringup_ok=False, survived=False, onset_reason="docker_core_failed")
        write_nogate_row(row)
        return 1

    if not m41.restart_native_stack(ts, slices):
        row.update(bringup_ok=False, survived=False, onset_reason="bringup_failed")
        m41.teardown(None, ts)
        write_nogate_row(row)
        return 1
    row["bringup_ok"] = True

    m41.start_traffic(args.load_mult, ts, slices)
    print("[nogate] traffic launched, 30s stabilization window (no gate)...", file=sys.stderr)
    time.sleep(30)

    probe_args = argparse.Namespace(
        role="probe", condition_label=args.condition_label,
        config=m41.DEFAULT_CONFIG, checkpoint=m41.DEFAULT_CHECKPOINT,
        write_interval_s=args.write_interval_s, write_mode=args.write_mode,
        write_magnitude_cap=None, probe_episodes=500, gnb_id="gnb-0", seed=900,
    )
    probe_proc = m41.launch_probe(probe_args, ts)
    print(f"[nogate] probe launched, PID={probe_proc.pid}, monitoring for up to {args.duration_s}s...",
          file=sys.stderr)

    ue_log_names = {"embb": "ue1", "mmtc": "ue2", "urllc": "ue3"}
    ue_logs = {s: m41.RIG / f"experiments/logs/{ue_log_names[s]}_m41_{ts}.log" for s in slices}
    retx_seen: dict = {}
    for lp in ue_logs.values():
        m41.tail_new_retx(lp, retx_seen)

    start = time.monotonic()
    last_log_tick = 0
    survived = True
    onset_s = None
    onset_reason = None
    condition_log_path = m41.RESULTS_ROOT / args.condition_label / "condition_timeline.jsonl"
    condition_log_path.parent.mkdir(parents=True, exist_ok=True)
    condition_log_fh = open(condition_log_path, "w")

    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= args.duration_s:
                break
            if probe_proc.poll() is not None:
                onset_s, onset_reason, survived = elapsed, "probe_exited_early", False
                break

            new_retx = {slc: m41.tail_new_retx(lp, retx_seen) for slc, lp in ue_logs.items()}
            if any(v > 0 for v in new_retx.values()):
                onset_s, onset_reason, survived = elapsed, f"max_retx:{new_retx}", False
                condition_log_fh.write(json.dumps({"t": elapsed, "event": "max_retx", "detail": new_retx}) + "\n")
                break

            if int(elapsed) // 10 > last_log_tick:
                last_log_tick = int(elapsed) // 10
                loss = {s: m41.check_loss(m41.UE_DEF[s][3]) for s in slices}
                ram_mb = m41.ram_available_mb()
                tick = {"t": elapsed, "loss_pct": loss, "ram_available_mb": ram_mb}
                condition_log_fh.write(json.dumps(tick) + "\n")
                condition_log_fh.flush()
                print(f"[nogate] t={elapsed:.0f}s loss={loss} ram={ram_mb:.0f}MB", file=sys.stderr)
                if any(v >= 100.0 for v in loss.values()):
                    onset_s, onset_reason, survived = elapsed, f"100pct_loss:{loss}", False
                    break

            time.sleep(2)
    finally:
        condition_log_fh.close()
        m41.teardown(probe_proc, ts)

    row.update(
        survived=survived,
        onset_s=f"{onset_s:.1f}" if onset_s is not None else "",
        onset_reason=onset_reason or "",
        elapsed_s=f"{time.monotonic() - start:.1f}",
    )
    write_nogate_row(row)
    print(f"[nogate] === condition {args.condition_label} DONE: survived={survived} "
          f"onset={onset_s} reason={onset_reason} ===", file=sys.stderr)
    return 0 if survived else 1


if __name__ == "__main__":
    sys.exit(main())
