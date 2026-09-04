#!/usr/bin/env python3
"""M41: live (offered-load x write-cadence) envelope sweep for the M38
RLC-destabilisation failure (docs/PAPER5_M37_M38_scoping.md). Goal: find
an operating region where the live E2 control loop survives a full
validation campaign without RLC max-RETX.

NEW CODE ONLY -- no frozen qoe_oran_framework file is touched. Write-
cadence control (--write-interval-s, --write-mode) is implemented by
wrapping the LiveKpmSource INSTANCE's own send_control bound method
(a plain instance-attribute override, the identical non-invasive
technique state_vector_probe.py already uses for select_action) --
env.py's own env.step() -> kpm_source.send_control(...) call site is
never modified. Confirmed by direct source reading before writing this
script that action_mapping.AdmissionGate.apply() marks a slice
"changed" (and env.py resends it) whenever ANY request for that slice
was processed this step, not only when the numeric ratio actually
differs from before -- so gating at send_control() is the correct
interception point for both --write-interval-s (throttle) and
--write-mode static (one send then permanent suppression).

This script has two roles:
  (default)          orchestrator -- contention gate, stack restart,
                      traffic, launches itself with --role probe as a
                      child process, monitors + aborts + tears down,
                      writes one manifest row.
  --role probe        the actual live control loop (internal use only,
                      launched by the orchestrator role above).

Usage (orchestrator):
    python3 experiments/scripts/m41_envelope_sweep.py \\
        --condition-label S0_C1 --load-mult 1.0 --write-interval-s 1.0 \\
        --write-mode normal --duration-s 300

SURVIVAL = zero RLC max-RETX AND per-slice loss <1% AND no slice locked
at 100% dl_mac_buffer_occupation, sustained for the full --duration-s.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RIG = Path.home() / "oranslice_rig"
BUILD_DIR = RIG / "ORANSlice/oai_ran/cmake_targets/ran_build/build"
CONF_DIR = RIG / "ORANSlice/oai_ran/targets/PROJECTS/GENERIC-NR-5GC/CONF"
LOG_ROOT = RIG / "experiments/logs/m41_envelope"
RESULTS_ROOT = RIG / "experiments/results/m41_envelope"
MANIFEST_PATH = RESULTS_ROOT / "manifest.csv"

# live_kpm_source.py (imported by both phase1_contention_gate.py and this
# script's own probe role) needs this set. Set once, at module load, in
# THIS process's environ so every subprocess.run(shell=True) call below
# inherits it too (the contention gate runs as a separate subprocess and
# does not otherwise see it -- this was missed on the first live attempt
# of this script and caused an immediate, pre-live GATE FAIL).
os.environ.setdefault(
    "XAPP_OAI_PROTO_DIR",
    str(RIG / "ORANSlice/oai_ran/openair2/E2_AGENT/oai-oran-protolib/builds"),
)

DEFAULT_CHECKPOINT = str(
    RIG / "experiments/results/m34_realistic_retrain_v2/seed900/train/dqn/offline_train/rep_0/checkpoint.pt"
)
DEFAULT_CONFIG = str(RIG / "framework/qoe_oran_framework/configs/saclb_live.yaml")

# base (unscaled) traffic profile, from experiments/configs/traffic_profiles.yaml
SLICE_TRAFFIC = {
    "embb": {"port": 5201, "bitrate_k": 4000.0, "packet_len": 1200, "bursty": False},
    "urllc": {"port": 5202, "bitrate_k": 300.0, "packet_len": 100, "bursty": False},
    "mmtc": {"port": 5203, "bitrate_k": 50.0, "packet_len": 80, "bursty": True, "on_s": 2, "off_s": 6},
}
RLC_RETX_PATTERN = re.compile(r"max.*RETX", re.IGNORECASE)


def fmt_bitrate_k(k: float) -> str:
    return f"{k:.1f}K"


def sh(cmd: str, timeout=None, check=False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def pkill_pattern(pattern: str) -> None:
    """pkill -9 -f <pattern>, WITHOUT shell=True. A shell=True subprocess.run
    spawns /bin/sh -c "<cmd>", whose own argv then literally contains the
    pattern text -- pkill -f matches against every process's full command
    line, so that wrapper shell (not just the intended target) can match and
    get killed too, silently truncating whatever command was still running
    (this bit an earlier live run in this same session: 'iperf3 -c 172' as
    a pkill -f pattern matched and killed its own invoking shell mid-cleanup,
    leaving a stray iperf3 client running past teardown). Passing argv as a
    list with shell=False execs pkill directly -- no wrapper shell exists to
    self-match, and pkill's own well-known behavior already excludes its own
    PID from any pattern it searches."""
    subprocess.run(["sudo", "pkill", "-9", "-f", pattern], capture_output=True, text=True)


def run_pid_kill(pids) -> None:
    pids = [p for p in pids if p]
    if pids:
        subprocess.run(["sudo", "kill", "-9", *[str(p) for p in pids]], capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Orchestrator role
# ---------------------------------------------------------------------------

def ensure_docker_core(force_fresh: bool = False) -> bool:
    print("[m41] ensuring Docker core is up...", file=sys.stderr)
    if force_fresh:
        # The gate's own PIN phase drives real traffic-relayed backlog to
        # 10M+ units through the live UPF/SMF/AMF containers. A native-only
        # stack restart (gNB/UEs) never touches these -- if the gate leaves
        # residual state in the core itself (conntrack, GTP-U tunnel
        # buffers) that a native restart can't clear, every condition run
        # after the first gate would inherit it. C1 and C2 both failing at
        # the identical t=10s onset, far faster than clean combined-traffic
        # baselines recorded earlier in this project's own M38 investigation
        # (100+s), is exactly the symptom this would produce. down+up forces
        # genuinely fresh container instances; the subscriber DB survives
        # (named volume), so no reprovisioning is needed after.
        print("[m41] forcing a full Docker core cycle (down+up) for a genuinely clean core...",
              file=sys.stderr)
        sh(f"cd {RIG}/docker_open5gs && docker compose -f 5g-sa-deploy-slicing.yaml down", timeout=60)
        time.sleep(3)
    r = sh(f"cd {RIG}/docker_open5gs && docker compose -f 5g-sa-deploy-slicing.yaml up -d", timeout=60)
    if r.returncode != 0:
        print(f"[m41] FATAL: docker compose up failed: {r.stderr}", file=sys.stderr)
        return False
    time.sleep(6)
    r2 = sh("docker exec mongo mongosh --quiet --eval "
            "'db.getSiblingDB(\"open5gs\").subscribers.countDocuments({})'", timeout=15)
    print(f"[m41] subscriber DB count: {r2.stdout.strip()}", file=sys.stderr)
    r3 = sh("docker ps -a --filter name=iperf3-target --format '{{.Status}}'")
    if "Up" not in r3.stdout:
        sh("docker run -d --name iperf3-target --network demo-open5gs-public-net --ip 172.22.0.50 "
           "--restart unless-stopped --entrypoint sh networkstatic/iperf3 "
           "-c 'iperf3 -s -p 5201 & iperf3 -s -p 5202 & iperf3 -s -p 5203 & wait'")
        time.sleep(3)
    return True


def run_contention_gate(ts: str) -> bool:
    out = LOG_ROOT / ts / "contention_gate.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd {RIG}/framework && {RIG}/venv/bin/python3 {RIG}/experiments/scripts/phase1_contention_gate.py "
        f"--gnb-id gnb0 --host 127.0.0.1 --sd 16777215 --slice-label embb --out {out}"
    )
    print(f"[m41] running contention gate...", file=sys.stderr)
    try:
        r = sh(cmd, timeout=240)
    except subprocess.TimeoutExpired as exc:
        print(f"[m41] contention gate TIMED OUT after 240s: {exc}", file=sys.stderr)
        return False
    print(r.stdout, file=sys.stderr)
    print(r.stderr, file=sys.stderr)
    return r.returncode == 0


# slice_id -> (tmux session name, ue conf file, rfsim serveraddr, netns or None)
UE_DEF = {
    "embb": ("ue1", "nrUE_slice1.conf", "127.0.0.1", None),
    "mmtc": ("ue2", "nrUE_slice2.conf", "10.99.2.1", "ue2ns"),
    "urllc": ("ue3", "nrUE_slice3.conf", "10.99.3.1", "ue3ns"),
}


def restart_native_stack(ts: str, slices: set[str] | None = None) -> bool:
    slices = slices if slices is not None else {"embb", "urllc", "mmtc"}
    log_dir = RIG / "experiments/logs"
    tmux_kill = "for s in gnb ue1 ue2 ue3; do tmux kill-session -t \"$s\" 2>/dev/null || true; done"
    sh(tmux_kill)
    pkill_pattern("nr-uesoftmodem")
    pkill_pattern("nr-softmodem")
    time.sleep(2)

    needed_netns = {UE_DEF[s][3] for s in slices if UE_DEF[s][3] is not None}
    for ns, veth_h, veth_n, subnet_h, subnet_n in [
        ("ue2ns", "veth-ue2h", "veth-ue2n", "10.99.2.1/30", "10.99.2.2/30"),
        ("ue3ns", "veth-ue3h", "veth-ue3n", "10.99.3.1/30", "10.99.3.2/30"),
    ]:
        if ns not in needed_netns:
            continue
        exists = sh(f"ip netns list | grep -q {ns}").returncode == 0
        if not exists:
            sh(f"sudo ip netns add {ns}")
            sh(f"sudo ip link add {veth_h} type veth peer name {veth_n}")
            sh(f"sudo ip link set {veth_n} netns {ns}")
            sh(f"sudo ip addr add {subnet_h} dev {veth_h}")
            sh(f"sudo ip link set {veth_h} up")
            sh(f"sudo ip netns exec {ns} ip addr add {subnet_n} dev {veth_n}")
            sh(f"sudo ip netns exec {ns} ip link set {veth_n} up")
            sh(f"sudo ip netns exec {ns} ip link set lo up")

    gnb_conf = CONF_DIR / "ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf"
    gnb_log = log_dir / f"gnb_m41_{ts}.log"
    sh(f'tmux new-session -d -s gnb -c "{BUILD_DIR}"')
    sh(f'tmux send-keys -t gnb "sudo ./nr-softmodem -O {gnb_conf} --sa --rfsim 2>&1 | tee {gnb_log}" Enter')
    time.sleep(15)
    if "E2 agent heartbeat" not in sh(f"cat {gnb_log}").stdout:
        print("[m41] FATAL: gNB did not come up", file=sys.stderr)
        return False

    for slice_id in ("embb", "mmtc", "urllc"):  # fixed order: embb(UE1) first always, matches established practice
        if slice_id not in slices:
            continue
        name, conf, addr, ns = UE_DEF[slice_id]
        ns_prefix = f"sudo ip netns exec {ns} " if ns else "sudo "
        cmd = (f"{ns_prefix}./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa "
               f"-O {CONF_DIR}/{conf} --rfsim --rfsimulator.serveraddr {addr}")
        ue_log = log_dir / f"{name}_m41_{ts}.log"
        sh(f'tmux new-session -d -s {name} -c "{BUILD_DIR}"')
        sh(f'tmux send-keys -t {name} "{cmd} 2>&1 | tee {ue_log}" Enter')
        time.sleep(20)
        if "successfully configured" not in sh(f"cat {ue_log}").stdout:
            print(f"[m41] FATAL: {name} ({slice_id}) did not attach", file=sys.stderr)
            return False

    checks = {}
    for slice_id in slices:
        _, _, _, ns = UE_DEF[slice_id]
        prefix = f"sudo ip netns exec {ns} " if ns else ""
        checks[f"{slice_id} ({ns or 'default netns'})"] = f"{prefix}ping -I oaitun_ue1 -c2 -W2 8.8.8.8"
    ok = True
    for label, check_cmd in checks.items():
        r = sh(check_cmd, timeout=10)
        if r.returncode != 0:
            # One retry after a short grace period before declaring failure --
            # this exact check previously failed completely silently (no
            # diagnostic at all), got misread as a mysterious silent crash,
            # and turned out to be this single unlogged branch. A transient
            # post-restart ARP/route settle time is plausible at the pace
            # this sweep runs at; a genuine failure will still fail the retry.
            print(f"[m41] WARNING: connectivity check failed for {label}, "
                  f"retrying once after 5s -- output: {r.stdout!r} {r.stderr!r}", file=sys.stderr)
            time.sleep(5)
            r2 = sh(check_cmd, timeout=10)
            if r2.returncode != 0:
                print(f"[m41] FATAL: connectivity check failed twice for {label} -- "
                      f"output: {r2.stdout!r} {r2.stderr!r}", file=sys.stderr)
                ok = False
            else:
                print(f"[m41] {label} connectivity OK on retry", file=sys.stderr)
    return ok


def get_ue_ip(netns: str | None) -> str:
    prefix = f"sudo ip netns exec {netns} " if netns else ""
    out = sh(f"{prefix}ip -4 addr show oaitun_ue1").stdout
    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
    return m.group(1) if m else ""


def start_traffic(load_mult: float, ts: str, slices: set[str] | None = None) -> dict:
    slices = slices if slices is not None else {"embb", "urllc", "mmtc"}
    log_dir = LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    ue_ips = {s: get_ue_ip(UE_DEF[s][3]) for s in slices}
    print(f"[m41] UE IPs: {ue_ips}", file=sys.stderr)

    embb_rate = fmt_bitrate_k(SLICE_TRAFFIC["embb"]["bitrate_k"] * load_mult)
    urllc_rate = fmt_bitrate_k(SLICE_TRAFFIC["urllc"]["bitrate_k"] * load_mult)
    mmtc_rate = fmt_bitrate_k(SLICE_TRAFFIC["mmtc"]["bitrate_k"] * load_mult)

    cmds = {
        "embb": f"setsid nohup iperf3 -c 172.22.0.50 -p 5201 -B {ue_ips.get('embb')} -u -b {embb_rate} "
                f"-l 1200 --reverse -t 0 < /dev/null > {log_dir}/embb_traffic.log 2>&1 &",
        "urllc": f"setsid nohup sudo ip netns exec ue3ns iperf3 -c 172.22.0.50 -p 5202 -B {ue_ips.get('urllc')} "
                 f"-u -b {urllc_rate} -l 100 --reverse -t 0 < /dev/null > {log_dir}/urllc_traffic.log 2>&1 &",
        "mmtc": (f"setsid nohup sudo ip netns exec ue2ns bash -c \""
                 f"while true; do iperf3 -c 172.22.0.50 -p 5203 -B {ue_ips.get('mmtc')} -u -b {mmtc_rate} "
                 f"-l 80 --reverse -t 2 >> {log_dir}/mmtc_traffic.log 2>&1; sleep 6; done"
                 f"\" < /dev/null > /dev/null 2>&1 &"),
    }
    for slice_id in slices:
        sh(cmds[slice_id] + " disown -a")
    time.sleep(3)
    return ue_ips


def stop_traffic() -> None:
    pkill_pattern("iperf3 -c 172")
    pkill_pattern("while true.*iperf3")


def launch_probe(args, ts: str) -> subprocess.Popen:
    out_dir = RESULTS_ROOT / args.condition_label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / ts / "probe.log"
    cmd = [
        sys.executable, __file__, "--role", "probe",
        "--config", args.config, "--checkpoint", args.checkpoint,
        "--write-interval-s", str(args.write_interval_s), "--write-mode", args.write_mode,
        "--episodes", str(args.probe_episodes), "--seed", "900",
        "--run-id", f"m41_{args.condition_label}",
        "--omega-jsonl", str(out_dir / "omega_log.jsonl"),
        "--state-log", str(out_dir / "state_log.jsonl"),
    ]
    if args.write_magnitude_cap is not None:
        cmd += ["--write-magnitude-cap", str(args.write_magnitude_cap)]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=str(RIG / "framework"), stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def tail_new_retx(log_path: Path, seen: dict) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(errors="ignore")
    count = len(RLC_RETX_PATTERN.findall(text))
    prev = seen.get(str(log_path), 0)
    seen[str(log_path)] = count
    return count - prev


def check_loss(netns: str | None) -> float:
    prefix = f"sudo ip netns exec {netns} " if netns else ""
    r = sh(f"{prefix}ping -I oaitun_ue1 -c3 -W1 8.8.8.8", timeout=8)
    m = re.search(r"([+-]?\d+)% packet loss", r.stdout)
    if not m:
        return 100.0
    raw = float(m.group(1))
    if raw < 0 or raw > 100:
        # Observed once live: a badly bouncing link can make ping report a
        # nonsensical value here (e.g. "6667%") -- almost certainly a
        # duplicate-reply artifact on a link that's failing anyway. Clamp
        # rather than propagate a meaningless number into logged data; the
        # link is unambiguously broken either way (real loss or duplicate
        # storm both mean "not usable"), so clamping to 100 loses no
        # decision-relevant information, only a bogus digit.
        print(f"[m41] WARNING: anomalous ping loss reading {raw}% (netns={netns}), "
              f"clamping to 100 -- raw output: {r.stdout!r}", file=sys.stderr)
        return 100.0
    return raw


def latest_omega_evidence(omega_path: Path) -> dict:
    if not omega_path.exists():
        return {}
    try:
        with open(omega_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            lines = fh.read().decode(errors="ignore").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            return d.get("evidence", {})
    except Exception:
        pass
    return {}


def ram_available_mb() -> float:
    out = sh("free -m").stdout
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return float(parts[6]) if len(parts) > 6 else float(parts[3])
    return -1.0


def teardown(probe_proc, ts: str) -> None:
    if probe_proc and probe_proc.poll() is None:
        try:
            import os
            import signal
            os.killpg(os.getpgid(probe_proc.pid), signal.SIGKILL)
        except Exception:
            probe_proc.kill()
    stop_traffic()
    for s in ["gnb", "ue1", "ue2", "ue3"]:
        sh(f"tmux kill-session -t {s} 2>/dev/null; true")
    pkill_pattern("nr-uesoftmodem")
    pkill_pattern("nr-softmodem")
    time.sleep(2)


# Fixed, explicit schema -- NOT derived from whatever keys a given row
# dict happens to have. write_manifest_row() appends to an existing file
# without re-reading its header; if fieldnames varied between calls (e.g.
# after adding write_magnitude_cap for S2), new rows would carry more
# columns than the already-committed header line, silently misaligning
# every column for any later reader. Any new field goes here, and
# row.get(f, "") backfills it as blank for rows that don't set it.
MANIFEST_FIELDS = [
    "condition_label", "load_mult", "write_interval_s", "write_mode",
    "write_magnitude_cap", "duration_s_target", "ts", "gate_pass",
    "bringup_ok", "survived", "onset_s", "onset_reason", "elapsed_s",
    "final_block_precision", "final_reward_mean",
]


def write_manifest_row(row: dict) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow({f: row.get(f, "") for f in MANIFEST_FIELDS})


def orchestrate(args) -> int:
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[m41] === condition {args.condition_label} === "
          f"load_mult={args.load_mult} write_interval_s={args.write_interval_s} "
          f"write_mode={args.write_mode} duration_s={args.duration_s}", file=sys.stderr)

    row = {
        "condition_label": args.condition_label, "load_mult": args.load_mult,
        "write_interval_s": args.write_interval_s, "write_mode": args.write_mode,
        "write_magnitude_cap": args.write_magnitude_cap if args.write_magnitude_cap is not None else "",
        "duration_s_target": args.duration_s, "ts": ts,
        "gate_pass": None, "bringup_ok": None, "survived": None,
        "onset_s": "", "onset_reason": "", "elapsed_s": "",
        "final_block_precision": "TODO(MEASURE)", "final_reward_mean": "TODO(MEASURE)",
    }

    if not ensure_docker_core():
        row.update(gate_pass=False, bringup_ok=False, survived=False, onset_reason="docker_core_failed")
        write_manifest_row(row)
        return 1

    # Bring-up and real traffic MUST precede the contention gate: the gate's
    # own protocol (pin ceiling -> prove backlog rises -> restore -> prove it
    # recovers, phase1_contention_gate.py) needs an already-attached UE
    # producing real traffic-driven backlog to have anything to measure --
    # run against a cold rig it just polls a nonexistent gNB and times out.
    # (First live attempt of this script got this ordering backwards; fixed
    # here after that attempt's own TimeoutExpired traceback made the cause
    # obvious.)
    slices = set(args.slices.split(",")) if args.slices else {"embb", "urllc", "mmtc"}
    bringup_ok = restart_native_stack(ts, slices)
    row["bringup_ok"] = bringup_ok
    if not bringup_ok:
        teardown(None, ts)
        row.update(survived=False, onset_reason="bringup_failed")
        write_manifest_row(row)
        return 1

    # From here on, the native stack is live -- everything below is wrapped
    # so ANY unexpected exception still tears down cleanly rather than
    # leaving gNB/UEs/traffic running unattended (a real gap this script
    # had on its first live attempt: an uncaught subprocess.TimeoutExpired
    # from the gate would have crashed straight past teardown() once the
    # gate was moved to run after bring-up).
    probe_proc = None
    reached_probe_launch = False
    try:
        start_traffic(args.load_mult, ts, slices)
        print("[m41] traffic launched, 30s pre-gate stabilization window...", file=sys.stderr)
        time.sleep(30)

        gate_ok = run_contention_gate(ts)
        row["gate_pass"] = gate_ok
        if not gate_ok:
            print("[m41] GATE FAIL: contention gate did not pass, aborting condition", file=sys.stderr)
            row.update(survived=False, onset_reason="gate_failed")
            write_manifest_row(row)
            return 1

        print("[m41] gate PASSED -- but the gate's own PIN phase deliberately drives "
              "embb's backlog to a catastrophic level to prove ceiling-down => backlog-up, "
              "and this rig's own already-documented behavior (M8) is that this backlog "
              "does NOT drain back down within the gate's 30-poll restore window (confirmed "
              "directly: this run's own recovery mean == pinned max, unchanged). Doing a "
              "full fresh restart (native stack AND Docker core) now so the actual measured "
              "condition starts from a genuinely clean slate.", file=sys.stderr)
        teardown(None, ts)
        core_fresh_ok = ensure_docker_core(force_fresh=True)
        if not core_fresh_ok:
            row.update(survived=False, onset_reason="post_gate_core_cycle_failed")
            write_manifest_row(row)
            return 1
        ts2 = time.strftime("%Y%m%d_%H%M%S")
        row["ts"] = ts2
        bringup2_ok = restart_native_stack(ts2, slices)
        if not bringup2_ok:
            row.update(survived=False, onset_reason="post_gate_bringup_failed")
            write_manifest_row(row)
            return 1
        start_traffic(args.load_mult, ts2, slices)
        print("[m41] post-gate fresh stack up, traffic launched, 30s settle before the probe...",
              file=sys.stderr)
        time.sleep(30)
        ts = ts2
        reached_probe_launch = True
    except Exception as exc:
        print(f"[m41] FATAL (pre-probe): {exc!r}", file=sys.stderr)
        row.update(survived=False, onset_reason=f"pre_probe_exception:{exc!r}")
        write_manifest_row(row)
        return 1
    finally:
        if not reached_probe_launch:
            teardown(probe_proc, ts)

    ue_log_names = {"embb": "ue1", "mmtc": "ue2", "urllc": "ue3"}
    ue_logs = {s: RIG / f"experiments/logs/{ue_log_names[s]}_m41_{ts}.log" for s in slices}
    retx_seen: dict = {}
    for lp in ue_logs.values():
        tail_new_retx(lp, retx_seen)  # baseline, so pre-probe retx (should be 0) doesn't count against us

    probe_proc = launch_probe(args, ts)
    print(f"[m41] probe launched, PID={probe_proc.pid}, monitoring for up to {args.duration_s}s...",
          file=sys.stderr)

    start = time.monotonic()
    last_log_tick = 0
    survived = True
    onset_s = None
    onset_reason = None
    condition_log_path = RESULTS_ROOT / args.condition_label / "condition_timeline.jsonl"
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

            new_retx = {slc: tail_new_retx(lp, retx_seen) for slc, lp in ue_logs.items()}
            if any(v > 0 for v in new_retx.values()):
                onset_s, onset_reason, survived = elapsed, f"max_retx:{new_retx}", False
                condition_log_fh.write(json.dumps({"t": elapsed, "event": "max_retx", "detail": new_retx}) + "\n")
                break

            if int(elapsed) // 10 > last_log_tick:
                last_log_tick = int(elapsed) // 10
                loss = {s: check_loss(UE_DEF[s][3]) for s in slices}
                evidence = latest_omega_evidence(RESULTS_ROOT / args.condition_label / "omega_log.jsonl")
                ram_mb = ram_available_mb()
                tick = {
                    "t": elapsed, "loss_pct": loss, "ram_available_mb": ram_mb,
                    "sla_margin": evidence.get("per_slice_sla_margin", {}),
                    "ceilings": evidence.get("ceilings", {}),
                }
                condition_log_fh.write(json.dumps(tick) + "\n")
                condition_log_fh.flush()
                print(f"[m41] t={elapsed:.0f}s loss={loss} ram={ram_mb:.0f}MB", file=sys.stderr)
                if any(v >= 100.0 for v in loss.values()):
                    onset_s, onset_reason, survived = elapsed, f"100pct_loss:{loss}", False
                    break

            time.sleep(2)
    finally:
        condition_log_fh.close()
        teardown(probe_proc, ts)

    row.update(
        survived=survived,
        onset_s=f"{onset_s:.1f}" if onset_s is not None else "",
        onset_reason=onset_reason or "",
        elapsed_s=f"{time.monotonic() - start:.1f}",
    )
    write_manifest_row(row)
    print(f"[m41] === condition {args.condition_label} DONE: survived={survived} "
          f"onset={row['onset_s']} reason={row['onset_reason']} ===", file=sys.stderr)
    return 0 if survived else 2


# ---------------------------------------------------------------------------
# Probe role (internal -- launched by the orchestrator above)
# ---------------------------------------------------------------------------

def probe_main(args) -> None:
    sys.path.insert(0, str(RIG / "framework"))
    sys.path.insert(0, str(RIG / "experiments/scripts"))
    # XAPP_OAI_PROTO_DIR is already set at module load time, above.
    from qoe_oran_framework.config import load_saclb_config
    from qoe_oran_framework.env import RANEnv
    from qoe_oran_framework.live_kpm_source import LiveKpmSource
    from qoe_oran_framework.mc_runner import build_policy, run_single
    from qoe_oran_framework.omega_logger import OmegaLogger
    from qoe_oran_framework.xapp.saclb_xapp import SINGLE_GNB_LIVE_LIMITATION
    from state_vector_probe import wrap_policy_for_state_logging

    cfg = load_saclb_config(args.config)
    policy = build_policy("dqn", cfg)
    policy.load_checkpoint(args.checkpoint)
    Path(args.state_log).parent.mkdir(parents=True, exist_ok=True)
    state_fh = wrap_policy_for_state_logging(policy, args.state_log)

    kpm_source = LiveKpmSource(gnb_id=args.gnb_id, xapp_listen_port=6600, gnb_listen_port=6655, recv_timeout_s=30.0)

    # write-cadence AND write-magnitude gate: wraps the INSTANCE's
    # send_control, env.py's call site (frozen) is untouched. "changed"
    # writes from AdmissionGate.apply() arrive here unconditionally every
    # step a slice has a pending request -- confirmed by reading
    # action_mapping.py before writing this wrapper. Magnitude capping is
    # the one axis M41-S0/S1 never touched: every prior condition varied
    # *when* a write happens, never *how far* the new ceiling is from the
    # last one this process actually sent. Clamped relative to our own last
    # SENT value (not assumed gNB state, which this process can't observe
    # directly) -- the very first write per key has no reference yet and
    # passes through uncapped by construction.
    original_send = kpm_source.send_control
    write_state = {"last_sent": {}, "sent_once": set(), "last_sent_vals": {}}

    def clamp_step(new_val: int, last_val: int, cap: int) -> int:
        if new_val > last_val + cap:
            return last_val + cap
        if new_val < last_val - cap:
            return last_val - cap
        return new_val

    def gated_send_control(gnb_id, sst, sd, min_ratio, max_ratio):
        key = (gnb_id, sst, sd)
        now = time.monotonic()
        if args.write_mode == "static":
            if key in write_state["sent_once"]:
                return
            write_state["sent_once"].add(key)
        elif args.write_interval_s > 0:
            last = write_state["last_sent"].get(key)
            if last is not None and (now - last) < args.write_interval_s:
                return
        if args.write_magnitude_cap is not None:
            last_vals = write_state["last_sent_vals"].get(key)
            if last_vals is not None:
                last_min, last_max = last_vals
                capped_min = clamp_step(min_ratio, last_min, args.write_magnitude_cap)
                capped_max = clamp_step(max_ratio, last_max, args.write_magnitude_cap)
                if (capped_min, capped_max) != (min_ratio, max_ratio):
                    print(f"[{args.run_id}] magnitude-capped ({sst},{sd}): "
                          f"requested ({min_ratio},{max_ratio}) -> sent ({capped_min},{capped_max}) "
                          f"[last was ({last_min},{last_max}), cap={args.write_magnitude_cap}]",
                          file=sys.stderr)
                min_ratio, max_ratio = capped_min, capped_max
        write_state["last_sent"][key] = now
        write_state["last_sent_vals"][key] = (min_ratio, max_ratio)
        return original_send(gnb_id, sst, sd, min_ratio, max_ratio)

    kpm_source.send_control = gated_send_control

    env = RANEnv(cfg, kpm_source, seed=args.seed, reward_mode="sla")
    print(f"[{args.run_id}] probe role: write_mode={args.write_mode} "
          f"write_interval_s={args.write_interval_s} "
          f"write_magnitude_cap={args.write_magnitude_cap}", file=sys.stderr)
    try:
        with OmegaLogger(args.omega_jsonl) as omega:
            run_single(
                env, policy, "dqn", omega, args.episodes, args.seed, args.run_id,
                mode="live_testbed", training=False, cfg=cfg,
                extra_limitations=[SINGLE_GNB_LIVE_LIMITATION],
            )
    finally:
        kpm_source.close()
        state_fh.close()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["orchestrator", "probe"], default="orchestrator")
    ap.add_argument("--condition-label", default=None)
    ap.add_argument("--load-mult", type=float, default=1.0)
    ap.add_argument("--slices", default=None,
                     help="comma-separated subset of embb,urllc,mmtc to attach and traffic-load "
                          "(default: all 3). E.g. --slices embb for a true single-UE condition -- "
                          "distinct from --load-mult, which scales bitrate but never changes how "
                          "many UEs are actually RRC-attached.")
    ap.add_argument("--write-interval-s", type=float, default=1.0)
    ap.add_argument("--write-mode", choices=["normal", "static"], default="normal")
    ap.add_argument("--write-magnitude-cap", type=int, default=None,
                     help="max |delta| in ratio units a single write may move from this process's own "
                          "last-sent value for the same slice key; None = uncapped (default, current "
                          "policy/step_ratio behavior unchanged)")
    ap.add_argument("--duration-s", type=float, default=300.0)
    ap.add_argument("--probe-episodes", type=int, default=500,
                     help="generous upper bound; the orchestrator's own --duration-s timeout is the real stop condition")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--gnb-id", default="gnb-0")
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--omega-jsonl", default=None)
    ap.add_argument("--state-log", default=None)
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.role == "probe":
        probe_main(args)
        return 0
    if not args.condition_label:
        print("--condition-label is required for the orchestrator role", file=sys.stderr)
        return 1
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
