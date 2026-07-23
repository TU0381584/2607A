#!/usr/bin/env python3
"""Live-rig phase-timed traffic control: restarts each slice's iperf3
client at a new rate at step boundaries, synced to the SAME 3-phase
schedule used by the offline probe (experiments/scripts/
phase_varying_kpm_source.py) -- steps 1-20 low (0.7x), 21-40 high (1.3x),
41-60 medium (1.0x, == run_traffic_profiles.sh's existing baseline rate).

Live step cadence is fixed at cfg.episode.step_seconds (5.0s) by
mc_runner.run_single's own pacing (time.sleep to the configured cadence),
so phase boundaries are called by STEP INDEX, not a separate wall-clock
timer -- the live evaluation driver calls set_phase_for_step(step_idx)
once per step, before env.step(), exactly mirroring how the offline
probe's PhaseVaryingClosedLoopKpmSource checks phase per poll().

UE PDU session IPs are re-detected fresh each call (not cached, not
hardcoded) -- see run_traffic_profiles.sh's own module docstring for why
(Open5GS's UE IP pool advances across restarts).
"""
import subprocess
import sys
from typing import Optional

TARGET_IP = "172.22.0.50"
BASE_RATES = {"embb": "4M", "urllc": "300K", "mmtc": "50K"}
BASE_PACKET_LEN = {"embb": 1200, "urllc": 100, "mmtc": 80}
PHASE_MULTIPLIER = {"low": 0.7, "high": 1.3, "medium": 1.0}
PHASE_BOUNDARIES = [(1, 20, "low"), (21, 40, "high"), (41, 60, "medium")]

LOG_DIR = "/home/kmanojp/oranslice_rig/experiments/logs/traffic"


def phase_for_step(step_idx: int) -> str:
    for lo, hi, name in PHASE_BOUNDARIES:
        if lo <= step_idx <= hi:
            return name
    return PHASE_BOUNDARIES[-1][2]


def _rate_with_multiplier(base: str, mult: float) -> str:
    """e.g. "4M" * 0.7 -> "2800000" (bytes/s spec iperf3 accepts numerically)."""
    unit = base[-1]
    value = float(base[:-1])
    scaled = value * mult
    return f"{scaled:.0f}{unit}"


def _ue_ip(netns: Optional[str]) -> str:
    if netns is None:
        out = subprocess.run(["ip", "-4", "addr", "show", "oaitun_ue1"], capture_output=True, text=True).stdout
    else:
        out = subprocess.run(
            ["sudo", "ip", "netns", "exec", netns, "ip", "-4", "addr", "show", "oaitun_ue1"],
            capture_output=True, text=True,
        ).stdout
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("inet "):
            return line.split()[1].split("/")[0]
    raise RuntimeError(f"could not detect oaitun_ue1 IP in netns={netns}")


def _kill_slice(slice_id: str) -> None:
    port = {"embb": 5201, "urllc": 5202, "mmtc": 5203}[slice_id]
    if slice_id == "embb":
        subprocess.run(["pkill", "-f", f"iperf3 -c {TARGET_IP} -p {port}"], check=False)
    elif slice_id == "urllc":
        subprocess.run(["sudo", "ip", "netns", "exec", "ue3ns", "pkill", "-f", f"iperf3 -c {TARGET_IP} -p {port}"], check=False)
    else:  # mmtc -- kill the bursty bash loop AND any in-flight iperf3 client
        subprocess.run(["sudo", "ip", "netns", "exec", "ue2ns", "pkill", "-f", f"iperf3 -c {TARGET_IP} -p {port}"], check=False)
        subprocess.run(["sudo", "pkill", "-f", "ip netns exec ue2ns bash -c"], check=False)


def _start_slice(slice_id: str, rate: str) -> None:
    port = {"embb": 5201, "urllc": 5202, "mmtc": 5203}[slice_id]
    pkt_len = BASE_PACKET_LEN[slice_id]
    if slice_id == "embb":
        ip = _ue_ip(None)
        cmd = f"nohup iperf3 -c {TARGET_IP} -p {port} -B {ip} -u -b {rate} -l {pkt_len} --reverse -t 0 >> {LOG_DIR}/embb.log 2>&1 &"
        subprocess.run(["bash", "-c", cmd])
    elif slice_id == "urllc":
        ip = _ue_ip("ue3ns")
        cmd = f"nohup iperf3 -c {TARGET_IP} -p {port} -B {ip} -u -b {rate} -l {pkt_len} --reverse -t 0 >> {LOG_DIR}/urllc.log 2>&1 &"
        subprocess.run(["sudo", "ip", "netns", "exec", "ue3ns", "bash", "-c", cmd])
    else:  # mmtc: 2s-on/6s-off bursty loop, matching run_traffic_profiles.sh
        ip = _ue_ip("ue2ns")
        cmd = (
            f"nohup bash -c 'while true; do "
            f"iperf3 -c {TARGET_IP} -p {port} -B {ip} -u -b {rate} -l {pkt_len} --reverse -t 2 >> {LOG_DIR}/mmtc.log 2>&1; "
            f"sleep 6; done' > /dev/null 2>&1 &"
        )
        subprocess.run(["sudo", "ip", "netns", "exec", "ue2ns", "bash", "-c", cmd])


def apply_phase(phase_name: str) -> None:
    mult = PHASE_MULTIPLIER[phase_name]
    for slice_id, base_rate in BASE_RATES.items():
        _kill_slice(slice_id)
    import time
    time.sleep(0.5)
    for slice_id, base_rate in BASE_RATES.items():
        rate = _rate_with_multiplier(base_rate, mult)
        _start_slice(slice_id, rate)


_last_phase = {"name": None}


def set_phase_for_step(step_idx: int) -> None:
    """Call once per live step, BEFORE env.step(). Only restarts traffic
    when the phase actually changes (not every step)."""
    phase_name = phase_for_step(step_idx)
    if phase_name != _last_phase["name"]:
        print(f"[phase_traffic_control] step={step_idx} -> phase={phase_name}", file=sys.stderr)
        apply_phase(phase_name)
        _last_phase["name"] = phase_name


def reset_phase_clock() -> None:
    _last_phase["name"] = None


if __name__ == "__main__":
    # Manual smoke test of the rate-restart mechanics alone (no RANEnv).
    for step in [1, 21, 41]:
        set_phase_for_step(step)
        print(f"applied phase for step {step}")
