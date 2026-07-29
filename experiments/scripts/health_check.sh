#!/usr/bin/env bash
# Quick health check: all 3 UEs reachable, all 4 RAN processes alive, no
# segfault in dmesg since a given marker. Exit 0 = healthy, 1 = unhealthy.
# Used by experiments/scripts/run_live_eval_arm.sh between episode batches
# to decide whether experiments/scripts/restart_ran_stack.sh is needed.
set -uo pipefail

ok=1

n_procs=$(pgrep -af "nr-softmodem|nr-uesoftmodem" | grep -v grep | wc -l)
if [[ "$n_procs" -lt 16 ]]; then
  echo "[health] FAIL: expected >=16 RAN process/thread entries, found $n_procs"
  ok=0
fi

if ! ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1; then
  echo "[health] FAIL: UE1 (embb) unreachable"
  ok=0
fi
if ! sudo ip netns exec ue2ns ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1; then
  echo "[health] FAIL: UE2 (mmtc) unreachable"
  ok=0
fi
if ! sudo ip netns exec ue3ns ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1; then
  echo "[health] FAIL: UE3 (urllc) unreachable"
  ok=0
fi

# Restricted to RAN process names specifically (2026-07-29 fix): the
# unrestricted "segfault|general protection" match also fires on
# unrelated host processes -- confirmed directly via dmesg during Stage 5
# ("iperf3[...]: segfault ... in libiperf.so", a traffic-generator client
# crash, not a RAN process) -- which then falsely marks the rig
# unhealthy and burns restart attempts that can never fix a non-RAN
# crash. Only a segfault naming nr-softmodem/nr-uesoftmodem indicates a
# genuine RAN-stack problem this check should act on.
if sudo dmesg | tail -200 | grep -iE "segfault|general protection" | grep -qiE "nr-softmodem|nr-uesoftmodem"; then
  echo "[health] FAIL: segfault signature in recent dmesg (RAN process)"
  ok=0
fi

if [[ "$ok" -eq 1 ]]; then
  echo "[health] OK: all 3 UEs reachable, $n_procs RAN processes alive, no recent segfault"
  exit 0
else
  exit 1
fi
