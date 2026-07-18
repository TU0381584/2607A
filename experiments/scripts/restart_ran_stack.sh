#!/usr/bin/env bash
# Full RAN stack restart: gNB + all 3 UEs (native processes, tmux sessions).
# Does NOT touch the Docker core (17 containers) -- that has been stable
# throughout this campaign; only the native rfsim processes need this.
#
# Exists because of a finding recorded in CAMPAIGN_LOG.md: hot-restarting a
# SINGLE UE into an already-running stack repeatedly failed (rfsimulator
# socket backlog, RLC max-RETX, PDU session never completes) across 4
# attempts, while a full stop-everything-then-relaunch-in-sequence restart
# succeeded immediately every time. Used by
# experiments/scripts/run_live_eval_arm.sh's health-check loop -- never
# try to restart just the affected UE.
set -uo pipefail

ORANSLICE_HOME="$HOME/oranslice_rig"
BUILD_DIR="$ORANSLICE_HOME/ORANSlice/oai_ran/cmake_targets/ran_build/build"
CONF_DIR="$ORANSLICE_HOME/ORANSlice/oai_ran/targets/PROJECTS/GENERIC-NR-5GC/CONF"
LOG_DIR="$ORANSLICE_HOME/logs"
TS=$(date +%Y%m%d_%H%M%S)

echo "[restart] stopping all RAN processes and tmux sessions"
for s in gnb ue1 ue2 ue3; do tmux kill-session -t "$s" 2>/dev/null || true; done
sudo pkill -9 -f "nr-uesoftmodem" 2>/dev/null || true
sudo pkill -9 -f "nr-softmodem" 2>/dev/null || true
sleep 3

if pgrep -af "nr-softmodem|nr-uesoftmodem" >/dev/null 2>&1; then
  echo "[restart] FATAL: processes still alive after kill, aborting"
  exit 1
fi
echo "[restart] confirmed clean: no RAN processes running"

# Network namespaces + veth pairs for UE2/UE3 do NOT survive a machine
# reboot (found the hard way: a reboot between sessions left `ip netns
# list` empty, causing UE2's launch to silently hang with no PDU session).
# Idempotent: only (re)creates what's missing, never touches an
# already-healthy setup.
if ! ip netns list 2>/dev/null | grep -q ue2ns; then
  echo "[restart] ue2ns missing, recreating netns + veth pair"
  sudo ip netns add ue2ns
  sudo ip link add veth-ue2h type veth peer name veth-ue2n
  sudo ip link set veth-ue2n netns ue2ns
  sudo ip addr add 10.99.2.1/30 dev veth-ue2h
  sudo ip link set veth-ue2h up
  sudo ip netns exec ue2ns ip addr add 10.99.2.2/30 dev veth-ue2n
  sudo ip netns exec ue2ns ip link set veth-ue2n up
  sudo ip netns exec ue2ns ip link set lo up
fi
if ! ip netns list 2>/dev/null | grep -q ue3ns; then
  echo "[restart] ue3ns missing, recreating netns + veth pair"
  sudo ip netns add ue3ns
  sudo ip link add veth-ue3h type veth peer name veth-ue3n
  sudo ip link set veth-ue3n netns ue3ns
  sudo ip addr add 10.99.3.1/30 dev veth-ue3h
  sudo ip link set veth-ue3h up
  sudo ip netns exec ue3ns ip addr add 10.99.3.2/30 dev veth-ue3n
  sudo ip netns exec ue3ns ip link set veth-ue3n up
  sudo ip netns exec ue3ns ip link set lo up
fi

echo "[restart] launching gNB"
tmux new-session -d -s gnb -c "$BUILD_DIR"
tmux send-keys -t gnb "sudo ./nr-softmodem -O $CONF_DIR/ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf --sa --rfsim 2>&1 | tee $LOG_DIR/gnb_${TS}.log" Enter
sleep 15
if ! grep -q "E2 agent heartbeat" "$LOG_DIR/gnb_${TS}.log" 2>/dev/null; then
  echo "[restart] FATAL: gNB did not come up (no E2 heartbeat after 15s)"
  exit 1
fi
echo "[restart] gNB up, E2 agent alive"

echo "[restart] launching UE1 (embb, default netns)"
tmux new-session -d -s ue1 -c "$BUILD_DIR"
tmux send-keys -t ue1 "sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa -O $CONF_DIR/nrUE_slice1.conf --rfsim --rfsimulator.serveraddr 127.0.0.1 2>&1 | tee $LOG_DIR/ue1_${TS}.log" Enter
sleep 20
if ! grep -q "successfully configured" "$LOG_DIR/ue1_${TS}.log" 2>/dev/null; then
  echo "[restart] FATAL: UE1 did not attach (no PDU session after 20s)"
  exit 1
fi
echo "[restart] UE1 attached"

echo "[restart] launching UE2 (mmtc, ue2ns)"
tmux new-session -d -s ue2 -c "$BUILD_DIR"
tmux send-keys -t ue2 "sudo ip netns exec ue2ns ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa -O $CONF_DIR/nrUE_slice2.conf --rfsim --rfsimulator.serveraddr 10.99.2.1 2>&1 | tee $LOG_DIR/ue2_${TS}.log" Enter
sleep 20
if ! grep -q "successfully configured" "$LOG_DIR/ue2_${TS}.log" 2>/dev/null; then
  echo "[restart] FATAL: UE2 did not attach (no PDU session after 20s)"
  exit 1
fi
echo "[restart] UE2 attached"

echo "[restart] launching UE3 (urllc, ue3ns)"
tmux new-session -d -s ue3 -c "$BUILD_DIR"
tmux send-keys -t ue3 "sudo ip netns exec ue3ns ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa -O $CONF_DIR/nrUE_slice3.conf --rfsim --rfsimulator.serveraddr 10.99.3.1 2>&1 | tee $LOG_DIR/ue3_${TS}.log" Enter
sleep 20
if ! grep -q "successfully configured" "$LOG_DIR/ue3_${TS}.log" 2>/dev/null; then
  echo "[restart] FATAL: UE3 did not attach (no PDU session after 20s)"
  exit 1
fi
echo "[restart] UE3 attached"

echo "[restart] verifying connectivity all 3 UEs"
ok=1
ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1 || ok=0
sudo ip netns exec ue2ns ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1 || ok=0
sudo ip netns exec ue3ns ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1 || ok=0

if [[ "$ok" -eq 1 ]]; then
  echo "[restart] SUCCESS: all 3 UEs attached and reachable"
  exit 0
else
  echo "[restart] FATAL: post-restart connectivity check failed"
  exit 1
fi
