#!/usr/bin/env bash
# M36: add UE2 (mmtc, ue2ns) to an already-running gNB+UE1 stack, without
# touching UE1. Netns/veth commands copied exactly from the validated
# restart_ran_stack.sh (idempotent creation block), launch command copied
# exactly from its UE2 launch block, minus the tmux wrapping (this rig
# session doesn't use tmux for the individual UE processes it manages).
set -uo pipefail
BUILD_DIR="$HOME/oranslice_rig/ORANSlice/oai_ran/cmake_targets/ran_build/build"
CONF_DIR="$HOME/oranslice_rig/ORANSlice/oai_ran/targets/PROJECTS/GENERIC-NR-5GC/CONF"
LOG_DIR="$HOME/oranslice_rig/experiments/logs/m36_live"
mkdir -p "$LOG_DIR"

if ! ip netns list 2>/dev/null | grep -q ue2ns; then
  echo "[add-ue2] creating ue2ns + veth pair"
  sudo ip netns add ue2ns
  sudo ip link add veth-ue2h type veth peer name veth-ue2n
  sudo ip link set veth-ue2n netns ue2ns
  sudo ip addr add 10.99.2.1/30 dev veth-ue2h
  sudo ip link set veth-ue2h up
  sudo ip netns exec ue2ns ip addr add 10.99.2.2/30 dev veth-ue2n
  sudo ip netns exec ue2ns ip link set veth-ue2n up
  sudo ip netns exec ue2ns ip link set lo up
else
  echo "[add-ue2] ue2ns already exists, reusing"
fi

cd "$BUILD_DIR"
echo "[add-ue2] launching UE2 (mmtc)"
setsid nohup sudo ip netns exec ue2ns ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa \
  -O "$CONF_DIR/nrUE_slice2.conf" --rfsim --rfsimulator.serveraddr 10.99.2.1 \
  < /dev/null > "$LOG_DIR/ue2.log" 2>&1 &
disown -a
echo "[add-ue2] launched, waiting for attach..."
sleep 20

if grep -q "successfully configured" "$LOG_DIR/ue2.log" 2>/dev/null; then
  echo "[add-ue2] UE2 attach confirmed"
else
  echo "[add-ue2] WARNING: 'successfully configured' not seen yet in log, check manually"
fi

echo "[add-ue2] connectivity check"
sudo ip netns exec ue2ns ping -I oaitun_ue1 -c3 -W2 8.8.8.8

echo "[add-ue2] re-checking UE1 still healthy (not clobbered)"
ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1 && echo "UE1 OK" || echo "UE1 WARNING: connectivity check failed"

echo "[add-ue2] starting mmtc traffic (bursty UDP 50K/80B, 2s on/6s off)"
UE2_IP=$(sudo ip netns exec ue2ns ip -4 addr show oaitun_ue1 | grep -oP 'inet \K[\d.]+')
echo "[add-ue2] UE2 IP: $UE2_IP"
setsid nohup sudo ip netns exec ue2ns bash -c "
  while true; do
    iperf3 -c 172.22.0.50 -p 5203 -B $UE2_IP -u -b 50K -l 80 --reverse -t 2 >> $LOG_DIR/mmtc.log 2>&1
    sleep 6
  done
" < /dev/null > /dev/null 2>&1 &
disown -a
echo "[add-ue2] done. Memory check:"
free -h
