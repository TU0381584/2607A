#!/usr/bin/env bash
# M36: add UE3 (urllc, ue3ns) to an already-running gNB+UE1+UE2 stack,
# without touching UE1/UE2. Same pattern as m36_add_ue2.sh, commands
# copied exactly from restart_ran_stack.sh's UE3 block.
set -uo pipefail
BUILD_DIR="$HOME/oranslice_rig/ORANSlice/oai_ran/cmake_targets/ran_build/build"
CONF_DIR="$HOME/oranslice_rig/ORANSlice/oai_ran/targets/PROJECTS/GENERIC-NR-5GC/CONF"
LOG_DIR="$HOME/oranslice_rig/experiments/logs/m36_live"
mkdir -p "$LOG_DIR"

if ! ip netns list 2>/dev/null | grep -q ue3ns; then
  echo "[add-ue3] creating ue3ns + veth pair"
  sudo ip netns add ue3ns
  sudo ip link add veth-ue3h type veth peer name veth-ue3n
  sudo ip link set veth-ue3n netns ue3ns
  sudo ip addr add 10.99.3.1/30 dev veth-ue3h
  sudo ip link set veth-ue3h up
  sudo ip netns exec ue3ns ip addr add 10.99.3.2/30 dev veth-ue3n
  sudo ip netns exec ue3ns ip link set veth-ue3n up
  sudo ip netns exec ue3ns ip link set lo up
else
  echo "[add-ue3] ue3ns already exists, reusing"
fi

cd "$BUILD_DIR"
echo "[add-ue3] launching UE3 (urllc)"
setsid nohup sudo ip netns exec ue3ns ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa \
  -O "$CONF_DIR/nrUE_slice3.conf" --rfsim --rfsimulator.serveraddr 10.99.3.1 \
  < /dev/null > "$LOG_DIR/ue3.log" 2>&1 &
disown -a
echo "[add-ue3] launched, waiting for attach..."
sleep 20

if grep -q "successfully configured" "$LOG_DIR/ue3.log" 2>/dev/null; then
  echo "[add-ue3] UE3 attach confirmed"
else
  echo "[add-ue3] WARNING: 'successfully configured' not seen yet in log, check manually"
fi

echo "[add-ue3] connectivity check"
sudo ip netns exec ue3ns ping -I oaitun_ue1 -c3 -W2 8.8.8.8

echo "[add-ue3] re-checking UE1/UE2 still healthy (not clobbered)"
ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1 && echo "UE1 OK" || echo "UE1 WARNING: connectivity check failed"
sudo ip netns exec ue2ns ping -I oaitun_ue1 -c2 -W2 8.8.8.8 >/dev/null 2>&1 && echo "UE2 OK" || echo "UE2 WARNING: connectivity check failed"

echo "[add-ue3] starting urllc traffic (sustained UDP 300K/100B)"
UE3_IP=$(sudo ip netns exec ue3ns ip -4 addr show oaitun_ue1 | grep -oP 'inet \K[\d.]+')
echo "[add-ue3] UE3 IP: $UE3_IP"
setsid nohup sudo ip netns exec ue3ns iperf3 -c 172.22.0.50 -p 5202 -B "$UE3_IP" -u -b 300K -l 100 --reverse -t 0 \
  < /dev/null > "$LOG_DIR/urllc.log" 2>&1 &
disown -a
echo "[add-ue3] done. Memory check:"
free -h
