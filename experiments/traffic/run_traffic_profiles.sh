#!/usr/bin/env bash
# Starts/stops the campaign's 3 per-slice traffic generators against the
# iperf3-target container, reading shape parameters from
# experiments/configs/traffic_profiles.yaml (values hardcoded below to match
# that file exactly -- keep the two in sync if either changes; this script
# has no YAML parser dependency by design so it has no extra pip deps).
#
# UE1 (embb) runs in the default netns; UE2 (mmtc) / UE3 (urllc) run inside
# their dedicated netns (ue2ns/ue3ns) -- see CAMPAIGN_LOG.md's stand-up
# section for why (native nr-uesoftmodem TUN interface name collision).
set -uo pipefail

TARGET_IP=172.22.0.50
LOG_DIR="$HOME/oranslice_rig/experiments/logs/traffic"
PID_DIR="$LOG_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# UE PDU session IPs are NOT stable across restarts (Open5GS's UE IP pool
# advances on each new session -- seen firsthand: UE1 went 192.168.100.2 ->
# .100.3 -> .100.6 across this session's restarts) -- auto-detect the
# current IP each time this script runs rather than hardcoding, so it
# doesn't silently target a stale/dead address after any future restart.
ue_ip() {  # $1 = netns name, or "-" for default netns
  if [[ "$1" == "-" ]]; then
    ip -4 addr show oaitun_ue1 | grep -oP 'inet \K[\d.]+'
  else
    sudo ip netns exec "$1" ip -4 addr show oaitun_ue1 | grep -oP 'inet \K[\d.]+'
  fi
}

start_embb() {
  local ip; ip=$(ue_ip -)
  echo "[traffic] starting embb: sustained UDP 4M, 1200B, UE1 ($ip, default netns), port 5201"
  nohup iperf3 -c "$TARGET_IP" -p 5201 -B "$ip" -u -b 4M -l 1200 --reverse -t 0 \
    > "$LOG_DIR/embb.log" 2>&1 &
  echo $! > "$PID_DIR/embb.pid"
}

start_urllc() {
  local ip; ip=$(ue_ip ue3ns)
  echo "[traffic] starting urllc: sustained UDP 300K, 100B, UE3 ($ip, ue3ns), port 5202"
  sudo ip netns exec ue3ns bash -c \
    "nohup iperf3 -c $TARGET_IP -p 5202 -B $ip -u -b 300K -l 100 --reverse -t 0 > $LOG_DIR/urllc.log 2>&1 & echo \$! > $PID_DIR/urllc.pid"
}

start_mmtc() {
  local ip; ip=$(ue_ip ue2ns)
  echo "[traffic] starting mmtc: bursty UDP 50K/80B, 2s on/6s off, UE2 ($ip, ue2ns), port 5203"
  nohup sudo ip netns exec ue2ns bash -c '
    while true; do
      iperf3 -c '"$TARGET_IP"' -p 5203 -B '"$ip"' -u -b 50K -l 80 --reverse -t 2 >> '"$LOG_DIR"'/mmtc.log 2>&1
      sleep 6
    done
  ' > /dev/null 2>&1 &
  echo $! > "$PID_DIR/mmtc.pid"
}

stop_all() {
  echo "[traffic] stopping all traffic generators"
  for f in "$PID_DIR"/*.pid; do
    [[ -f "$f" ]] || continue
    pid=$(cat "$f")
    kill "$pid" 2>/dev/null || true
  done
  # mmtc's loop spawns a fresh iperf3 client each cycle -- kill by pattern too.
  sudo pkill -f "iperf3 -c $TARGET_IP -p 5201" 2>/dev/null || true
  sudo ip netns exec ue3ns pkill -f "iperf3 -c $TARGET_IP -p 5202" 2>/dev/null || true
  sudo ip netns exec ue2ns pkill -f "iperf3 -c $TARGET_IP -p 5203" 2>/dev/null || true
  pkill -f "ip netns exec ue2ns bash -c" 2>/dev/null || true
  rm -f "$PID_DIR"/*.pid
}

status() {
  echo "[traffic] embb:  "; pgrep -af "iperf3 -c $TARGET_IP -p 5201" || echo "  not running"
  echo "[traffic] urllc: "; sudo ip netns exec ue3ns pgrep -af "iperf3 -c $TARGET_IP -p 5202" || echo "  not running"
  echo "[traffic] mmtc:  "; sudo ip netns exec ue2ns pgrep -af "iperf3 -c $TARGET_IP -p 5203" || echo "  not running (between bursts is normal)"
}

case "${1:-}" in
  start)
    stop_all
    start_embb
    start_urllc
    start_mmtc
    sleep 2
    status
    ;;
  stop)
    stop_all
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 2
    ;;
esac
