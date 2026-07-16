#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

EXPECTED_UE_COUNT=3
MIN_SESSIONS=3
PROM_URL="http://127.0.0.1:9090"
TARGET_IP="12.1.1.1"
SESSION_RETRIES=45
SESSION_WAIT_SECONDS=2
COUNTER_SAMPLE_SECONDS=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expected-ue-count)
      EXPECTED_UE_COUNT="$2"
      shift 2
      ;;
    --min-sessions)
      MIN_SESSIONS="$2"
      shift 2
      ;;
    --prom-url)
      PROM_URL="$2"
      shift 2
      ;;
    --target-ip)
      TARGET_IP="$2"
      shift 2
      ;;
    --session-retries)
      SESSION_RETRIES="$2"
      shift 2
      ;;
    --session-wait-seconds)
      SESSION_WAIT_SECONDS="$2"
      shift 2
      ;;
    --counter-sample-seconds)
      COUNTER_SAMPLE_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--expected-ue-count N] [--min-sessions N] [--prom-url URL] [--target-ip IP]"
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "[e2e-check] Docker CLI is required but not found"
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[e2e-check] curl is required but not found"
  exit 1
fi

mapfile -t running_names < <(docker ps --format '{{.Names}}')
required=(amf smf upf oaignb)
missing=()
for name in "${required[@]}"; do
  if ! printf '%s\n' "${running_names[@]}" | grep -q "^${name}$"; then
    missing+=("$name")
  fi
done
if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "[e2e-check] Missing required containers: ${missing[*]}"
  exit 1
fi

mapfile -t ue_containers < <(printf '%s\n' "${running_names[@]}" | grep '^nr_ue_' | sort || true)
if [[ "${#ue_containers[@]}" -ne "$EXPECTED_UE_COUNT" ]]; then
  echo "[e2e-check] Expected ${EXPECTED_UE_COUNT} UE containers, found ${#ue_containers[@]}"
  printf '  UE containers: %s\n' "${ue_containers[*]:-<none>}"
  exit 1
fi

echo "[e2e-check] Restarting profile-based random UE traffic loops"
bash "$ROOT_DIR/drl_slicing/scripts/start_ue_traffic_profiles.sh" --target-ip "$TARGET_IP" >/dev/null

for ue in "${ue_containers[@]}"; do
  docker exec "$ue" bash -lc "iface=\$(ip -o link show | awk -F': ' '\$2 ~ /^(uesimtun0|oaitun_ue[0-9][0-9]*)$/ {print \$2; exit}'); if [[ -n \"\$iface\" ]]; then ip route replace ${TARGET_IP}/32 dev \"\$iface\" >/dev/null 2>&1 || true; fi"
done

loop_missing=()
route_missing=()
for ue in "${ue_containers[@]}"; do
  if ! docker exec "$ue" sh -lc "pgrep -f 'ORANSLICE_TRAFFIC_LOOP=1' >/dev/null"; then
    loop_missing+=("$ue")
  fi
  if ! docker exec "$ue" sh -lc "ip route show | grep -q '${TARGET_IP}'"; then
    route_missing+=("$ue")
  fi
done
if [[ "${#loop_missing[@]}" -gt 0 ]]; then
  echo "[e2e-check] Missing traffic loops in: ${loop_missing[*]}"
  exit 1
fi
if [[ "${#route_missing[@]}" -gt 0 ]]; then
  echo "[e2e-check] Missing ${TARGET_IP} route in: ${route_missing[*]}"
  exit 1
fi

session_metric_url="${PROM_URL%/}/api/v1/query?query=fivegs_upffunction_upf_sessionnbr"
session_value=""
for ((attempt=1; attempt<=SESSION_RETRIES; attempt++)); do
  payload="$(curl -fsS "$session_metric_url" || true)"
  session_value="$(echo "$payload" | sed -n 's/.*"value":\[[^,]*,"\([^"]*\)"\].*/\1/p' | head -n 1)"

  if [[ -n "$session_value" ]] && awk "BEGIN {exit !($session_value >= $MIN_SESSIONS)}"; then
    break
  fi

  sleep "$SESSION_WAIT_SECONDS"
done

if [[ -z "$session_value" ]] || ! awk "BEGIN {exit !($session_value >= $MIN_SESSIONS)}"; then
  echo "[e2e-check] UPF sessions did not reach ${MIN_SESSIONS}; observed='${session_value:-unset}'"
  exit 1
fi

read_upf_ogstun_rx_packets() {
  docker exec upf sh -lc "awk -F'[: ]+' '/^ *ogstun:/ {print \$3; found=1; exit} END{if(!found) print 0}' /proc/net/dev"
}

rx_before="$(read_upf_ogstun_rx_packets | tr -d '[:space:]')"
sleep "$COUNTER_SAMPLE_SECONDS"
rx_after="$(read_upf_ogstun_rx_packets | tr -d '[:space:]')"

if [[ ! "$rx_before" =~ ^[0-9]+$ ]] || [[ ! "$rx_after" =~ ^[0-9]+$ ]]; then
  echo "[e2e-check] Could not parse UPF ogstun packet counters"
  exit 1
fi
if (( rx_after <= rx_before )); then
  echo "[e2e-check] UPF ogstun RX packets did not increase (${rx_before} -> ${rx_after})"
  exit 1
fi

echo "[e2e-check] Live ORAN E2E verified: sessions=${session_value}, upf_ogstun_rx=${rx_before}->${rx_after}, ues=${#ue_containers[@]}"
