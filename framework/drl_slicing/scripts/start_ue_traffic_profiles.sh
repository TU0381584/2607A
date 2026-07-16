#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_PROFILE_CSV="$ROOT_DIR/docker_open5gs/generated/ue_fleet_profiles.csv"

PROFILE_CSV="$DEFAULT_PROFILE_CSV"
TARGET_IP="12.1.1.1"
DRY_RUN=0
TRAFFIC_SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      PROFILE_CSV="$2"
      shift 2
      ;;
    --target-ip)
      TARGET_IP="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --seed)
      TRAFFIC_SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--profiles <csv>] [--target-ip <ip>] [--seed <int>] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ ! -f "$PROFILE_CSV" ]]; then
  echo "Profile CSV not found: $PROFILE_CSV"
  exit 1
fi

echo "Starting profile-based UE traffic"
echo "  Profiles: $PROFILE_CSV"
echo "  Target : $TARGET_IP"

started=0
skipped=0

while IFS=, read -r _service container _component _imsi _imei _imeisv _ki _op _amf profile _sst _sd _slice_id; do
  [[ -z "$container" ]] && continue
  profile="${profile//$'\r'/}"
  profile="${profile,,}"

  if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
    skipped=$((skipped + 1))
    continue
  fi

  iface_finder='iface=$(ip -o link show | awk -F": " '\''$2 ~ /^(uesimtun0|oaitun_ue[0-9][0-9]*)$/ {print $2; exit}'\'' ); if [[ -z "$iface" ]]; then iface="uesimtun0"; fi;'

  # Per-profile traffic shape:
  # - ping path (preferred when available)
  # - bash /dev/udp fallback for minimal UE images without ping
  case "$profile" in
    embb)
      pkt_base=1200
      pkt_span=180
      burst_count=25
      inter_pkt_sleep=0.03
      post_burst_sleep='0.$((RANDOM % 3 + 1))'
      ;;
    urllc)
      pkt_base=48
      pkt_span=64
      burst_count=8
      inter_pkt_sleep=0.15
      post_burst_sleep='0.$((RANDOM % 2 + 1))'
      ;;
    mmtc)
      pkt_base=24
      pkt_span=24
      burst_count=2
      inter_pkt_sleep=1.0
      post_burst_sleep='$((RANDOM % 5 + 2))'
      ;;
    *)
      pkt_base=256
      pkt_span=1
      burst_count=10
      inter_pkt_sleep=0.1
      post_burst_sleep='1'
      ;;
  esac

  seed_prefix=""
  if [[ -n "$TRAFFIC_SEED" ]]; then
    seed_value=$((TRAFFIC_SEED + started * 97))
    seed_prefix="RANDOM=${seed_value}; "
  fi

  traffic_cmd="${seed_prefix}ORANSLICE_TRAFFIC_LOOP=1; while true; do ${iface_finder} ip route replace ${TARGET_IP}/32 dev \"\$iface\" >/dev/null 2>&1 || true; if command -v ping >/dev/null 2>&1; then ping -I \"\$iface\" -s \$(( ${pkt_base} + RANDOM % ${pkt_span} )) -i ${inter_pkt_sleep} -c ${burst_count} ${TARGET_IP} >/dev/null 2>&1 || true; else for ((i=0;i<${burst_count};i++)); do payload_size=\$(( ${pkt_base} + RANDOM % ${pkt_span} )); printf -v payload '%*s' \"\$payload_size\" ''; payload=\${payload// /x}; printf '%s' \"\$payload\" > /dev/udp/${TARGET_IP}/5001 || true; sleep ${inter_pkt_sleep}; done; fi; sleep ${post_burst_sleep}; done"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] docker exec -d $container bash -lc '<traffic loop for $profile>'"
    started=$((started + 1))
    continue
  fi

  # Avoid self-termination by matching anchored legacy loop patterns and the
  # explicit marker used by current loops.
  docker exec "$container" bash -lc "pkill -f '^bash -lc while true; do iface=\\$\\(ip -o link show' >/dev/null 2>&1 || true; pkill -f '^bash -lc ORANSLICE_TRAFFIC_LOOP=1; while true' >/dev/null 2>&1 || true; pkill -x ping >/dev/null 2>&1 || true"
  docker exec -d "$container" bash -lc "${traffic_cmd}"
  started=$((started + 1))
done < <(tail -n +2 "$PROFILE_CSV")

echo "Traffic launch complete: started=$started skipped_not_running=$skipped"
