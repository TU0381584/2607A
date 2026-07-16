#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_PROFILE_CSV="$ROOT_DIR/docker_open5gs/generated/ue_fleet_profiles.csv"

PROFILE_CSV="$DEFAULT_PROFILE_CSV"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profiles)
      PROFILE_CSV="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--profiles <csv>]"
      exit 2
      ;;
  esac
done

if [[ ! -f "$PROFILE_CSV" ]]; then
  echo "Profile CSV not found: $PROFILE_CSV"
  exit 1
fi

stopped=0

while IFS=, read -r _service container _rest; do
  [[ -z "$container" ]] && continue
  if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
    continue
  fi

  docker exec "$container" bash -lc "pkill -f '^bash -lc while true; do ping -I uesimtun0' >/dev/null 2>&1 || true; pkill -x ping >/dev/null 2>&1 || true"
  stopped=$((stopped + 1))
done < <(tail -n +2 "$PROFILE_CSV")

echo "Stopped UE traffic loops on $stopped containers"
