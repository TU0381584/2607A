#!/usr/bin/env bash
# Companion teardown for run_saclb_live_testbed.sh. Mirrors
# drl_slicing/scripts/stop_oran_live_testbed_xapps.sh's pattern, plus
# stopping the natively-run nr-softmodem gNB via the pid file
# run_saclb_live_testbed.sh writes to results/live/gnb_nr-softmodem.pid.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
GEN_DIR="$DOCKER_DIR/generated"
RESULTS_DIR="$ROOT_DIR/qoe_oran_framework/results/live"

KEEP_CORE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-core) KEEP_CORE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--keep-core] [--dry-run]"
      exit 2
      ;;
  esac
done

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $cmd"
  else
    eval "$cmd"
  fi
}

echo "Stopping Stage Zero live testbed"

echo "[1/4] Stop native gNB"
GNB_PID_FILE="$RESULTS_DIR/gnb_nr-softmodem.pid"
if [[ -f "$GNB_PID_FILE" ]]; then
  gnb_pid="$(cat "$GNB_PID_FILE")"
  run_cmd "kill -TERM '$gnb_pid' >/dev/null 2>&1 || true"
  run_cmd "rm -f '$GNB_PID_FILE'"
else
  echo "No gNB pid file at $GNB_PID_FILE -- nothing to stop"
fi

echo "[2/4] Stop UE traffic loops"
run_cmd "cd '$DRL_DIR' && bash scripts/stop_ue_traffic_profiles.sh --profiles '$GEN_DIR/ue_fleet_profiles.csv' || true"

echo "[3/4] Stop generated UE fleet"
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$GEN_DIR/nr-ue-fleet.yaml' down >/dev/null 2>&1 || true"
run_cmd "docker ps -a --format '{{.Names}}' | grep '^nr_ue_' | xargs -r docker rm -f >/dev/null 2>&1 || true"

echo "[4/4] Stop Open5GS core"
if [[ "$KEEP_CORE" -eq 1 ]]; then
  echo "Keeping Open5GS core running (--keep-core set)"
else
  run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml down >/dev/null 2>&1 || true"
fi

echo "Stop sequence complete."
