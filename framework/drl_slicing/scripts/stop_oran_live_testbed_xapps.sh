#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
GEN_DIR="$DOCKER_DIR/generated"

KEEP_CORE=0
DRY_RUN=0
GNB_MODE="both"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-core)
      KEEP_CORE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --gnb-mode)
      GNB_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--keep-core] [--dry-run] [--gnb-mode ueransim|oai|both]"
      exit 2
      ;;
  esac
done

if [[ "$GNB_MODE" != "ueransim" && "$GNB_MODE" != "oai" && "$GNB_MODE" != "both" ]]; then
  echo "--gnb-mode must be ueransim, oai, or both"
  exit 2
fi

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $cmd"
  else
    eval "$cmd"
  fi
}

echo "Stopping ORAN live run"

echo "[1/5] Stop UE traffic loops"
run_cmd "cd '$DRL_DIR' && bash scripts/stop_ue_traffic_profiles.sh --profiles '$GEN_DIR/ue_fleet_profiles.csv'"

echo "[2/5] Stop xApp service processes"
for pid_file in "$DRL_DIR"/results/xapps/*.pid; do
  [[ -f "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  if [[ -n "$pid" ]]; then
    run_cmd "kill '$pid' >/dev/null 2>&1 || true"
  fi
  run_cmd "rm -f '$pid_file'"
done

echo "[3/5] Stop generated UE fleet"
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$GEN_DIR/nr-ue-fleet.yaml' down >/dev/null 2>&1 || true"
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$GEN_DIR/nr-ue-oai-fleet.yaml' down >/dev/null 2>&1 || true"
run_cmd "docker ps -a --format '{{.Names}}' | grep '^nr_ue_' | xargs -r docker rm -f >/dev/null 2>&1 || true"

echo "[4/5] Stop gNB stack(s)"
if [[ "$GNB_MODE" == "ueransim" || "$GNB_MODE" == "both" ]]; then
  run_cmd "cd '$DOCKER_DIR' && docker compose -f nr-gnb.yaml down >/dev/null 2>&1 || true"
fi
if [[ "$GNB_MODE" == "oai" || "$GNB_MODE" == "both" ]]; then
  run_cmd "cd '$DOCKER_DIR' && docker compose -f oaignb.yaml down >/dev/null 2>&1 || true"
fi

echo "[5/5] Stop Open5GS core"
if [[ "$KEEP_CORE" -eq 1 ]]; then
  echo "Keeping Open5GS core running (--keep-core set)"
else
  run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml down >/dev/null 2>&1 || true"
fi

echo "Stop sequence complete."
