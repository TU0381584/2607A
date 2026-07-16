#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"

DRY_RUN=0
ACTIVE_ALGO="dqn"
UE_COUNT=3
PROFILE_MODE="auto"
FORCE_HIGH_LOAD=0
OAI_UE_IMAGE=""
OAI_LAUNCH_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --active-algo)
      ACTIVE_ALGO="$2"
      shift 2
      ;;
    --ue-count)
      UE_COUNT="$2"
      shift 2
      ;;
    --profile-mode)
      PROFILE_MODE="$2"
      shift 2
      ;;
    --force-high-load)
      FORCE_HIGH_LOAD=1
      shift
      ;;
    --oai-ue-image)
      OAI_UE_IMAGE="$2"
      shift 2
      ;;
    --oai-launch-mode)
      OAI_LAUNCH_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--active-algo dqn|a2c] [--ue-count N] [--profile-mode auto|ratio|triad] [--force-high-load] [--oai-ue-image IMAGE] [--oai-launch-mode env|local-build] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ "$ACTIVE_ALGO" != "dqn" && "$ACTIVE_ALGO" != "a2c" ]]; then
  echo "--active-algo must be dqn or a2c"
  exit 2
fi

if ! [[ "$UE_COUNT" =~ ^[0-9]+$ ]] || (( UE_COUNT <= 0 )); then
  echo "--ue-count must be a positive integer"
  exit 2
fi

if [[ "$PROFILE_MODE" != "auto" && "$PROFILE_MODE" != "ratio" && "$PROFILE_MODE" != "triad" ]]; then
  echo "--profile-mode must be auto, ratio, or triad"
  exit 2
fi

if [[ "$PROFILE_MODE" == "triad" && "$UE_COUNT" -ne 3 ]]; then
  echo "--profile-mode triad requires --ue-count 3"
  exit 2
fi

if [[ -n "$OAI_LAUNCH_MODE" && "$OAI_LAUNCH_MODE" != "env" && "$OAI_LAUNCH_MODE" != "local-build" ]]; then
  echo "--oai-launch-mode must be env or local-build"
  exit 2
fi

CMD=(
  bash "$DRL_DIR/scripts/run_oran_live_testbed_xapps.sh"
  --gnb-mode oai
  --ue-stack oai
  --ue-count "$UE_COUNT"
  --profile-mode "$PROFILE_MODE"
  --active-algo "$ACTIVE_ALGO"
)

if [[ "$FORCE_HIGH_LOAD" -eq 1 ]]; then
  CMD+=(--force-high-load)
fi

if [[ -n "$OAI_UE_IMAGE" ]]; then
  CMD+=(--oai-ue-image "$OAI_UE_IMAGE")
fi

if [[ -n "$OAI_LAUNCH_MODE" ]]; then
  CMD+=(--oai-launch-mode "$OAI_LAUNCH_MODE")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

echo "Launching closed-loop simulation preset"
echo "  gNB mode         : oai"
echo "  UE stack         : oai (native OAI NR-UE)"
echo "  Requested UEs    : $UE_COUNT"
echo "  Profile mode     : $PROFILE_MODE"
echo "  Active controller: $ACTIVE_ALGO"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "  Dry-run          : enabled"
fi

"${CMD[@]}"