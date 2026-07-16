#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

OUT_DIR="${1:-results/ieee_access_rerun}"
HORIZON_STEPS="${HORIZON_STEPS:-1000}"
LIVE_CONFIG="${LIVE_CONFIG:-configs/openran_live_prom_true_3ue_triad.yaml}"
UE_COUNT="${UE_COUNT:-3}"
PROFILE_MODE="${PROFILE_MODE:-triad}"
ACTIVE_ALGO="${ACTIVE_ALGO:-dqn}"

if [[ $# -gt 1 ]]; then
  SEEDS=("${@:2}")
else
  SEEDS=(42 123 456 789 999 1001 1002 1003 1004 1005)
fi

echo "Running IEEE Access short rerun package"
echo "  Output dir : $OUT_DIR"
echo "  Horizon    : $HORIZON_STEPS"
echo "  Seeds      : ${SEEDS[*]}"
echo "  Config     : $LIVE_CONFIG"
echo "  UE count   : $UE_COUNT"
echo "  Profile    : $PROFILE_MODE"

RUNNING_UE_COUNT="$(docker ps --format '{{.Names}}' | grep -c '^nr_ue_' || true)"

if ! docker ps --format '{{.Names}}' | grep -q '^amf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^smf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^upf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^oaignb$' \
  || [[ "$RUNNING_UE_COUNT" -ne "$UE_COUNT" ]]; then
  echo "Live OAI stack not detected. Starting it now..."
  bash scripts/run_closed_loop_sim.sh --ue-count "$UE_COUNT" --profile-mode "$PROFILE_MODE" --active-algo "$ACTIVE_ALGO"
fi

PYTHONPATH=. python3 scripts/train_drl.py \
  --config "$LIVE_CONFIG" \
  --controllers dqn_train a2c_train ppo_train rule_based static random \
  --seeds "${SEEDS[@]}" \
  --horizon-steps "$HORIZON_STEPS" \
  --output-dir "$OUT_DIR" \
  --continue-on-error \
  --compare

echo "Done. Results are in: $OUT_DIR"
