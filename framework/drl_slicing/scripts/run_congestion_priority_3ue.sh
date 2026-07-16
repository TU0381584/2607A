#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

CONFIG="${1:-configs/openran_live_prom_strict_3ue_congestion_600step.yaml}"
OUT_DIR="${2:-results/phase3/congestion_3ue_priority_600step_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
TRAFFIC_SEED="${TRAFFIC_SEED:-$SEED}"
HORIZON_STEPS="${HORIZON_STEPS:-600}"
DQN_CKPT="${DQN_CKPT:-results/offline_checkpoints/dqn_3ue_congestion.pt}"
A2C_CKPT="${A2C_CKPT:-results/offline_checkpoints/a2c_3ue_congestion.pt}"

CONTROLLERS=(dqn a2c rule_based)

if [[ "$DQN_CKPT" != /* ]]; then
  DQN_CKPT="$BASE_DIR/$DQN_CKPT"
fi
if [[ "$A2C_CKPT" != /* ]]; then
  A2C_CKPT="$BASE_DIR/$A2C_CKPT"
fi

if [[ ! -f "$DQN_CKPT" ]]; then
  echo "Missing offline DQN checkpoint: $DQN_CKPT"
  echo "Set DQN_CKPT to a pre-trained offline checkpoint before running live evaluation."
  exit 2
fi
if [[ ! -f "$A2C_CKPT" ]]; then
  echo "Missing offline A2C checkpoint: $A2C_CKPT"
  echo "Set A2C_CKPT to a pre-trained offline checkpoint before running live evaluation."
  exit 2
fi

RUNNING_UE_COUNT="$(docker ps --format '{{.Names}}' | grep -c '^nr_ue_' || true)"
if ! docker ps --format '{{.Names}}' | grep -q '^amf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^smf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^upf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^oaignb$' \
  || [[ "$RUNNING_UE_COUNT" -ne 3 ]]; then
  echo "Live 3-UE OAI stack not detected. Starting it now..."
  bash scripts/run_closed_loop_sim.sh --ue-count 3 --profile-mode triad --active-algo dqn
fi

echo "Verifying live ORAN E2E stack (5GC + RAN + UE traffic + sessions)..."
bash scripts/verify_live_e2e_stack.sh \
  --expected-ue-count 3 \
  --min-sessions 3 \
  --prom-url "http://127.0.0.1:9090" \
  --target-ip "12.1.1.1"

echo "Running congestion-priority simulation"
echo "  Config  : $CONFIG"
echo "  Output  : $OUT_DIR"
echo "  Seed    : $SEED"
echo "  Traffic : seeded live random ($TRAFFIC_SEED)"
echo "  Horizon : $HORIZON_STEPS"
echo "  DQN ckpt: $DQN_CKPT"
echo "  A2C ckpt: $A2C_CKPT"

for controller in "${CONTROLLERS[@]}"; do
  echo "Resetting live traffic for controller '$controller' with seed $TRAFFIC_SEED"
  bash scripts/start_ue_traffic_profiles.sh --target-ip "12.1.1.1" --seed "$TRAFFIC_SEED" >/dev/null
  sleep 2

  PYTHONPATH=. /usr/bin/python3 scripts/train_drl.py \
    --config "$CONFIG" \
    --controllers "$controller" \
    --controller-checkpoint "dqn=$DQN_CKPT" \
    --controller-checkpoint "a2c=$A2C_CKPT" \
    --require-inference-checkpoints \
    --seeds "$SEED" \
    --horizon-steps "$HORIZON_STEPS" \
    --output-dir "$OUT_DIR"
done

PYTHONPATH=. /usr/bin/python3 scripts/train_drl.py \
  --output-dir "$OUT_DIR" \
  --controllers "${CONTROLLERS[@]}" \
  --compare

PYTHONPATH=. /usr/bin/python3 scripts/plot_slice_priority_congestion.py \
  --output-dir "$OUT_DIR" \
  --controllers "${CONTROLLERS[@]}" \
  --seed "$SEED" \
  --ma-window 10 \
  --shock-start 200 \
  --shock-interval 200 \
  --shock-duration 20

PYTHONPATH=. /usr/bin/python3 scripts/plot_sla_superiority_live.py \
  --output-dir "$OUT_DIR" \
  --controllers "${CONTROLLERS[@]}" \
  --seed "$SEED" \
  --ma-window 10 \
  --shock-start 200 \
  --shock-interval 200 \
  --shock-duration 20

echo "Done. Congestion scenario outputs in: $OUT_DIR"
