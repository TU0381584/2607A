#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

CONFIG="${1:-configs/openran_live_prom_true_3ue_triad.yaml}"
OUT_DIR="${2:-results/online_ran_compare}"
UE_COUNT="${UE_COUNT:-3}"
PROFILE_MODE="${PROFILE_MODE:-triad}"
ACTIVE_ALGO="${ACTIVE_ALGO:-dqn}"

# Use 10 seeds by default for publication-grade significance
SEEDS=(42 123 456 789 999 1001 1002 1003 1004 1005)

RUNNING_UE_COUNT="$(docker ps --format '{{.Names}}' | grep -c '^nr_ue_' || true)"

if ! docker ps --format '{{.Names}}' | grep -q '^amf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^smf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^upf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^oaignb$' \
  || [[ "$RUNNING_UE_COUNT" -ne "$UE_COUNT" ]]; then
  echo "Live OAI stack not detected. Starting it now..."
  bash scripts/run_closed_loop_sim.sh --ue-count "$UE_COUNT" --profile-mode "$PROFILE_MODE" --active-algo "$ACTIVE_ALGO"
fi

echo "[1/5] Running DQN only (live OAI E2E)..."
PYTHONPATH=. python3 scripts/train_drl.py \
  --config "$CONFIG" \
  --algorithms dqn \
  --seeds "${SEEDS[@]}" \
  --output-dir "$OUT_DIR" \
  --train

echo "[2/5] Running A2C only (live OAI E2E)..."
PYTHONPATH=. python3 scripts/train_drl.py \
  --config "$CONFIG" \
  --algorithms a2c \
  --seeds "${SEEDS[@]}" \
  --output-dir "$OUT_DIR" \
  --train

echo "[3/5] Creating aggregate comparison..."
PYTHONPATH=. python3 scripts/train_drl.py \
  --output-dir "$OUT_DIR" \
  --compare

echo "[4/5] Plotting comparison bars..."
PYTHONPATH=. python3 scripts/evaluate_and_plot.py \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --seeds "${SEEDS[@]}" \
  --skip-run

echo "[5/5] Plotting convergence over time..."
PYTHONPATH=. python3 scripts/plot_convergence.py \
  --output-dir "$OUT_DIR" \
  --mode train

echo "Done. Results: $OUT_DIR"
