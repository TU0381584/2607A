#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

CONFIG="${1:-configs/openran_live_prom_true_3ue_triad.yaml}"
REPEAT="${2:-1}"
UE_COUNT="${UE_COUNT:-3}"
ACTIVE_ALGO="${ACTIVE_ALGO:-dqn}"

if ! docker ps --format '{{.Names}}' | grep -q '^amf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^smf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^upf$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^oaignb$' \
  || ! docker ps --format '{{.Names}}' | grep -q '^nr_ue_'; then
  echo "Live OAI stack not detected. Starting it now..."
  bash scripts/run_closed_loop_sim.sh --ue-count "$UE_COUNT" --active-algo "$ACTIVE_ALGO"
fi

mkdir -p results/xapps

PYTHONPATH=. nohup python3 scripts/xapp_service.py \
  --config "$CONFIG" \
  --algorithm dqn \
  --train \
  --repeat "$REPEAT" \
  --name xapp-dqn-train \
  > results/xapps/xapp-dqn.log 2>&1 &
DQN_PID=$!

PYTHONPATH=. nohup python3 scripts/xapp_service.py \
  --config "$CONFIG" \
  --algorithm a2c \
  --train \
  --repeat "$REPEAT" \
  --name xapp-a2c-train \
  > results/xapps/xapp-a2c.log 2>&1 &
A2C_PID=$!

echo "$DQN_PID" > results/xapps/xapp-dqn.pid
echo "$A2C_PID" > results/xapps/xapp-a2c.pid

echo "Started xApps:"
echo "  DQN PID: $DQN_PID"
echo "  A2C PID: $A2C_PID"
echo "Logs: results/xapps/xapp-dqn.log, results/xapps/xapp-a2c.log"
