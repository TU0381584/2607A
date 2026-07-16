#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

CONFIG="${1:-configs/openran_live_prom_true_3ue_triad.yaml}"
OUT_DIR="${2:-results/phaseB_eval}"

# 1) Load both models as xApps (single run warm start)
bash scripts/load_xapps.sh "$CONFIG" 1

# 2) Wait for xApps to finish this run
DQN_PID="$(cat results/xapps/xapp-dqn.pid)"
A2C_PID="$(cat results/xapps/xapp-a2c.pid)"

while kill -0 "$DQN_PID" 2>/dev/null || kill -0 "$A2C_PID" 2>/dev/null; do
  sleep 1
done

echo "xApps finished initial run. Starting evaluation + plotting..."

# 3) Evaluate and plot
PYTHONPATH=. python3 scripts/evaluate_and_plot.py \
  --config "$CONFIG" \
  --output-dir "$OUT_DIR" \
  --seeds 42 123 456

echo "Done. See: $OUT_DIR and $OUT_DIR/plots"
