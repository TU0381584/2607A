#!/usr/bin/env bash
# M6 pilot: full (N, topology) grid at 3 seeds/arm (matching M4's own
# established pilot precedent -- "3 seeds, full condition grid" -- before
# committing to the full 30-seed budget). N=3/fully_connected is skipped:
# that is the existing, already-committed M2 campaign, not new work.
# All arms run under gnb_load_multiplier_mode=default (the now-corrected
# understanding: this IS the heterogeneous-load condition M6's brief asks
# for -- see docs/PAPER5_M6_topology.md Part 2). homogeneous mode is for
# M7's later use, not this pilot.
set -euo pipefail

cd "$(dirname "$0")/../../framework"
PY=../venv/bin/python3
SCRIPT=../experiments/scripts/m6_run_experiment.py
OUT_ROOT=../experiments/results/m6_pilot
SEEDS="900 901 902"
TRAIN_EP=300
EVAL_EP=50

declare -a COMBOS=(
  "qoe_oran_framework/configs/saclb_offline_dqn_n7.yaml fully_connected n7_fully_connected"
  "qoe_oran_framework/configs/saclb_offline_dqn_n7.yaml ring n7_ring"
  "qoe_oran_framework/configs/saclb_offline_dqn_n7.yaml hex n7_hex"
  "qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml fully_connected n19_fully_connected"
  "qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml ring n19_ring"
  "qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml hex n19_hex"
)

t0=$(date +%s)
for combo in "${COMBOS[@]}"; do
  read -r config topology tag <<< "$combo"
  echo "=== [m6-pilot] $tag (config=$config, topology=$topology) ==="
  $PY $SCRIPT \
    --config-path "$config" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --resume-seeds
  echo "=== [m6-pilot] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-pilot] ALL COMBINATIONS DONE, total elapsed $(( $(date +%s) - t0 ))s"
