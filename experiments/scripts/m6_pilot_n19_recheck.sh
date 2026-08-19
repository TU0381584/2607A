#!/usr/bin/env bash
# Re-run of the 3 N=19 combinations only, against the corrected
# saclb_offline_dqn_n19.yaml (max_pending_per_step 12->76, fixing the
# arrival-truncation confound found in docs/PAPER5_M6_topology.md Part 4).
# Written to NEW output dirs (n19_*_capfix) rather than overwriting the
# original (confounded) m6_pilot/n19_* results, so the original run stays
# available for comparison/audit rather than being silently replaced.
set -euo pipefail

cd "$(dirname "$0")/../../framework"
PY=../venv/bin/python3
SCRIPT=../experiments/scripts/m6_run_experiment.py
OUT_ROOT=../experiments/results/m6_pilot
SEEDS="900 901 902"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix"
  echo "=== [m6-n19-recheck] $tag ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --resume-seeds
  echo "=== [m6-n19-recheck] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-n19-recheck] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
