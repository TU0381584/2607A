#!/usr/bin/env bash
# Independent-seed replication of the N=19 finding (docs/PAPER5_M6_topology.md
# Part 8): seeds 1000-1002, disjoint from the 900-911 primary sample, all
# three topologies. Tests whether single-agent DQN's total collapse and
# GAT-CTDE's own partial collapse rate / precision-not-reward-margin
# advantage hold on seeds this analysis was never tuned against -- this
# project's own established reproduction-vs-replication discipline
# (same distinction that caught the M4 churn-immunity overclaim).
# Launched only after explicitly confirming no other m6_run_experiment.py
# process is running (pgrep check immediately before this script starts).
set -euo pipefail

cd "$(dirname "$0")/../../framework"
PY=../venv/bin/python3
SCRIPT=../experiments/scripts/m6_run_experiment.py
OUT_ROOT=../experiments/results/m6_pilot
SEEDS="1000 1001 1002"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix_replication"
  echo "=== [m6-n19-replication] $tag (seeds 1000-1002) ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --resume-seeds
  echo "=== [m6-n19-replication] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-n19-replication] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
