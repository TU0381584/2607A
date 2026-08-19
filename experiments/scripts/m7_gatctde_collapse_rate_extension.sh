#!/usr/bin/env bash
# M7's collapse-reliability half: docs/PAPER5_M6_topology.md Part 9 found
# GAT-CTDE's own N=19 collapse rate did not clearly replicate between the
# primary 12-seed sample (31%, 11/36) and the 3-seed independent
# replication (78%, 7/9) -- flagged as needing more seeds specifically
# aimed at pinning down that number, not the broader topology-sparsity/
# reward-margin questions Part 8 already found weak or null. This adds 6
# NEW seeds (2000-2005, disjoint from both 900-929 and 1000-1029), all
# three topologies, gat_ctde ONLY (single_agent_dqn's 15/15 total
# collapse and independent_dqn's topology-invariant mediocre profile are
# already well-supported and not what this run is trying to narrow) --
# cuts the cost roughly 3x versus a full 3-arm campaign. Reuses the SAME
# n19_*_capfix output directories the primary sample lives in (not a new
# directory), so this genuinely extends that sample rather than starting
# a third, separate one.
#
# Per-seed-per-topology cost for gat_ctde alone at full 300/50-episode
# budget, N=19: ~414s (experiments/logs/m6_timing_probe_n19.log). 6 seeds
# x 3 topologies x ~414s =~ 7450s (~2.1h) estimated.
#
# Launched ONLY after explicitly confirming via pgrep that no other
# m6_run_experiment.py process is running anywhere -- the root cause of
# BOTH earlier corruption incidents (Parts 6-7) was exactly this check
# being skipped/invalidated by an orphaned concurrent process.
set -euo pipefail

cd "$(dirname "$0")/../../framework"
PY=../venv/bin/python3
SCRIPT=../experiments/scripts/m6_run_experiment.py
OUT_ROOT=../experiments/results/m6_pilot
SEEDS="2000 2001 2002 2003 2004 2005"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix"
  echo "=== [m7-gatctde-ext] $tag (seeds 2000-2005, gat_ctde only) ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --arms gat_ctde \
    --resume-seeds
  echo "=== [m7-gatctde-ext] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m7-gatctde-ext] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
