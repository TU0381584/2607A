#!/usr/bin/env bash
# Second collapse-rate-narrowing extension, per the paper's own flagged
# future-work item: docs/PAPER5_M6_topology.md Part 10 / Papers_4-5/Paper_5/WPC/main.tex
# Section X-D combined all three samples so far (primary 900-911,
# replication 1000-1002, extension 2000-2005 -- 21 seeds, 63 cells) into
# 46.0% [28.6%, 65.1%], a still-wide 36.5-point seed-level bootstrap CI.
# Adds 9 MORE seeds (2006-2014, disjoint from every prior sample), all
# three topologies, gat_ctde only (single_agent_dqn's 15/15 total collapse
# and independent_dqn's topology-invariant profile are already
# well-supported and not what needs narrowing). Reuses the SAME
# n19_*_capfix output directories every prior sample lives in.
#
# Per-seed-per-topology cost for gat_ctde alone at full 300/50-episode
# budget, N=19, per the two prior extensions' own measured timing:
# ~414-523s (average ~490s). 9 seeds x 3 topologies x ~490s =~ 13230s
# (~3.7h) estimated.
#
# Launched ONLY after explicitly confirming via pgrep that no other
# m6_run_experiment.py process is running anywhere -- the root cause of
# the two earlier corruption incidents (docs/PAPER5_M6_topology.md Parts
# 6-7) was exactly this check being skipped/invalidated by an orphaned
# concurrent process.
set -euo pipefail

cd "$(dirname "$0")/../../framework"
PY=../venv/bin/python3
SCRIPT=../experiments/scripts/m6_run_experiment.py
OUT_ROOT=../experiments/results/m6_pilot
SEEDS="2006 2007 2008 2009 2010 2011 2012 2013 2014"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix"
  echo "=== [m6-gatctde-ext2] $tag (seeds 2006-2014, gat_ctde only) ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --arms gat_ctde \
    --resume-seeds
  echo "=== [m6-gatctde-ext2] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-gatctde-ext2] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
