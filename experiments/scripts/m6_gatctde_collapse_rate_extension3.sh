#!/usr/bin/env bash
# Third collapse-rate-narrowing extension, per direct user request ("run
# more seeds for GAT-CTDE, up to 6 hours of compute time") continuing the
# same line of work as extension2 (docs/PAPER5_M6_topology.md Part 11 /
# Paper5/WPC/main.tex Section X-D), which combined primary (900-911),
# replication (1000-1002), and extension 2000-2014 (30 seeds, 90 cells)
# into 36.7% [22.2%, 52.2%].
#
# Adds 10 MORE seeds (2015-2024, disjoint from every prior sample), all
# three topologies, gat_ctde only (the other two arms' profiles are
# already well-supported and not what needs narrowing). Reuses the SAME
# n19_*_capfix output directories every prior sample lives in.
#
# Per-seed-per-topology cost for gat_ctde alone at full 300/50-episode
# budget, N=19, per extension2's own measured timing: 14834s for 9 seeds
# x 3 topologies = ~549s/cell average (range ~447-713s/cell across
# topologies). 10 seeds x 3 topologies x ~549-713s =~ 16470-21390s
# (~4.6-5.9h) estimated -- sized to fit within a 6-hour budget with
# margin even at the slower end of that range.
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
SEEDS="2015 2016 2017 2018 2019 2020 2021 2022 2023 2024"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix"
  echo "=== [m6-gatctde-ext3] $tag (seeds 2015-2024, gat_ctde only) ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --arms gat_ctde \
    --resume-seeds
  echo "=== [m6-gatctde-ext3] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-gatctde-ext3] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
