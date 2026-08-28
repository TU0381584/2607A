#!/usr/bin/env bash
# Fourth collapse-rate-narrowing extension, per direct user request ("run
# another seed batch for 9 more hours") continuing the same line of work
# as extensions 2-3 (docs/PAPER5_M6_topology.md Parts 11-12 /
# Papers_4-5/Paper_5/WPC/main.tex Section X-D), which combined primary (900-911),
# replication (1000-1002), and extension 2000-2024 (40 seeds, 120 cells)
# into 37.5% [25.0%, 50.8%].
#
# Adds 18 MORE seeds (2025-2042, disjoint from every prior sample), all
# three topologies, gat_ctde only (the other two arms' profiles are
# already well-supported and not what needs narrowing). Reuses the SAME
# n19_*_capfix output directories every prior sample lives in.
#
# Per-seed-per-topology cost for gat_ctde alone at full 300/50-episode
# budget, N=19, per the two prior extensions' own measured timing:
# extension2 14834s/27 cells = ~549s/cell average (up to 713s/cell for
# ring); extension3 13595s/30 cells = ~453s/cell, consistent across all
# three topologies. Blended average ~500s/cell. 18 seeds x 3 topologies
# x ~500-549s =~ 27000-29646s (~7.5-8.2h) estimated -- sized to fit
# within a 9-hour budget with margin even at the slower end of the
# observed range.
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
SEEDS="2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix"
  echo "=== [m6-gatctde-ext4] $tag (seeds 2025-2042, gat_ctde only) ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --arms gat_ctde \
    --resume-seeds
  echo "=== [m6-gatctde-ext4] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-gatctde-ext4] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
