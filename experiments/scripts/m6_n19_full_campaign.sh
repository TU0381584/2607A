#!/usr/bin/env bash
# Extension of the N=19 collapse-resistance investigation from 6 to 12
# seeds (900-911), all three topologies. Launched ONLY after explicitly
# confirming via pgrep that no other m6_run_experiment.py process is
# running anywhere -- the root cause of BOTH corruption incidents in
# docs/PAPER5_M6_topology.md Parts 6-7 was exactly this check being
# skipped/invalidated by an orphaned concurrent process. Do not launch
# any other job against these same output dirs while this runs.
# --resume-seeds reuses already-verified-clean data: gat_ctde's
# fully_connected 900-911 is already complete (900-905 from the capfix
# recheck, 906-911 from the orphan's leftover work, verified clean in
# Part 7) so this adds close to zero new gat_ctde/fully_connected
# compute; independent_dqn/single_agent_dqn/fully_connected and all
# three arms for ring/hex need seeds 906-911 fresh.
set -euo pipefail

cd "$(dirname "$0")/../../framework"
PY=../venv/bin/python3
SCRIPT=../experiments/scripts/m6_run_experiment.py
OUT_ROOT=../experiments/results/m6_pilot
SEEDS="900 901 902 903 904 905 906 907 908 909 910 911"
TRAIN_EP=300
EVAL_EP=50
CONFIG=qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml

t0=$(date +%s)
for topology in fully_connected ring hex; do
  tag="n19_${topology}_capfix"
  echo "=== [m6-n19-full] $tag (12 seeds) ==="
  $PY $SCRIPT \
    --config-path "$CONFIG" --topology "$topology" \
    --seeds $SEEDS --train-episodes $TRAIN_EP --eval-episodes $EVAL_EP \
    --out-dir "$OUT_ROOT/$tag" \
    --resume-seeds
  echo "=== [m6-n19-full] $tag done, elapsed so far: $(( $(date +%s) - t0 ))s ==="
done

echo "[m6-n19-full] ALL DONE, total elapsed $(( $(date +%s) - t0 ))s"
