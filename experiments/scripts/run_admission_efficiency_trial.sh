#!/usr/bin/env bash
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
cd /home/kmanojp/oranslice_rig/framework

OUT_BASE=/home/kmanojp/oranslice_rig/experiments/results/admission_efficiency_offline
SEED=256
EPISODES=300

for algo in dqn a2c; do
  for mode in sla qoe; do
    echo "=== $(date +%H:%M:%S) training algo=$algo mode=$mode seed=$SEED ==="
    python3 /home/kmanojp/oranslice_rig/experiments/scripts/train_offline_admission_efficiency.py \
      --algorithm "$algo" --reward-mode "$mode" --episodes "$EPISODES" --seed "$SEED" \
      --results-dir "$OUT_BASE" \
      > "$OUT_BASE/log_${algo}_${mode}_seed${SEED}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "!!! FAILED: algo=$algo mode=$mode seed=$SEED (rc=$rc) !!!"
    else
      echo "--- done: algo=$algo mode=$mode seed=$SEED ---"
    fi
  done
done
echo "=== admission-efficiency trial ALL DONE $(date +%H:%M:%S) ==="
