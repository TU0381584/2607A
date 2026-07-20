#!/usr/bin/env bash
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
cd /home/kmanojp/oranslice_rig/framework

OUT_BASE=/home/kmanojp/oranslice_rig/experiments/results/admission_efficiency_live_offline
SEEDS="256 257 258"
EPISODES=300

for algo in dqn a2c; do
  for mode in sla qoe; do
    for seed in $SEEDS; do
      echo "=== $(date +%H:%M:%S) training algo=$algo mode=$mode seed=$seed ==="
      python3 /home/kmanojp/oranslice_rig/experiments/scripts/train_offline_admission_efficiency_live.py \
        --algorithm "$algo" --reward-mode "$mode" --episodes "$EPISODES" --seed "$seed" \
        --results-dir "$OUT_BASE" \
        > "$OUT_BASE/log_${algo}_${mode}_seed${seed}.log" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "!!! FAILED: algo=$algo mode=$mode seed=$seed (rc=$rc) !!!"
      else
        echo "--- done: algo=$algo mode=$mode seed=$seed ---"
      fi
    done
  done
done
echo "=== admission-efficiency LIVE-TRANSFERABLE (3-seed) training ALL DONE $(date +%H:%M:%S) ==="
