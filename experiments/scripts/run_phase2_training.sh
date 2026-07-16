#!/usr/bin/env bash
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh
cd /home/kmanojp/oranslice_rig/framework

CONFIG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_campaign.yaml
OUT_BASE=/home/kmanojp/oranslice_rig/experiments/results/offline
SEEDS="256 257 258"
EPISODES=300

for algo in dqn a2c; do
  for mode in sla qoe; do
    for seed in $SEEDS; do
      out_dir="$OUT_BASE/${mode}/seed${seed}"
      echo "=== $(date +%H:%M:%S) training algo=$algo mode=$mode seed=$seed -> $out_dir ==="
      python3 -m qoe_oran_framework.scripts.train_offline \
        --algorithm "$algo" --config "$CONFIG" \
        --episodes "$EPISODES" --seed "$seed" --reward-mode "$mode" \
        --results-dir "$out_dir" \
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
echo "=== Phase 2 offline training ALL DONE $(date +%H:%M:%S) ==="
