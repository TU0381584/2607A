#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG="results/phase3/pure_oai_retry_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "SWEEP_LOG=$LOG"

CKPTS=(
  "results/offline_checkpoints/dqn_hybrid_eval_actionfix.pt|results/offline_checkpoints/a2c_hybrid_eval_actionfix.pt|0.20|0.45|0.35"
  "results/offline_checkpoints/dqn_pure_oai_quick_tuned_20260410_134931.pt|results/offline_checkpoints/a2c_pure_oai_quick_tuned_20260410_134931.pt|0.20|0.45|0.35"
  "results/offline_checkpoints/dqn_3ue_congestion.pt|results/offline_checkpoints/a2c_3ue_congestion.pt|0.20|0.45|0.35"
  "results/offline_checkpoints/dqn_3ue_live_finetune.pt|results/offline_checkpoints/a2c_3ue_live_finetune.pt|0.20|0.45|0.35"
)
SEEDS=(42 123 256)

found=0
for combo in "${CKPTS[@]}"; do
  IFS='|' read -r dqn a2c wb wd wa <<< "$combo"
  for seed in "${SEEDS[@]}"; do
    echo "===== seed=$seed dqn=$(basename "$dqn") a2c=$(basename "$a2c") weights=($wb,$wd,$wa) =====" | tee -a "$LOG"

    out=$(SEED="$seed" \
      HORIZON=18 \
      STEP_SECONDS=1 \
      DQN_CKPT="$dqn" \
      A2C_CKPT="$a2c" \
      HYBRID_BASE_WEIGHT="$wb" \
      HYBRID_DQN_WEIGHT="$wd" \
      HYBRID_A2C_WEIGHT="$wa" \
      bash scripts/run_pure_oai_retry_4way.sh)

    echo "$out" | tee -a "$LOG"
    all=$(echo "$out" | awk -F= '/all_beat_rule/{print $2}' | tail -n1)

    if [[ "$all" == "1" ]]; then
      echo "SUCCESS seed=$seed dqn=$dqn a2c=$a2c weights=$wb,$wd,$wa" | tee -a "$LOG"
      found=1
      break 2
    fi
  done
done

if [[ "$found" == "0" ]]; then
  echo "NO_SUCCESS_FOUND" | tee -a "$LOG"
fi

echo "SWEEP_COMPLETE=1"
