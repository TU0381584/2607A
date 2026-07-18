#!/usr/bin/env bash
# Phase A: the full live evaluation campaign -- 5 arms x 3 seeds x 5
# episodes, via run_live_eval_arm.py's health-checked batch orchestration.
# Arm order is rotated per seed (not fixed) so no single arm always runs
# at the same point in the rig's drift/uptime curve -- the "interleave
# arm order across seeds" requirement. Ceilings reset + backlog drained
# (drain_backlog.sh) between every arm within a seed pass.
#
# Crash-safe progress: every completed (arm, seed) appends one line to
# PROGRESS_LOG (plain, append-only, independent of this script's own
# state) BEFORE moving on -- if this script or the whole session dies
# mid-campaign, PROGRESS_LOG is the ground truth for what's already done
# and this script can be restarted-and-skip-completed (manually, by
# reading PROGRESS_LOG) rather than losing accounting.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml
BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline.yaml
CKPT_ROOT=/home/kmanojp/oranslice_rig/experiments/results/offline
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign
PROGRESS_LOG=/home/kmanojp/oranslice_rig/experiments/results/live_campaign/PROGRESS.log
EPISODES_TOTAL=5
BATCH_SIZE=2
CKPT_SEED=256   # which offline-trained seed's checkpoint to evaluate (fixed across all 3 eval seeds/reps)

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

declare -A ALGO_OF=( [baseline]="baseline_static" [dqn_sla]="dqn" [a2c_sla]="a2c" [dqn_qoe]="dqn" [a2c_qoe]="a2c" )
declare -A MODE_OF=( [baseline]="sla" [dqn_sla]="sla" [a2c_sla]="sla" [dqn_qoe]="qoe" [a2c_qoe]="qoe" )

ARMS_BASE=(baseline dqn_sla a2c_sla dqn_qoe a2c_qoe)
SEEDS=(950 951 952)

rotate() {  # rotate($1=array-name-by-ref-via-echo, $2=n) -- prints rotated list
  local n="$2"
  local arr=("${ARMS_BASE[@]}")
  for ((i=0; i<n; i++)); do
    arr=("${arr[@]:1}" "${arr[0]}")
  done
  echo "${arr[@]}"
}

is_done() {  # is_done(arm, seed) -- checks PROGRESS_LOG for a prior DONE entry
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {
  local arm="$1" seed="$2" rotation_idx="$3"
  local algo="${ALGO_OF[$arm]}" mode="${MODE_OF[$arm]}"

  if is_done "$arm" "$seed"; then
    echo "[campaign] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
    return 0
  fi

  echo "=== $(date +%H:%M:%S) DRAIN before arm=$arm seed=$seed (rotation_idx=$rotation_idx) ==="
  bash "$DRAIN" 2>&1 | tail -5

  local cfg ckpt
  if [[ "$arm" == "baseline" ]]; then
    cfg="$BASELINE_CFG"
    ckpt=""
  else
    cfg="$CAMPAIGN_CFG"
    ckpt="$CKPT_ROOT/$mode/seed${CKPT_SEED}/$algo/offline_closed_loop/rep_0/checkpoint.pt"
  fi

  echo "=== $(date +%H:%M:%S) RUN arm=$arm seed=$seed algo=$algo mode=$mode ==="
  local t0=$(date +%s)
  if [[ -z "$ckpt" ]]; then
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --episodes-total "$EPISODES_TOTAL" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --out-dir "$OUT_DIR"
  else
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --checkpoint "$ckpt" --episodes-total "$EPISODES_TOTAL" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --out-dir "$OUT_DIR"
  fi
  local rc=$?
  local t1=$(date +%s)
  local elapsed=$((t1 - t0))

  if [[ $rc -eq 0 ]]; then
    echo "DONE arm=$arm seed=$seed elapsed_s=$elapsed rotation_idx=$rotation_idx ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "=== $(date +%H:%M:%S) DONE arm=$arm seed=$seed (${elapsed}s) ==="
  else
    echo "FAILED arm=$arm seed=$seed elapsed_s=$elapsed rc=$rc ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "!!! FAILED arm=$arm seed=$seed (rc=$rc) !!!"
  fi
}

echo "=== $(date +%H:%M:%S) PHASE A CAMPAIGN START ==="
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  read -ra arm_order <<< "$(rotate ARMS_BASE "$idx")"
  echo "=== seed=$seed arm_order=${arm_order[*]} ==="
  for arm in "${arm_order[@]}"; do
    run_one "$arm" "$seed" "$idx"
  done
done
echo "=== $(date +%H:%M:%S) PHASE A CAMPAIGN COMPLETE ==="
