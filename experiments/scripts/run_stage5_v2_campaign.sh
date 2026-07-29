#!/usr/bin/env bash
# Stage 5 v2 campaign -- the "3 hour" run (user: "proceed with the longer
# run at 3 hours, no need for 6 hours"). Full 4-arm protocol (baseline,
# dqn_sla, dqn_qoe, static_at_cap) under the corrected calibration +
# retrained checkpoints, across 3 REAL seeds (950/951/952) -- not just
# the 1-seed/2-episode directional trial -- so seed-to-seed real-hardware
# variability is captured, the exact thing that made the 1-hour trial's
# single-seed draw unreliable to generalize from.
#
# Scope cut from the "full" 3-seed x 5-episode x 4-arm protocol (~5-6h)
# to 3-seed x 2-episode x 4-arm (~2.5-3h) to fit the requested budget --
# episodes per seed reduced, not seed count, so real cross-seed
# replication is preserved (the thing n=2/1-seed could not give); only
# statistical power per seed is reduced relative to the eventual
# full-power run.
#
# Same crash-safe PROGRESS_LOG discipline, arm-rotation-per-seed, and
# drain-between-arms discipline as run_phase_a_campaign.sh /
# run_phase3_static_at_cap.sh. Output routed to its own directory
# (live_campaign_v2), never mixed with the v1 campaign or the v2 trial.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2
PROGRESS_LOG="$OUT_DIR/PROGRESS.log"
EPISODES_TOTAL=2
BATCH_SIZE=2

BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline_v2.yaml
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap_v2.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
DQN_QOE_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

declare -A ALGO_OF=( [baseline]="baseline_static" [dqn_sla]="dqn" [dqn_qoe]="dqn" [static_at_cap]="baseline_static" )
declare -A MODE_OF=( [baseline]="sla" [dqn_sla]="sla" [dqn_qoe]="qoe" [static_at_cap]="sla" )
declare -A CFG_OF=( [baseline]="$BASELINE_CFG" [dqn_sla]="$CAMPAIGN_CFG" [dqn_qoe]="$CAMPAIGN_CFG" [static_at_cap]="$STATIC_CFG" )
declare -A CKPT_OF=( [baseline]="" [dqn_sla]="$DQN_SLA_CKPT" [dqn_qoe]="$DQN_QOE_CKPT" [static_at_cap]="" )

ARMS_BASE=(baseline dqn_sla dqn_qoe static_at_cap)
SEEDS=(950 951 952)

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

rotate() {
  local n="$2"
  local arr=("${ARMS_BASE[@]}")
  for ((i=0; i<n; i++)); do
    arr=("${arr[@]:1}" "${arr[0]}")
  done
  echo "${arr[@]}"
}

is_done() {
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {
  local arm="$1" seed="$2" rotation_idx="$3"
  local algo="${ALGO_OF[$arm]}" mode="${MODE_OF[$arm]}" cfg="${CFG_OF[$arm]}" ckpt="${CKPT_OF[$arm]}"

  if is_done "$arm" "$seed"; then
    echo "[campaign] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
    return 0
  fi

  echo "=== $(date +%H:%M:%S) DRAIN before arm=$arm seed=$seed (rotation_idx=$rotation_idx) ==="
  bash "$DRAIN" 2>&1 | tail -5

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

echo "=== $(date +%H:%M:%S) STAGE5 V2 CAMPAIGN START ==="
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  read -ra arm_order <<< "$(rotate ARMS_BASE "$idx")"
  echo "=== seed=$seed arm_order=${arm_order[*]} ==="
  for arm in "${arm_order[@]}"; do
    run_one "$arm" "$seed" "$idx"
  done
done
echo "=== $(date +%H:%M:%S) STAGE5 V2 CAMPAIGN COMPLETE ==="
