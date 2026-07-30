#!/usr/bin/env bash
# Live reverification (~10h budget): settles the one open statistical
# question left by docs/STAGE5_recalibration.md section 5c -- the
# validation round found DQN-SLA 16/16 vs static_at_cap 19/21 fully-
# compliant episodes, Fisher exact p=0.495 (NOT significant), and
# explicitly flagged that closing this needs "more static_at_cap
# episodes... or accepting the directional replication". This adds a
# full new 5-seed x 5-episode block (956-960) to ALL FOUR arms (not just
# the two under the collapse-rate question), bringing baseline/dqn_qoe
# from n=21 (post-topup) to n=46, and dqn_sla/static_at_cap from n=21 to
# n=46 as well -- a consistent, well-powered n across all arms for a
# final live table, and a much stronger Fisher test on the
# collapse-rate question specifically.
#
# 5 seeds x 5 episodes x 4 arms = 100 episodes. At this campaign's own
# historical pace (~6 min/episode across every prior stage), ~10h.
#
# Same PROGRESS_LOG (shared with campaign/topup), same crash-safe
# skip-if-done, same drain-between-arms discipline as every prior live
# stage. Arm order rotated per seed (mirrors run_stage5_v2_campaign.sh)
# so no single arm systematically goes last/first every block.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2
PROGRESS_LOG="$OUT_DIR/PROGRESS.log"
EPISODES_TOTAL=5
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
SEEDS=(956 957 958 959 960)

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
    echo "[reverify10h] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
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

echo "=== $(date +%H:%M:%S) STAGE5 V2 REVERIFY-10H START ==="
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  read -ra arm_order <<< "$(rotate ARMS_BASE "$idx")"
  echo "=== seed=$seed arm_order=${arm_order[*]} ==="
  for arm in "${arm_order[@]}"; do
    run_one "$arm" "$seed" "$idx"
  done
done
echo "=== $(date +%H:%M:%S) STAGE5 V2 REVERIFY-10H COMPLETE ==="
