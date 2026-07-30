#!/usr/bin/env bash
# Stage 5 v2 top-up: completes the deferred "full 3-seed x 5-episode x
# 4-arm" live re-evaluation under the corrected (v2) calibration --
# docs/STAGE5_recalibration.md section 6, "user's own words: 'we'll do
# the 6 hour one later'". This is that later run, topping up rather than
# repeating what already exists in experiments/results/live_campaign_v2/:
#
#   static_at_cap_v2: already at n=21 (950/951/952 x2 + 953/954/955 x5) -- SKIP.
#   dqn_sla_v2:       already at n=16 (950/951/952 x2 + 953/954 x5) -- needs
#                     ONE more seed (955 x5) to reach n=21.
#   baseline_v2:      only at n=6 (950/951/952 x2) -- needs 953/954/955 x5
#                     (new seeds, full protocol) to reach n=21.
#   dqn_qoe_v2:       only at n=6 (950/951/952 x2) -- same, needs
#                     953/954/955 x5 to reach n=21.
#
# All four arms land at n=21 on a consistent seed set (950/951/952 at the
# 2-episode campaign depth, 953/954/955 at the full 5-episode depth),
# matching the precedent already set by static_at_cap_v2/dqn_sla_v2's own
# validation round. Same PROGRESS_LOG (shared with the campaign + validate
# scripts), same crash-safe skip-if-done, same drain-between-arms
# discipline as every prior live stage.
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
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
DQN_QOE_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

is_done() {
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {
  local arm="$1" algo="$2" mode="$3" cfg="$4" ckpt="$5" seed="$6"

  if is_done "$arm" "$seed"; then
    echo "[topup] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
    return 0
  fi

  echo "=== $(date +%H:%M:%S) DRAIN before arm=$arm seed=$seed ==="
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
    echo "DONE arm=$arm seed=$seed elapsed_s=$elapsed ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "=== $(date +%H:%M:%S) DONE arm=$arm seed=$seed (${elapsed}s) ==="
  else
    echo "FAILED arm=$arm seed=$seed elapsed_s=$elapsed rc=$rc ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "!!! FAILED arm=$arm seed=$seed (rc=$rc) !!!"
  fi
}

echo "=== $(date +%H:%M:%S) STAGE5 V2 TOPUP START ==="
run_one dqn_sla   dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT" 955
run_one baseline  baseline_static sla "$BASELINE_CFG"  ""              953
run_one dqn_qoe   dqn             qoe "$CAMPAIGN_CFG" "$DQN_QOE_CKPT" 953
run_one baseline  baseline_static sla "$BASELINE_CFG"  ""              954
run_one dqn_qoe   dqn             qoe "$CAMPAIGN_CFG" "$DQN_QOE_CKPT" 954
run_one baseline  baseline_static sla "$BASELINE_CFG"  ""              955
run_one dqn_qoe   dqn             qoe "$CAMPAIGN_CFG" "$DQN_QOE_CKPT" 955
echo "=== $(date +%H:%M:%S) STAGE5 V2 TOPUP COMPLETE ==="
