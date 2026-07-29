#!/usr/bin/env bash
# Stage 5 v2 validation round: settles the one open question flagged in
# docs/STAGE5_recalibration.md -- does DQN's collapse-avoidance edge over
# static_at_cap (established in Stage 3, p=0.0149, under the OLD
# calibration) still hold under the v2 (corrected calibration + retrained
# checkpoints) setup? The 3h campaign's n=6/arm sample was consistent
# with either answer (0/6 collapses for static_at_cap is plausible even
# if its true ~27% collapse rate is unchanged).
#
# Adds NEW seeds (mirrors Stage 3's own reverification precedent:
# completing a 3rd seed + testing on never-before-used seeds) at the
# FULL 5-episode protocol, not the campaign's 2-episode compression:
#   - static_at_cap_v2: seeds 953, 954, 955 (the arm most likely to show
#     a collapse; brings it to n=21 combined with the existing 6)
#   - dqn_sla_v2: seeds 953, 954 (brings it to n=16 combined with the
#     existing 6) -- the comparison arm for a Fisher exact test against
#     static_at_cap_v2's combined record.
#
# Same crash-safe PROGRESS_LOG (shared with the 3h campaign's, so
# resuming/skipping already-done work is automatic), drain-between-arms,
# health-checked batch discipline as every prior live stage.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2
PROGRESS_LOG="$OUT_DIR/PROGRESS.log"
EPISODES_TOTAL=5
BATCH_SIZE=2

CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap_v2.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

is_done() {
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {
  local arm="$1" algo="$2" mode="$3" cfg="$4" ckpt="$5" seed="$6"

  if is_done "$arm" "$seed"; then
    echo "[validate] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
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

echo "=== $(date +%H:%M:%S) STAGE5 V2 VALIDATE START ==="
run_one static_at_cap baseline_static sla "$STATIC_CFG"   "" 953
run_one static_at_cap baseline_static sla "$STATIC_CFG"   "" 954
run_one static_at_cap baseline_static sla "$STATIC_CFG"   "" 955
run_one dqn_sla       dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT" 953
run_one dqn_sla       dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT" 954
echo "=== $(date +%H:%M:%S) STAGE5 V2 VALIDATE COMPLETE ==="
