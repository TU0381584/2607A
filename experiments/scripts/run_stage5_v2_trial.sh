#!/usr/bin/env bash
# Stage 5 v2 recalibration -- ~1h smoke trial across all 4 arms (user:
# "do the full run, but in 1 hour. we'll do the 6 hour one later").
# Mirrors run_phase3_trial30.sh's precedent (real cadence, real configs
# and checkpoints, just fewer episodes) -- NOT the final statistically
# powered re-run; that's the deferred 3-seed x 5-episode x 4-arm session.
# 1 seed (950), 2 episodes/arm, all 4 arms under the corrected (_v2)
# calibration + (for dqn_sla/dqn_qoe) the newly retrained checkpoints.
# Output routed to its own directory, never mixed with the v1 campaign.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2_trial
SEED=950
EPISODES=2

BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline_v2.yaml
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap_v2.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
DQN_QOE_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

run_arm() {  # run_arm(arm, algo, mode, config, checkpoint_or_empty)
  local arm="$1" algo="$2" mode="$3" cfg="$4" ckpt="$5"
  echo "=== $(date +%H:%M:%S) DRAIN before arm=$arm ==="
  bash "$DRAIN" 2>&1 | tail -5
  echo "=== $(date +%H:%M:%S) RUN arm=$arm algo=$algo mode=$mode ==="
  if [[ -z "$ckpt" ]]; then
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --episodes-total "$EPISODES" --batch-size "$EPISODES" \
      --seed "$SEED" --out-dir "$OUT_DIR"
  else
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --checkpoint "$ckpt" --episodes-total "$EPISODES" --batch-size "$EPISODES" \
      --seed "$SEED" --out-dir "$OUT_DIR"
  fi
}

echo "=== $(date +%H:%M:%S) STAGE5 V2 TRIAL START ==="
run_arm baseline      baseline_static sla "$BASELINE_CFG" ""
run_arm dqn_sla       dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT"
run_arm dqn_qoe       dqn             qoe "$CAMPAIGN_CFG" "$DQN_QOE_CKPT"
run_arm static_at_cap baseline_static sla "$STATIC_CFG"   ""
echo "=== $(date +%H:%M:%S) STAGE5 V2 TRIAL COMPLETE ==="
