#!/usr/bin/env bash
# Retry of the 3 arms that failed in run_stage5_v2_trial.sh's first pass
# (dqn_sla/dqn_qoe/static_at_cap), after fixing two real, unrelated
# problems found along the way: health_check.sh's segfault check was
# matching an UNRELATED iperf3 traffic-client crash (not a RAN process
# crash) and burning restart attempts that could never fix it (fixed,
# see health_check.sh's 2026-07-29 comment); and iperf3-target's port
# 5201 server had wedged again (known failure mode, same fix: recreate
# the container). baseline already completed successfully in the first
# pass and is not re-run here.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2_trial
SEED=950
EPISODES=2

CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap_v2.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
DQN_QOE_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

run_arm() {
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

echo "=== $(date +%H:%M:%S) STAGE5 V2 TRIAL RETRY START ==="
run_arm dqn_sla       dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT"
run_arm dqn_qoe       dqn             qoe "$CAMPAIGN_CFG" "$DQN_QOE_CKPT"
run_arm static_at_cap baseline_static sla "$STATIC_CFG"   ""
echo "=== $(date +%H:%M:%S) STAGE5 V2 TRIAL RETRY COMPLETE ==="
