#!/usr/bin/env bash
# 30-minute Phase 3 END-TO-END trial: 1 FULL-LENGTH episode (60 steps x 5s
# = 5 min, the real campaign's episode length) per arm, across all 5 arms,
# through the real orchestrator (run_live_eval_arm.py) using the REAL
# campaign config (not the compressed trial variant) and the real
# seed-256 checkpoints. Unlike the earlier 90s-episode pipeline smoke
# test, this run's numbers ARE at real campaign scale (just n=1
# episode/arm) -- a genuine preview of what the full 8h run will look
# like, not just a wiring check.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml
BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline.yaml
CKPT_ROOT=/home/kmanojp/oranslice_rig/experiments/results/offline
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_trial30
TRIAL_SEED=950

echo "=== $(date +%H:%M:%S) TRIAL30: baseline ==="
python3 "$ORCH" --arm baseline --algorithm baseline_static --reward-mode sla \
  --config "$BASELINE_CFG" --episodes-total 1 --batch-size 1 --seed "$TRIAL_SEED" --out-dir "$OUT_DIR"

for algo in dqn a2c; do
  for mode in sla qoe; do
    arm="${algo}_${mode}"
    ckpt="$CKPT_ROOT/$mode/seed256/$algo/offline_closed_loop/rep_0/checkpoint.pt"
    echo "=== $(date +%H:%M:%S) TRIAL30: $arm (checkpoint=$ckpt) ==="
    if [[ ! -f "$ckpt" ]]; then
      echo "!!! checkpoint not found: $ckpt -- skipping $arm !!!"
      continue
    fi
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$CAMPAIGN_CFG" --checkpoint "$ckpt" \
      --episodes-total 1 --batch-size 1 --seed "$TRIAL_SEED" --out-dir "$OUT_DIR"
  done
done

echo "=== $(date +%H:%M:%S) TRIAL30 COMPLETE ==="
