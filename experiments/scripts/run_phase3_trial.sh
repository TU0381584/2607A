#!/usr/bin/env bash
# 20-minute Phase 3 TRIAL: exercises the full live-eval pipeline (health
# checks, orchestration, all 5 arms, checkpoint loading) at short-episode
# scale before committing to the full ~6.5h campaign. Uses the *_trial.yaml
# config variants (30 steps x 3s = 90s/episode instead of 60 x 5s = 5min)
# -- NOT the real campaign's episode length; this run's numbers are a
# pipeline smoke test only, not evidence for the paper.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_trial.yaml
BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline_trial.yaml
CKPT_ROOT=/home/kmanojp/oranslice_rig/experiments/results/offline
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_trial
TRIAL_SEED=900

echo "=== $(date +%H:%M:%S) TRIAL: baseline ==="
python3 "$ORCH" --arm baseline --algorithm baseline_static --reward-mode sla \
  --config "$BASELINE_CFG" --episodes-total 1 --batch-size 1 --seed "$TRIAL_SEED" --out-dir "$OUT_DIR"

for algo in dqn a2c; do
  for mode in sla qoe; do
    arm="${algo}_${mode}"
    ckpt="$CKPT_ROOT/$mode/seed256/$algo/offline_closed_loop/rep_0/checkpoint.pt"
    echo "=== $(date +%H:%M:%S) TRIAL: $arm (checkpoint=$ckpt) ==="
    if [[ ! -f "$ckpt" ]]; then
      echo "!!! checkpoint not found: $ckpt -- skipping $arm !!!"
      continue
    fi
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$CAMPAIGN_CFG" --checkpoint "$ckpt" \
      --episodes-total 1 --batch-size 1 --seed "$TRIAL_SEED" --out-dir "$OUT_DIR"
  done
done

echo "=== $(date +%H:%M:%S) TRIAL COMPLETE ==="
