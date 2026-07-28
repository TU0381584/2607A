#!/usr/bin/env bash
# ~20-min smoke trial for the Stage 3 static_at_cap arm, mirroring the
# precedent set by run_phase3_trial30.sh (real cadence, real config, just
# fewer episodes) before committing rig time to the full 3-seed x 5-episode
# protocol. Real saclb_campaign_static_at_cap.yaml (60 steps x 5s = 5
# min/episode, same as the full campaign), single seed, 2 episodes = 10 min
# pure episode time, leaving headroom for bring-up/probe/drain within a
# 20-min window. Output routed to its OWN directory so it can never be
# confused with, or collide with, the real Stage 3 campaign's
# results/PROGRESS.log.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap.yaml
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_trial_static_at_cap
TRIAL_SEED=950

echo "=== $(date +%H:%M:%S) TRIAL (static_at_cap, 2 episodes, seed=$TRIAL_SEED) ==="
python3 "$ORCH" --arm static_at_cap --algorithm baseline_static --reward-mode sla \
  --config "$CFG" --episodes-total 2 --batch-size 2 --seed "$TRIAL_SEED" --out-dir "$OUT_DIR"
rc=$?
echo "=== $(date +%H:%M:%S) TRIAL COMPLETE (rc=$rc) ==="
exit $rc
