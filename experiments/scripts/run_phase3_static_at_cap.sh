#!/usr/bin/env bash
# Stage 3 (paper #4 rework plan): the STATIC-AT-CAP arm -- a fourth arm
# added to the live campaign, run under the identical per-seed protocol as
# the original 5 arms (5 episodes/seed, same traffic profiles, same
# logging, drain between transitions, health-checked batch orchestration
# via the already-proven run_live_eval_arm.py). Config-only addition
# (experiments/configs/saclb_campaign_static_at_cap.yaml); no framework
# or orchestrator source touched -- see that config's header for the
# mechanism (nominal_ratio == max_ratio_cap + ceiling_step_ratio: 0).
#
# SCOPE, deliberately reduced from the original 5-arm campaign's 3 seeds
# to 2 (950, 951 -- dropping 952): user asked for a ~1h run, not the ~1.5-2h
# a 3rd seed would add. 5 episodes/seed is kept UNCHANGED (not shortened)
# so each seed's data is directly comparable, episode-for-episode, to
# every other arm's per-seed block in Table II -- the cut is in seed
# COUNT (less cross-seed variance evidence), not episode depth within a
# seed. docs/STAGE3_oracle.md must say n=2 seeds/10 episodes, not silently
# present this as matching the other arms' n=3/15.
#
# Same crash-safe PROGRESS_LOG discipline as run_phase_a_campaign.sh:
# one line per completed seed, skip-on-restart.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap.yaml
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign
PROGRESS_LOG=/home/kmanojp/oranslice_rig/experiments/results/live_campaign/PROGRESS.log
EPISODES_TOTAL=5
BATCH_SIZE=2
ARM=static_at_cap
SEEDS=(950 951)   # reduced from (950 951 952) -- see header note, ~1h budget

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

is_done() {
  grep -q "DONE arm=$ARM seed=$1 " "$PROGRESS_LOG" 2>/dev/null
}

echo "=== $(date +%H:%M:%S) PHASE 3 (static_at_cap) START ==="
for seed in "${SEEDS[@]}"; do
  if is_done "$seed"; then
    echo "[phase3] SKIP (already DONE per PROGRESS_LOG): arm=$ARM seed=$seed"
    continue
  fi

  echo "=== $(date +%H:%M:%S) DRAIN before arm=$ARM seed=$seed ==="
  bash "$DRAIN" 2>&1 | tail -5

  echo "=== $(date +%H:%M:%S) RUN arm=$ARM seed=$seed ==="
  t0=$(date +%s)
  python3 "$ORCH" --arm "$ARM" --algorithm baseline_static --reward-mode sla \
    --config "$CFG" --episodes-total "$EPISODES_TOTAL" --batch-size "$BATCH_SIZE" \
    --seed "$seed" --out-dir "$OUT_DIR"
  rc=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [[ $rc -eq 0 ]]; then
    echo "DONE arm=$ARM seed=$seed elapsed_s=$elapsed ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "=== $(date +%H:%M:%S) DONE arm=$ARM seed=$seed (${elapsed}s) ==="
  else
    echo "FAILED arm=$ARM seed=$seed elapsed_s=$elapsed rc=$rc ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "!!! FAILED arm=$ARM seed=$seed (rc=$rc) !!!"
  fi
done
echo "=== $(date +%H:%M:%S) PHASE 3 (static_at_cap) COMPLETE ==="
