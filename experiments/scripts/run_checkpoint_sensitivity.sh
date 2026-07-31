#!/usr/bin/env bash
# Checkpoint-sensitivity check: Stage 10's online reverification found
# DQN-SLA (checkpoint trained on seed256) statistically tied with
# static-at-cap live (44/46 vs 44/46, Fisher p=1.0) -- overturning
# Stage 3's original claim. The offline reverification then showed
# TRAINING itself is not seed-sensitive (Q1->Q4 convergence consistent
# across 6 seeds), which rules out "seed256 was an unlucky training
# draw" as the explanation, but does NOT tell us whether seed256's
# LIVE behaviour generalizes to other independently-trained policies.
#
# This script closes that gap directly: live-evaluates all 5 OTHER
# already-trained DQN-SLA checkpoints (seeds 257-261,
# experiments/results/offline_v2_reverify/sla/seed{N}/...) against the
# same static-at-cap record (already at n=46, no need to re-run it),
# using the EXACT SAME seed structure as every other arm (950-952 at
# 2 ep/seed, 953-955 at 5 ep/seed = 21 episodes/checkpoint) so results
# are directly comparable.
#
# 5 checkpoints x 21 episodes = 105 new live episodes, ~10-11h at this
# campaign's historical pace (~6 min/episode).
#
# If ALL 5 new checkpoints also land close to static-at-cap's 44/46
# rate, that's strong evidence this is a property of the corrected
# training environment (or of real live-hardware conditions), not of
# one specific checkpoint. If they scatter widely instead, that points
# back toward checkpoint-specific variance as the explanation.
#
# Same crash-safe PROGRESS_LOG, skip-if-done, drain-between-arms
# discipline as every prior live stage.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_checkpoint_sensitivity
PROGRESS_LOG="$OUT_DIR/PROGRESS.log"
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
CKPT_ROOT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2_reverify/sla

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

is_done() {
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {
  local arm="$1" seed="$2" episodes_total="$3" ckpt="$4"

  if is_done "$arm" "$seed"; then
    echo "[cksens] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
    return 0
  fi

  echo "=== $(date +%H:%M:%S) DRAIN before arm=$arm seed=$seed ==="
  bash "$DRAIN" 2>&1 | tail -5

  echo "=== $(date +%H:%M:%S) RUN arm=$arm seed=$seed episodes_total=$episodes_total ==="
  local t0=$(date +%s)
  python3 "$ORCH" --arm "$arm" --algorithm dqn --reward-mode sla \
    --config "$CAMPAIGN_CFG" --checkpoint "$ckpt" \
    --episodes-total "$episodes_total" --batch-size 2 \
    --seed "$seed" --out-dir "$OUT_DIR"
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

echo "=== $(date +%H:%M:%S) CHECKPOINT SENSITIVITY START ==="
for seed_ckpt in 257 258 259 260 261; do
  arm="dqn_sla_seed${seed_ckpt}"
  ckpt="$CKPT_ROOT/seed${seed_ckpt}/dqn/offline_closed_loop/rep_0/checkpoint.pt"
  run_one "$arm" 950 2 "$ckpt"
  run_one "$arm" 951 2 "$ckpt"
  run_one "$arm" 952 2 "$ckpt"
  run_one "$arm" 953 5 "$ckpt"
  run_one "$arm" 954 5 "$ckpt"
  run_one "$arm" 955 5 "$ckpt"
done
echo "=== $(date +%H:%M:%S) CHECKPOINT SENSITIVITY COMPLETE ==="
