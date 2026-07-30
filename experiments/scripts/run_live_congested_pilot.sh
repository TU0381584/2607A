#!/usr/bin/env bash
# Live congested-scenario pilot -- the "lighter, honest" version of Future
# Work item 2 (docs -- manuscript's own Conclusion: "reproduce the
# congested, multi-slice scenario on the live rig rather than offline").
#
# IMPORTANT SCOPE NOTE, not to be silently lost: this does NOT reproduce
# Table II's shared-PRB-pool physics (SharedPoolCongestedKpmSource's fiat
# shared_pool_prb=8.0 budget) live -- the real gNB has 106 physical PRBs,
# and this campaign's combined ceiling headroom (embb 12 + urllc 4 + mmtc
# 3 = 19) is nowhere near enough to genuinely contend for that budget, so
# there is no real physical scarcity to reproduce at this cap scale. What
# this DOES do: live-evaluate the ALREADY-TRAINED congested/URLLC-
# preservation checkpoints (dqn_sla_congested, dqn_qoe_congested --
# experiments/results/offline_congested/{sla,qoe}/seed256/dqn/checkpoint.pt,
# trained via train_offline_congested.py on saclb_offline_congested_v1.yaml)
# under the SAME real traffic/ceilings the rest of this campaign already
# uses, and honestly reports whether the URLLC-preservation pattern still
# shows up under real (ceiling-only, not shared-pool) contention -- a
# real, different, more modest question than Table II answers, not a
# replacement for it.
#
# Pilot scope (start small, per this project's own established
# precedent -- 1h trial before the 3h campaign before the validation
# round): 1 seed, 2 episodes/arm, 3 arms. Scale up only after reviewing
# this pilot's results.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_congested_pilot
PROGRESS_LOG="$OUT_DIR/PROGRESS.log"
EPISODES_TOTAL=2
BATCH_SIZE=2
SEED=950

CONGESTED_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_congested_v1.yaml
CONGESTED_BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_congested_v1_baseline.yaml
DQN_SLA_CONGESTED_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_congested/sla/seed256/dqn/checkpoint.pt
DQN_QOE_CONGESTED_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_congested/qoe/seed256/dqn/checkpoint.pt

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

is_done() {
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {
  local arm="$1" algo="$2" mode="$3" cfg="$4" ckpt="$5" seed="$6"

  if is_done "$arm" "$seed"; then
    echo "[congested-pilot] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
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

echo "=== $(date +%H:%M:%S) LIVE CONGESTED PILOT START ==="
run_one baseline_congested  baseline_static sla "$CONGESTED_BASELINE_CFG" ""                          "$SEED"
run_one dqn_sla_congested   dqn             sla "$CONGESTED_CFG"          "$DQN_SLA_CONGESTED_CKPT"   "$SEED"
run_one dqn_qoe_congested   dqn             qoe "$CONGESTED_CFG"          "$DQN_QOE_CONGESTED_CKPT"   "$SEED"
echo "=== $(date +%H:%M:%S) LIVE CONGESTED PILOT COMPLETE ==="
