#!/usr/bin/env bash
# Offline reverification (~3h budget, zero rig time): checks that Stage
# 5's v2 retraining (docs/STAGE5_recalibration.md section 4) wasn't a
# single-seed (256) artifact, and substantially strengthens Table II's
# statistical basis.
#
# Part A: retrain dqn_sla_v2/dqn_qoe_v2 across 5 NEW seeds (257-261),
# same script/episode count Stage 5 itself used
# (train_offline_live_scale.py, 300 episodes). Reports Q1->Q4 reward
# improvement per seed so convergence can be checked across seeds, not
# just eyeballed on the original seed256 run.
#
# Part B: re-runs the offline congested held-out evaluation
# (eval_congested_vs_baseline.py, the exact script behind Table II) with
# 8 NEW seeds (953-960) IN ADDITION to the existing 950-952, using the
# SAME already-trained congested checkpoints (no retraining of those --
# this is a statistical-power extension of the existing evaluation, not
# a new training run). Combined with the original 3 seeds this brings
# Table II's evidence base from 3 seeds x 15 episodes (45/arm) to 11
# seeds x 15 episodes (165/arm).
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
cd /home/kmanojp/oranslice_rig

LOG_DIR=experiments/logs
mkdir -p "$LOG_DIR"

echo "=== $(date +%H:%M:%S) OFFLINE V2 REVERIFY START ==="

echo "--- Part A: retrain dqn_sla_v2 / dqn_qoe_v2 across seeds 257-261 ---"
for seed in 257 258 259 260 261; do
  for mode in sla qoe; do
    echo "=== $(date +%H:%M:%S) train dqn/$mode seed=$seed ==="
    python3 experiments/scripts/train_offline_live_scale.py \
      --algorithm dqn --reward-mode "$mode" --episodes 300 --seed "$seed" \
      --results-dir experiments/results/offline_v2_reverify \
      2>&1 | tee -a "$LOG_DIR/offline_v2_reverify_train.log"
  done
done

echo "--- Part B: expand congested offline held-out eval to seeds 953-960 ---"
python3 experiments/scripts/eval_congested_vs_baseline.py \
  --ckpt-root experiments/results/offline_congested \
  --seeds 953 954 955 956 957 958 959 960 \
  --episodes-per-seed 15 \
  --out experiments/results/congested_vs_baseline_v7_reverify \
  2>&1 | tee -a "$LOG_DIR/offline_v2_reverify_congested_eval.log"

echo "=== $(date +%H:%M:%S) OFFLINE V2 REVERIFY COMPLETE ==="
