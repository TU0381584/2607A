#!/usr/bin/env bash
# Stage 3 reverification round (user: "truly verify whether DQN is
# superior to the baseline by preventing collapsing. run more tests to
# confirm before proceeding.").
#
# Two additions, run back-to-back on one rig session:
#   1. static_at_cap's originally-planned 3rd seed (952) -- brings it to
#      n=15 episodes, matching every other arm's existing sample size
#      exactly, so the Fisher exact test vs DQN is no longer comparing
#      unequal n (was 8/10 vs 15/15).
#   2. DQN(SLA)'s seed-256 checkpoint re-evaluated on 2 BRAND-NEW live
#      seeds (953, 954) it has never been run on before (the original
#      campaign only ever used 950/951/952) -- to confirm its 0/15
#      collapse-free record generalizes to fresh conditions, not just
#      those 3 seeds. Written to arm=dqn_sla_reverify so the original
#      dqn_sla/sla/rep_seed{950,951,952} data is never touched or mixed
#      with this new data.
#
# No framework or orchestrator source touched; same drain-between-arms,
# health-checked batch orchestration, crash-safe PROGRESS_LOG discipline
# as every prior stage.
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap.yaml
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign
PROGRESS_LOG=/home/kmanojp/oranslice_rig/experiments/results/live_campaign/PROGRESS.log
EPISODES_TOTAL=5
BATCH_SIZE=2

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

is_done() {  # is_done(arm, seed)
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

run_one() {  # run_one(arm, algo, mode, config, checkpoint_or_empty, seed)
  local arm="$1" algo="$2" mode="$3" cfg="$4" ckpt="$5" seed="$6"

  if is_done "$arm" "$seed"; then
    echo "[reverify] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
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

echo "=== $(date +%H:%M:%S) REVERIFY START ==="
run_one static_at_cap    baseline_static sla "$STATIC_CFG"   ""            952
run_one dqn_sla_reverify dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT" 953
run_one dqn_sla_reverify dqn             sla "$CAMPAIGN_CFG" "$DQN_SLA_CKPT" 954
echo "=== $(date +%H:%M:%S) REVERIFY COMPLETE ==="
