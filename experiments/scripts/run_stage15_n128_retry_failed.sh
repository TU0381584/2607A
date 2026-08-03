#!/usr/bin/env bash
# Retries exactly the 9 blocks that failed in run_stage15_n128_campaign.sh's
# first pass, root-caused to health_check.sh's unbounded "last 200 dmesg
# lines" window latching onto one real-but-already-fixed segfault for
# hours (fixed in health_check.sh itself, 2026-08-03). Same safety
# discipline: rm -rf the target rep_seed dir before each run (none of
# these 9 have stale partial data right now -- verified -- but keeping
# the guard for consistency/future reuse of this script).
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh

ORCH=/home/kmanojp/oranslice_rig/experiments/scripts/run_live_eval_arm.py
DRAIN=/home/kmanojp/oranslice_rig/experiments/scripts/drain_backlog.sh
HEALTH=/home/kmanojp/oranslice_rig/experiments/scripts/health_check.sh
RESTART=/home/kmanojp/oranslice_rig/experiments/scripts/restart_ran_stack.sh
OUT_DIR=/home/kmanojp/oranslice_rig/experiments/results/live_campaign_v2
PROGRESS_LOG="$OUT_DIR/PROGRESS.log"
BATCH_SIZE=2

BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline_v2.yaml
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap_v2.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
DQN_QOE_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

declare -A ALGO_OF=( [baseline]="baseline_static" [dqn_sla]="dqn" [dqn_qoe]="dqn" [static_at_cap]="baseline_static" )
declare -A MODE_OF=( [baseline]="sla" [dqn_sla]="sla" [dqn_qoe]="qoe" [static_at_cap]="sla" )
declare -A CFG_OF=( [baseline]="$BASELINE_CFG" [dqn_sla]="$CAMPAIGN_CFG" [dqn_qoe]="$CAMPAIGN_CFG" [static_at_cap]="$STATIC_CFG" )
declare -A CKPT_OF=( [baseline]="" [dqn_sla]="$DQN_SLA_CKPT" [dqn_qoe]="$DQN_QOE_CKPT" [static_at_cap]="" )
EPISODES_TOTAL=5

# Exactly the 9 failed blocks (arm:seed)
BLOCKS=(
  "baseline:974"
  "baseline:975" "dqn_sla:975" "dqn_qoe:975" "static_at_cap:975"
  "baseline:976" "dqn_sla:976" "dqn_qoe:976" "static_at_cap:976"
)

maybe_fix_iperf3() {
  if sudo docker logs iperf3-target --since 2m 2>&1 | grep -qi "server is busy running a test"; then
    echo "[retry] iperf3-target port-wedge signature detected -- recreating container"
    sudo docker rm -f iperf3-target >/dev/null 2>&1 || true
    sudo docker run -d --name iperf3-target --network demo-open5gs-public-net --ip 172.22.0.50 \
      --restart unless-stopped --entrypoint sh networkstatic/iperf3 \
      -c 'iperf3 -s -p 5201 & iperf3 -s -p 5202 & iperf3 -s -p 5203 & wait' >/dev/null
    sleep 3
  fi
}

for block in "${BLOCKS[@]}"; do
  arm="${block%%:*}"; seed="${block##*:}"
  algo="${ALGO_OF[$arm]}"; mode="${MODE_OF[$arm]}"; cfg="${CFG_OF[$arm]}"; ckpt="${CKPT_OF[$arm]}"
  rep_dir="$OUT_DIR/$arm/$mode/rep_seed${seed}"

  echo "=== $(date +%H:%M:%S) RETRY arm=$arm seed=$seed ==="
  rm -rf "$rep_dir"
  maybe_fix_iperf3
  bash "$HEALTH" || bash "$RESTART"
  bash "$DRAIN" 2>&1 | tail -3

  t0=$(date +%s)
  if [[ -z "$ckpt" ]]; then
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --episodes-total "$EPISODES_TOTAL" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --out-dir "$OUT_DIR"
  else
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --checkpoint "$ckpt" --episodes-total "$EPISODES_TOTAL" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --out-dir "$OUT_DIR"
  fi
  rc=$?
  t1=$(date +%s)
  elapsed=$((t1 - t0))

  if [[ $rc -eq 0 ]]; then
    echo "DONE arm=$arm seed=$seed elapsed_s=$elapsed retry_pass=1 ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "=== $(date +%H:%M:%S) RETRY-DONE arm=$arm seed=$seed (${elapsed}s) ==="
  else
    echo "FAILED arm=$arm seed=$seed elapsed_s=$elapsed rc=$rc retry_pass=1 ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "!!! RETRY-FAILED arm=$arm seed=$seed (rc=$rc) !!!"
  fi
done
echo "=== $(date +%H:%M:%S) RETRY PASS COMPLETE ==="
