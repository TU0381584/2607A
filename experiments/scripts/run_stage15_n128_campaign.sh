#!/usr/bin/env bash
# n=128/arm extension of live_campaign_v2 (currently n=46/arm, seeds
# 950-960). User: "proceed to n=128 ... don't prompt for user commands
# in between ... I will leave my computer on for 2 or 3 days."
#
# Adds 82 new episodes/arm via seeds 961-976 (16 seeds x 5 ep/seed = 80)
# + seed 977 (1 seed x 2 ep) = 82, giving 46+82=128/arm exactly. Same
# OUT_DIR, same PROGRESS_LOG, same is_done/DONE bookkeeping, same
# arm-rotation discipline as run_stage5_v2_reverify_10h.sh -- this is an
# extension of that same dataset, not a new one, so metrics_stage5_v2.py
# picks it up by just widening ARM_SEEDS.
#
# SAFETY ADDITION over the prior templates (Stage 11's own documented
# lesson): a failed (arm, seed) block is NEVER retried by simply
# re-invoking the orchestrator in place -- run_live_eval_arm.py's
# per-batch run_id is deterministic and OmegaLogger appends, so retrying
# into an existing partial omega_log.jsonl duplicates rows under the
# same run_id (exactly Stage 11's caught data-corruption bug). Before
# any retry here, the ENTIRE rep_seed{N} output directory for that
# (arm, seed) is deleted first, so the retry always starts from a clean
# file. Also auto-recreates the iperf3-target container on the
# long-documented "server is busy running a test" port-wedge signature
# instead of requiring a human to notice and fix it (CAMPAIGN_LOG.md /
# STAGE11 precedent: same fix, every time, previously done by hand).
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
MAX_BLOCK_RETRIES=3

BASELINE_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_baseline_v2.yaml
CAMPAIGN_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml
STATIC_CFG=/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_static_at_cap_v2.yaml
DQN_SLA_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt
DQN_QOE_CKPT=/home/kmanojp/oranslice_rig/experiments/results/offline_v2/qoe/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt

declare -A ALGO_OF=( [baseline]="baseline_static" [dqn_sla]="dqn" [dqn_qoe]="dqn" [static_at_cap]="baseline_static" )
declare -A MODE_OF=( [baseline]="sla" [dqn_sla]="sla" [dqn_qoe]="qoe" [static_at_cap]="sla" )
declare -A CFG_OF=( [baseline]="$BASELINE_CFG" [dqn_sla]="$CAMPAIGN_CFG" [dqn_qoe]="$CAMPAIGN_CFG" [static_at_cap]="$STATIC_CFG" )
declare -A CKPT_OF=( [baseline]="" [dqn_sla]="$DQN_SLA_CKPT" [dqn_qoe]="$DQN_QOE_CKPT" [static_at_cap]="" )

ARMS_BASE=(baseline dqn_sla dqn_qoe static_at_cap)
# seed -> episodes_total for this seed
declare -A EPISODES_OF
for s in 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976; do EPISODES_OF[$s]=5; done
EPISODES_OF[977]=2
SEEDS=(961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977)

mkdir -p "$OUT_DIR"
touch "$PROGRESS_LOG"

rotate() {
  local n="$2"
  local arr=("${ARMS_BASE[@]}")
  for ((i=0; i<n; i++)); do
    arr=("${arr[@]:1}" "${arr[0]}")
  done
  echo "${arr[@]}"
}

is_done() {
  grep -q "DONE arm=$1 seed=$2 " "$PROGRESS_LOG" 2>/dev/null
}

rep_dir_for() {
  local arm="$1" seed="$2" mode="${MODE_OF[$1]}"
  echo "$OUT_DIR/$arm/$mode/rep_seed${seed}"
}

# Detects the long-documented iperf3-target port-wedge signature and
# recreates the container -- same fix this project has always applied
# by hand (CAMPAIGN_LOG.md, STAGE5, STAGE11), automated here so a
# multi-day unattended run doesn't stall on it.
maybe_fix_iperf3() {
  if sudo docker logs iperf3-target --since 2m 2>&1 | grep -qi "server is busy running a test"; then
    echo "[n128] iperf3-target port-wedge signature detected -- recreating container"
    sudo docker rm -f iperf3-target >/dev/null 2>&1 || true
    sudo docker run -d --name iperf3-target --network demo-open5gs-public-net --ip 172.22.0.50 \
      --restart unless-stopped --entrypoint sh networkstatic/iperf3 \
      -c 'iperf3 -s -p 5201 & iperf3 -s -p 5202 & iperf3 -s -p 5203 & wait' >/dev/null
    sleep 3
  fi
}

run_one_attempt() {
  local arm="$1" seed="$2" algo="${ALGO_OF[$1]}" mode="${MODE_OF[$1]}" cfg="${CFG_OF[$1]}" ckpt="${CKPT_OF[$1]}"
  local ep_total="${EPISODES_OF[$seed]}"

  echo "=== $(date +%H:%M:%S) DRAIN before arm=$arm seed=$seed ==="
  bash "$DRAIN" 2>&1 | tail -5

  echo "=== $(date +%H:%M:%S) RUN arm=$arm seed=$seed algo=$algo mode=$mode episodes=$ep_total ==="
  if [[ -z "$ckpt" ]]; then
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --episodes-total "$ep_total" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --out-dir "$OUT_DIR"
  else
    python3 "$ORCH" --arm "$arm" --algorithm "$algo" --reward-mode "$mode" \
      --config "$cfg" --checkpoint "$ckpt" --episodes-total "$ep_total" --batch-size "$BATCH_SIZE" \
      --seed "$seed" --out-dir "$OUT_DIR"
  fi
}

run_one() {
  local arm="$1" seed="$2" rotation_idx="$3"

  if is_done "$arm" "$seed"; then
    echo "[n128] SKIP (already DONE per PROGRESS_LOG): arm=$arm seed=$seed"
    return 0
  fi

  local rep_dir; rep_dir="$(rep_dir_for "$arm" "$seed")"
  local t0=$(date +%s)
  local attempt=0 rc=1

  while (( attempt <= MAX_BLOCK_RETRIES )); do
    if (( attempt > 0 )); then
      echo "[n128] RETRY $attempt/$MAX_BLOCK_RETRIES for arm=$arm seed=$seed -- " \
           "deleting stale partial output first (Stage 11 append-corruption lesson)"
      rm -rf "$rep_dir"
      maybe_fix_iperf3
      bash "$HEALTH" >/dev/null 2>&1 || bash "$RESTART" >/dev/null 2>&1
    fi
    run_one_attempt "$arm" "$seed"
    rc=$?
    if [[ $rc -eq 0 ]]; then
      break
    fi
    attempt=$((attempt + 1))
  done

  local t1=$(date +%s)
  local elapsed=$((t1 - t0))

  if [[ $rc -eq 0 ]]; then
    echo "DONE arm=$arm seed=$seed elapsed_s=$elapsed rotation_idx=$rotation_idx attempts=$((attempt+1)) ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "=== $(date +%H:%M:%S) DONE arm=$arm seed=$seed (${elapsed}s, $((attempt+1)) attempt(s)) ==="
  else
    echo "FAILED arm=$arm seed=$seed elapsed_s=$elapsed rc=$rc attempts=$((attempt+1)) ts=$(date -Iseconds)" >> "$PROGRESS_LOG"
    echo "!!! FAILED arm=$arm seed=$seed after $((attempt+1)) attempts (rc=$rc) -- moving on, not halting campaign !!!"
  fi
}

echo "=== $(date +%H:%M:%S) STAGE15 N=128 CAMPAIGN START ==="
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  read -ra arm_order <<< "$(rotate ARMS_BASE "$idx")"
  echo "=== seed=$seed episodes=${EPISODES_OF[$seed]} arm_order=${arm_order[*]} ==="
  for arm in "${arm_order[@]}"; do
    run_one "$arm" "$seed" "$idx"
  done
done
echo "=== $(date +%H:%M:%S) STAGE15 N=128 CAMPAIGN COMPLETE ==="
