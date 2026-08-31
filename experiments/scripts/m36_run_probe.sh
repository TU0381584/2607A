#!/usr/bin/env bash
# M36: launch the state-vector-logging live probe for a given UE count
# label (just a directory name -- it doesn't change what's actually
# attached, that's controlled separately by m36_add_ue2.sh/ue3.sh/etc.
# and manual UE4-6 launches). One arg: the UE-count label (e.g. "1", "2").
set -uo pipefail
N_UE="${1:?usage: m36_run_probe.sh <ue_count_label>}"
EPISODES="${2:-10}"

export XAPP_OAI_PROTO_DIR=~/oranslice_rig/ORANSlice/oai_ran/openair2/E2_AGENT/oai-oran-protolib/builds
OUT_DIR=~/oranslice_rig/experiments/results/m36_live/ue$N_UE
mkdir -p "$OUT_DIR"
cd ~/oranslice_rig/framework
setsid nohup env XAPP_OAI_PROTO_DIR="$XAPP_OAI_PROTO_DIR" ~/oranslice_rig/venv/bin/python3 \
  ~/oranslice_rig/experiments/scripts/m33_live_state_probe.py \
  --config ~/oranslice_rig/framework/qoe_oran_framework/configs/saclb_live.yaml \
  --algorithm dqn \
  --checkpoint ~/oranslice_rig/experiments/results/m8_live_anchor/offline_train/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt \
  --gnb-id gnb-0 \
  --episodes "$EPISODES" \
  --seed 900 \
  --run-id "m36_live_ue${N_UE}" \
  --omega-jsonl "$OUT_DIR/omega_log.jsonl" \
  --state-log "$OUT_DIR/state_log.jsonl" \
  --reward-mode sla \
  < /dev/null > ~/oranslice_rig/experiments/logs/m36_live/ue${N_UE}_probe.log 2>&1 &
disown -a
echo "[m36-probe] launched for ue_count=$N_UE, episodes=$EPISODES, PID=$!"
echo "[m36-probe] log: ~/oranslice_rig/experiments/logs/m36_live/ue${N_UE}_probe.log"
echo "[m36-probe] output: $OUT_DIR/{state,omega}_log.jsonl"
