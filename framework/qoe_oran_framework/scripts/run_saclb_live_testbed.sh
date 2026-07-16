#!/usr/bin/env bash
# Orchestrates a Stage Zero live SAC-LB run against the real OAI testbed,
# modeled on drl_slicing/scripts/run_oran_live_testbed_xapps.sh's bring-up
# sequence (UE fleet generation, Open5GS core, subscriber provisioning, UE
# fleet + traffic profiles, stabilization wait), reusing that script's own
# generic (non-algorithm-specific) helper scripts directly rather than
# duplicating them.
#
# One required difference from that script, because this run goes through
# the real E2 loop instead of the file-write rrmPolicy.json path: the gNB
# MUST be OAI's natively-built nr-softmodem (oai_ran, built via
# cmake_targets/build_oai) -- NOT the oaignb.yaml Docker service, which
# clones vanilla upstream OAI without the E2_AGENT/protobuf-c code this
# live loop depends on (confirmed by source inspection). This script
# starts it itself in the background. The README documents running it
# under sudo (presumably a real-hardware-USRP default); empirically, in
# --rfsim mode it runs fine unprivileged -- confirmed by an interactive
# run that reached "ALL gNBs ready" and printed live "[E2_AGENT] E2 agent
# heartbeat" lines with no elevated permissions, and a real LiveKpmSource
# poll()/send_control() round-trip against it succeeded.
#
# Usage:
#   ./run_saclb_live_testbed.sh --algorithm dqn \
#     --checkpoint results/offline/dqn/offline_closed_loop/rep_0/checkpoint.pt \
#     --episodes 2 --run-id live-smoke-dqn-001 [--dry-run]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
QOE_DIR="$ROOT_DIR/qoe_oran_framework"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
OAI_RAN_DIR="$ROOT_DIR/oai_ran"
GEN_DIR="$DOCKER_DIR/generated"

UE_COUNT=3
EMBB_RATIO=0.7
RANDOM_SEED=256
PROFILE_MODE="triad"
TRAFFIC_TARGET_IP="12.1.1.1"
CORE_CONTAINER="amf"
DRY_RUN=0

ALGORITHM=""
CHECKPOINT=""
EPISODES=50
RUN_ID=""
GNB_ID="gnb-0"
CONFIG="$QOE_DIR/configs/saclb_live.yaml"
RESULTS_DIR="$QOE_DIR/results/live"
REWARD_MODE="sla"

# OAI-native UE stack, NOT UERANSIM: UERANSIM's simulated air interface is
# not protocol-compatible with OAI's rfsimulator, and this script's gNB is
# always the real OAI nr-softmodem (that's the whole point -- E2_AGENT only
# exists there). Confirmed the hard way: UERANSIM UEs against this gNB fail
# "Cell selection failure, no suitable or acceptable cell found" forever.
UE_COMPOSE_FILE="$GEN_DIR/nr-ue-oai-fleet.yaml"
UE_PROFILE_CSV="$GEN_DIR/ue_fleet_profiles.csv"
NR_SOFTMODEM="$OAI_RAN_DIR/cmake_targets/ran_build/build/nr-softmodem"
GNB_CONF="$OAI_RAN_DIR/targets/PROJECTS/GENERIC-NR-5GC/CONF/ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithm) ALGORITHM="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gnb-id) GNB_ID="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --ue-count) UE_COUNT="$2"; shift 2 ;;
    --embb-ratio) EMBB_RATIO="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --core-container) CORE_CONTAINER="$2"; shift 2 ;;
    --reward-mode) REWARD_MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --algorithm {dqn|a2c|rainbow|lb_only} [--checkpoint PATH] [--episodes N] --run-id ID [--config PATH] [--results-dir DIR] [--ue-count N] [--reward-mode {sla|qoe}] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ -z "$ALGORITHM" ]]; then
  echo "--algorithm is required (dqn|a2c|rainbow|lb_only)"
  exit 2
fi
if [[ "$ALGORITHM" != "lb_only" && -z "$CHECKPOINT" ]]; then
  echo "--checkpoint is required unless --algorithm lb_only (live runs evaluate frozen weights only)"
  exit 2
fi
if [[ -z "$RUN_ID" ]]; then
  echo "--run-id is required"
  exit 2
fi

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $cmd"
  else
    eval "$cmd"
  fi
}

echo "Stage Zero live SAC-LB run"
echo "  Algorithm    : $ALGORITHM"
echo "  Checkpoint   : ${CHECKPOINT:-<none, lb_only>}"
echo "  Episodes     : $EPISODES"
echo "  Run ID       : $RUN_ID"
echo "  Config       : $CONFIG"
echo "  Reward mode  : $REWARD_MODE"
echo "  gNB binary   : $NR_SOFTMODEM"

GNB_LOG="$RESULTS_DIR/gnb_nr-softmodem.log"
GNB_PID_FILE="$RESULTS_DIR/gnb_nr-softmodem.pid"

echo ""
echo "[1/7] Start Open5GS core + metrics"
run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml up -d"

echo "[2/7] Generate UE fleet compose + profiles (OAI-native UE stack)"
# The gNB is a native host process (E2_AGENT only exists there), not a
# container -- containers reach it via the docker bridge gateway, not a
# docker-network peer address. Resolve that gateway now that the core's
# 'docker_open5gs_default' network exists (created by the compose up
# above), overriding the stale OAI_ENB_IP default in docker_open5gs/.env
# (which assumes a containerized oaignb service).
BRIDGE_GATEWAY=""
if [[ "$DRY_RUN" -eq 0 ]]; then
  BRIDGE_GATEWAY="$(docker network inspect docker_open5gs_default -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || true)"
  if [[ -z "$BRIDGE_GATEWAY" ]]; then
    echo "Could not resolve docker_open5gs_default bridge gateway -- is Open5GS core up?"
    exit 2
  fi
  echo "Docker bridge gateway (host, reachable from UE containers): $BRIDGE_GATEWAY"
fi
run_cmd "cd '$DRL_DIR' && /usr/bin/python3 scripts/generate_ue_fleet_compose.py --ue-count '$UE_COUNT' --embb-ratio '$EMBB_RATIO' --random-seed '$RANDOM_SEED' --profile-mode '$PROFILE_MODE' --output-compose '$UE_COMPOSE_FILE' --output-profiles '$UE_PROFILE_CSV' --ue-stack oai --oai-ue-image docker_oai_nrue:local --oai-launch-mode local-build --gnb-ip-override '$BRIDGE_GATEWAY'"

echo "[3/7] Start native OAI gNB (E2_AGENT built in, confirmed no root required for rfsim, listens 0.0.0.0:4043)"
mkdir -p "$RESULTS_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] cd '$OAI_RAN_DIR/cmake_targets/ran_build/build' && nohup '$NR_SOFTMODEM' -O '$GNB_CONF' --sa --rfsim > '$GNB_LOG' 2>&1 &"
else
  # cd happens synchronously (not inside the backgrounded job) so that $!
  # below is nohup's own PID, not an extra forked PID for a backgrounded
  # `cd ... && nohup ...` compound list (a real bash quirk that bit this
  # twice during testing -- stop_saclb_live_testbed.sh's pid-file kill was
  # missing the actual nr-softmodem process by one PID both times).
  pushd "$OAI_RAN_DIR/cmake_targets/ran_build/build" >/dev/null
  nohup "$NR_SOFTMODEM" -O "$GNB_CONF" --sa --rfsim > "$GNB_LOG" 2>&1 &
  echo $! > "$GNB_PID_FILE"
  popd >/dev/null
  sleep 5
  if ! kill -0 "$(cat "$GNB_PID_FILE")" 2>/dev/null; then
    echo "gNB process died immediately -- check $GNB_LOG"
    exit 1
  fi
  echo "gNB started, pid $(cat "$GNB_PID_FILE"), log: $GNB_LOG"
fi

echo "[4/7] Provision UE subscribers"
run_cmd "cd '$DRL_DIR' && bash scripts/provision_open5gs_subscribers.sh --profiles '$UE_PROFILE_CSV' --core-container '$CORE_CONTAINER'"

echo "[5/7] Start UE fleet containers"
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$UE_COMPOSE_FILE' up -d"

echo "[6/7] Start profile-based traffic loops (eMBB/URLLC/mMTC)"
run_cmd "cd '$DRL_DIR' && bash scripts/start_ue_traffic_profiles.sh --profiles '$UE_PROFILE_CSV' --target-ip '$TRAFFIC_TARGET_IP'"

echo "[7/7] Wait for UE sessions to stabilize"
if [[ "$DRY_RUN" -eq 0 ]]; then
  sleep 15
fi

mkdir -p "$RESULTS_DIR/$ALGORITHM"
OMEGA_PATH="$RESULTS_DIR/$ALGORITHM/${RUN_ID}_omega_log.jsonl"

echo ""
echo "Starting live xApp ($ALGORITHM), omega log -> $OMEGA_PATH"
XAPP_CMD="cd '$ROOT_DIR' && PYTHONPATH='$ROOT_DIR' /usr/bin/python3 qoe_oran_framework/xapp/saclb_xapp.py --config '$CONFIG' --algorithm '$ALGORITHM' --gnb-id '$GNB_ID' --episodes '$EPISODES' --seed '$RANDOM_SEED' --run-id '$RUN_ID' --omega-jsonl '$OMEGA_PATH' --reward-mode '$REWARD_MODE'"
if [[ -n "$CHECKPOINT" ]]; then
  XAPP_CMD+=" --checkpoint '$CHECKPOINT'"
fi
run_cmd "$XAPP_CMD"

echo ""
echo "Live run complete. Omega log: $OMEGA_PATH"
echo "To stop the testbed cleanly:"
echo "  $QOE_DIR/scripts/stop_saclb_live_testbed.sh"
