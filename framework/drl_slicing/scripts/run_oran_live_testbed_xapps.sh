#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DRL_DIR="$ROOT_DIR/drl_slicing"
DOCKER_DIR="$ROOT_DIR/docker_open5gs"
GEN_DIR="$DOCKER_DIR/generated"

UE_COUNT=3
EMBB_RATIO=0.7
RANDOM_SEED=42
PROFILE_MODE="triad"
ACTIVE_ALGO="dqn"
GNB_MODE="ueransim"
UE_STACK="ueransim"
TRAFFIC_TARGET_IP="12.1.1.1"
CORE_CONTAINER="amf"
DRY_RUN=0
FORCE_HIGH_LOAD=0
OAI_UE_IMAGE="oaisoftwarealliance/oai-nr-ue:develop"
OAI_LAUNCH_MODE="env"
OAI_UE_IMAGE_SET=0
OAI_LAUNCH_MODE_SET=0

LIVE_CONFIG="$DRL_DIR/configs/openran_live_prom_true_3ue_triad.yaml"
UE_COMPOSE_FILE="$GEN_DIR/nr-ue-fleet.yaml"
UE_PROFILE_CSV="$GEN_DIR/ue_fleet_profiles.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ue-count)
      UE_COUNT="$2"
      shift 2
      ;;
    --embb-ratio)
      EMBB_RATIO="$2"
      shift 2
      ;;
    --random-seed)
      RANDOM_SEED="$2"
      shift 2
      ;;
    --profile-mode)
      PROFILE_MODE="$2"
      shift 2
      ;;
    --active-algo)
      ACTIVE_ALGO="$2"
      shift 2
      ;;
    --gnb-mode)
      GNB_MODE="$2"
      shift 2
      ;;
    --ue-stack)
      UE_STACK="$2"
      shift 2
      ;;
    --traffic-target-ip)
      TRAFFIC_TARGET_IP="$2"
      shift 2
      ;;
    --core-container)
      CORE_CONTAINER="$2"
      shift 2
      ;;
    --oai-ue-image)
      OAI_UE_IMAGE="$2"
      OAI_UE_IMAGE_SET=1
      shift 2
      ;;
    --oai-launch-mode)
      OAI_LAUNCH_MODE="$2"
      OAI_LAUNCH_MODE_SET=1
      shift 2
      ;;
    --config)
      LIVE_CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force-high-load)
      FORCE_HIGH_LOAD=1
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--ue-count N] [--embb-ratio R] [--random-seed S] [--profile-mode auto|ratio|triad] [--active-algo dqn|a2c] [--gnb-mode ueransim|oai] [--ue-stack ueransim|oai|external] [--traffic-target-ip IP] [--core-container NAME] [--oai-ue-image IMAGE] [--oai-launch-mode env|local-build] [--config PATH] [--dry-run] [--force-high-load]"
      exit 2
      ;;
  esac
done

if [[ "$ACTIVE_ALGO" != "dqn" && "$ACTIVE_ALGO" != "a2c" ]]; then
  echo "--active-algo must be dqn or a2c"
  exit 2
fi

if ! [[ "$UE_COUNT" =~ ^[0-9]+$ ]] || (( UE_COUNT <= 0 )); then
  echo "--ue-count must be a positive integer"
  exit 2
fi

if [[ "$PROFILE_MODE" != "auto" && "$PROFILE_MODE" != "ratio" && "$PROFILE_MODE" != "triad" ]]; then
  echo "--profile-mode must be auto, ratio, or triad"
  exit 2
fi

if [[ "$PROFILE_MODE" == "triad" && "$UE_COUNT" -ne 3 ]]; then
  echo "--profile-mode triad requires --ue-count 3"
  exit 2
fi

if [[ "$GNB_MODE" != "ueransim" && "$GNB_MODE" != "oai" ]]; then
  echo "--gnb-mode must be ueransim or oai"
  exit 2
fi

if [[ "$UE_STACK" != "ueransim" && "$UE_STACK" != "oai" && "$UE_STACK" != "external" ]]; then
  echo "--ue-stack must be ueransim, oai, or external"
  exit 2
fi

if [[ "$UE_STACK" == "oai" && "$GNB_MODE" != "oai" ]]; then
  echo "--ue-stack oai is only supported with --gnb-mode oai"
  exit 2
fi

if [[ "$OAI_LAUNCH_MODE" != "env" && "$OAI_LAUNCH_MODE" != "local-build" ]]; then
  echo "--oai-launch-mode must be env or local-build"
  exit 2
fi

if [[ "$GNB_MODE" == "oai" ]]; then
  GNB_COMPOSE_FILE="oaignb.yaml"
  ACTIVE_POLICY_PATH="$DOCKER_DIR/oai/rrmPolicy.json"
  GNB_IP_OVERRIDE="$(grep -E '^OAI_ENB_IP=' "$DOCKER_DIR/.env" | head -n 1 | cut -d'=' -f2- | tr -d '\r')"
  if [[ -z "$GNB_IP_OVERRIDE" ]]; then
    echo "Could not resolve OAI_ENB_IP from $DOCKER_DIR/.env"
    exit 2
  fi
else
  GNB_COMPOSE_FILE="nr-gnb.yaml"
  ACTIVE_POLICY_PATH="$ROOT_DIR/oai_ran/rrmPolicy.json"
  GNB_IP_OVERRIDE=""
fi

if [[ "$UE_STACK" == "oai" ]]; then
  if [[ "$OAI_LAUNCH_MODE_SET" -eq 0 && "$OAI_UE_IMAGE_SET" -eq 1 && "$OAI_UE_IMAGE" == "docker_oai_nrue:local" ]]; then
    OAI_LAUNCH_MODE="local-build"
  fi

  if [[ "$OAI_UE_IMAGE_SET" -eq 0 && "$OAI_LAUNCH_MODE_SET" -eq 1 && "$OAI_LAUNCH_MODE" == "local-build" ]]; then
    OAI_UE_IMAGE="docker_oai_nrue:local"
  fi

  if [[ "$OAI_UE_IMAGE_SET" -eq 0 && "$OAI_LAUNCH_MODE_SET" -eq 0 ]]; then
    if docker image inspect docker_oai_nrue:local >/dev/null 2>&1; then
      OAI_UE_IMAGE="docker_oai_nrue:local"
      OAI_LAUNCH_MODE="local-build"
    fi
  fi

  if [[ "$OAI_LAUNCH_MODE" == "local-build" ]] && [[ "$DRY_RUN" -eq 0 ]]; then
    if ! docker image inspect "$OAI_UE_IMAGE" >/dev/null 2>&1; then
      echo "OAI UE image not found locally: $OAI_UE_IMAGE"
      echo "Either build/import the image, or use --oai-launch-mode env --oai-ue-image oaisoftwarealliance/oai-nr-ue:develop"
      exit 2
    fi
  fi

  UE_COMPOSE_FILE="$GEN_DIR/nr-ue-oai-fleet.yaml"
else
  UE_COMPOSE_FILE="$GEN_DIR/nr-ue-fleet.yaml"
fi

enforce_memory_safety() {
  local available_mb
  local per_ue_mb=90
  local base_mb=1200

  if [[ "$UE_STACK" == "external" ]]; then
    return
  fi

  if [[ "$UE_STACK" == "oai" ]]; then
    per_ue_mb=420
    base_mb=1800
  fi

  available_mb="$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || true)"
  if [[ -z "$available_mb" ]]; then
    echo "Skipping memory safety check (MemAvailable not detected)."
    return
  fi

  local safe_max=$(( (available_mb - base_mb) / per_ue_mb ))
  if (( safe_max < 1 )); then
    safe_max=1
  fi

  local estimated_mb=$(( base_mb + UE_COUNT * per_ue_mb ))

  echo "Memory preflight"
  echo "  Available RAM    : ${available_mb} MiB"
  echo "  Estimated demand : ${estimated_mb} MiB (base=${base_mb} + per_ue=${per_ue_mb})"
  echo "  Safe UE ceiling  : ${safe_max}"

  if (( UE_COUNT > safe_max )); then
    if [[ "$FORCE_HIGH_LOAD" -eq 1 ]]; then
      echo "WARNING: forcing UE_COUNT=$UE_COUNT above safe ceiling=$safe_max"
    else
      echo "Adjusting UE_COUNT from $UE_COUNT to $safe_max to avoid OOM"
      UE_COUNT="$safe_max"
    fi
  fi
}

enforce_memory_safety

run_cmd() {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] $cmd"
  else
    eval "$cmd"
  fi
}

seed_active_policy() {
  local source_policy="$ROOT_DIR/oai_ran/rrmPolicy.json"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] ensure active policy exists at $ACTIVE_POLICY_PATH"
    return
  fi

  mkdir -p "$(dirname "$ACTIVE_POLICY_PATH")"
  if [[ -f "$source_policy" && "$source_policy" != "$ACTIVE_POLICY_PATH" ]]; then
    cp "$source_policy" "$ACTIVE_POLICY_PATH"
  elif [[ ! -f "$ACTIVE_POLICY_PATH" ]]; then
    echo '{"scope":"e2","style":"radio","policyType":"ORAN_TrafficSteeringPreference","preferences":[]}' > "$ACTIVE_POLICY_PATH"
  fi
}

cleanup_stale_ue_containers() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] remove stale containers named nr_ue_*"
    return
  fi

  local stale
  stale="$(docker ps -a --format '{{.Names}}' | grep '^nr_ue_' || true)"
  if [[ -n "$stale" ]]; then
    echo "$stale" | xargs -r docker rm -f >/dev/null 2>&1 || true
  fi
}

start_xapp() {
  local algorithm="$1"
  local service_name="$2"
  local policy_path="$3"
  local run_log_csv="$4"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] start xApp $service_name ($algorithm) -> $policy_path"
    return
  fi

  mkdir -p "$DRL_DIR/results/xapps"
  (
    cd "$DRL_DIR"
    PYTHONPATH=. nohup /usr/bin/python3 scripts/xapp_service.py \
      --config "$LIVE_CONFIG" \
      --algorithm "$algorithm" \
      --repeat 0 \
      --name "$service_name" \
      --policy-json-path "$policy_path" \
      --run-log-csv "$run_log_csv" \
      > "results/xapps/${service_name}.log" 2>&1 &
    echo $! > "results/xapps/${service_name}.pid"
  )
}

echo "Starting ORAN live E2E run with DRL xApps"
echo "  UE count         : $UE_COUNT"
echo "  eMBB ratio       : $EMBB_RATIO"
echo "  Random seed      : $RANDOM_SEED"
echo "  Profile mode     : $PROFILE_MODE"
echo "  Active controller: $ACTIVE_ALGO"
echo "  gNB mode         : $GNB_MODE"
echo "  UE stack         : $UE_STACK"
if [[ "$UE_STACK" == "oai" ]]; then
  echo "  OAI UE image     : $OAI_UE_IMAGE"
  echo "  OAI launch mode  : $OAI_LAUNCH_MODE"
fi
if [[ -n "$GNB_IP_OVERRIDE" ]]; then
  echo "  UE gNB override  : $GNB_IP_OVERRIDE"
fi
echo "  Config           : $LIVE_CONFIG"

echo "[1/10] Generate UE fleet compose + profiles"
if [[ "$UE_STACK" == "ueransim" || "$UE_STACK" == "oai" ]]; then
  GEN_CMD="cd '$DRL_DIR' && /usr/bin/python3 scripts/generate_ue_fleet_compose.py --ue-count '$UE_COUNT' --embb-ratio '$EMBB_RATIO' --random-seed '$RANDOM_SEED' --profile-mode '$PROFILE_MODE' --output-compose '$UE_COMPOSE_FILE' --output-profiles '$UE_PROFILE_CSV' --ue-stack '$UE_STACK'"
  if [[ -n "$GNB_IP_OVERRIDE" ]]; then
    GEN_CMD+=" --gnb-ip-override '$GNB_IP_OVERRIDE'"
  fi
  if [[ "$UE_STACK" == "oai" ]]; then
    GEN_CMD+=" --oai-ue-image '$OAI_UE_IMAGE' --oai-launch-mode '$OAI_LAUNCH_MODE'"
  fi
  run_cmd "$GEN_CMD"
else
  echo "Skipping generated UE fleet (UE stack is external)"
fi

echo "[2/10] Start Open5GS core + metrics"
run_cmd "cd '$DOCKER_DIR' && docker compose -f sa-deploy.yaml up -d"

echo "[3/10] Start gNB stack ($GNB_MODE)"
seed_active_policy
run_cmd "cd '$DOCKER_DIR' && docker compose -f '$GNB_COMPOSE_FILE' up -d"

echo "[4/10] Provision UE subscribers"
if [[ "$UE_STACK" == "ueransim" || "$UE_STACK" == "oai" ]]; then
  run_cmd "cd '$DRL_DIR' && bash scripts/provision_open5gs_subscribers.sh --profiles '$UE_PROFILE_CSV' --core-container '$CORE_CONTAINER'"
else
  echo "Skipping subscriber reprovision (expected external/OAI UE provisioning flow)"
fi

echo "[5/10] Start UE fleet containers"
if [[ "$UE_STACK" == "ueransim" || "$UE_STACK" == "oai" ]]; then
  cleanup_stale_ue_containers
  run_cmd "cd '$DOCKER_DIR' && docker compose -f '$UE_COMPOSE_FILE' up -d"
else
  echo "Skipping UE fleet startup (UE stack is external)"
fi

echo "[6/10] Start profile-based traffic loops (eMBB/URLLC/mMTC)"
if [[ "$UE_STACK" == "ueransim" || "$UE_STACK" == "oai" ]]; then
  run_cmd "cd '$DRL_DIR' && bash scripts/start_ue_traffic_profiles.sh --profiles '$UE_PROFILE_CSV' --target-ip '$TRAFFIC_TARGET_IP'"
else
  echo "Skipping container traffic loops (expected external/OAI UE traffic)"
fi

echo "[7/10] Wait for UE sessions to stabilize"
if [[ "$DRY_RUN" -eq 0 ]]; then
  sleep 15
fi

echo "[8/10] Start DRL xApps (both loaded, one active)"
DQN_SHADOW_POLICY="$DRL_DIR/results/xapps/xapp-dqn-live/rrmPolicy.json"
A2C_SHADOW_POLICY="$DRL_DIR/results/xapps/xapp-a2c-live/rrmPolicy.json"

if [[ "$ACTIVE_ALGO" == "dqn" ]]; then
  start_xapp "dqn" "xapp-dqn-live" "$ACTIVE_POLICY_PATH" "$DRL_DIR/results/xapps/xapp-dqn-live/run_log.csv"
  start_xapp "a2c" "xapp-a2c-live" "$A2C_SHADOW_POLICY" "$DRL_DIR/results/xapps/xapp-a2c-live/run_log.csv"
else
  start_xapp "dqn" "xapp-dqn-live" "$DQN_SHADOW_POLICY" "$DRL_DIR/results/xapps/xapp-dqn-live/run_log.csv"
  start_xapp "a2c" "xapp-a2c-live" "$ACTIVE_POLICY_PATH" "$DRL_DIR/results/xapps/xapp-a2c-live/run_log.csv"
fi

echo "[9/10] Check active session metric (Prometheus)"
if [[ "$DRY_RUN" -eq 0 ]]; then
  SESSION_JSON="$(curl -s --get 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=fivegs_upffunction_upf_sessionnbr' || true)"
  SESSION_COUNT="$(printf '%s' "$SESSION_JSON" | /usr/bin/python3 -c 'import json,sys
raw=sys.stdin.read().strip()
if not raw:
    print("n/a")
    raise SystemExit(0)
try:
    obj=json.loads(raw)
    data=obj.get("data",{}).get("result",[])
    if not data:
        print("n/a")
    else:
        print(data[0].get("value",[0,"n/a"])[1])
except Exception:
    print("n/a")
')"
  echo "Prometheus fivegs_upffunction_upf_sessionnbr = $SESSION_COUNT"
else
  echo "[DRY-RUN] Prometheus query skipped"
fi

echo "[10/10] Launch complete"
echo "Monitoring commands:"
echo "  docker ps --format 'table {{.Names}}\t{{.Status}}'"
echo "  tail -f $DRL_DIR/results/xapps/xapp-dqn-live.log"
echo "  tail -f $DRL_DIR/results/xapps/xapp-a2c-live.log"
echo "  curl -s --get 'http://127.0.0.1:9090/api/v1/query' --data-urlencode 'query=fivegs_upffunction_upf_sessionnbr'"
echo ""
echo "To stop everything cleanly:"
echo "  $DRL_DIR/scripts/stop_oran_live_testbed_xapps.sh"
