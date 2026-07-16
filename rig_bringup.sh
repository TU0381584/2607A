#!/usr/bin/env bash
# =============================================================================
# Fresh-rig bring-up for the ORANSlice-based QoE O-RAN testbed (paper #4).
# Encodes the precondition sequencing from PROJECT_HANDOFF_SUMMARY.md §5 and
# the migration verification of 2026-07-14. Run stages IN ORDER; each stage
# gates the next. Designed for Ubuntu 24.04 LTS (verified supported by the
# 2024.w28-based build system) or 22.04 (ORANSlice's documented target).
#
# HARDWARE FLOOR (handoff §4.6): >= 8 GB RAM for gNB + 2-3 rfsim UEs;
# ~1 GB RSS per nr-uesoftmodem at 106 PRB / numerology 1 is a hard PHY-buffer
# cost, not tunable. 16 GB comfortably supports the 3-4 UE contention tests
# the old rig could not run.
# =============================================================================
set -euo pipefail
STAGE="${1:-help}"
WORK="${ORANSLICE_HOME:-$HOME/oranslice_rig}"
REPO_FRAMEWORK="${FRAMEWORK_DIR:-$WORK/framework}"   # extracted migration bundle
OAI_DIR="$WORK/ORANSlice/oai_ran"
PROTO_BUILDS="$OAI_DIR/openair2/E2_AGENT/oai-oran-protolib/builds"
GNB_CONF="$OAI_DIR/targets/PROJECTS/GENERIC-NR-5GC/CONF/ORANSlice.gnb.sa.band78.fr1.106PRB.usrpx310.conf"

case "$STAGE" in
# -----------------------------------------------------------------------------
clone)
  mkdir -p "$WORK" && cd "$WORK"
  # ORANSlice main = customized RAN slicing + UDP E2 agent on OAI 2024.w28.
  # Do NOT use vanilla upstream OAI: its only E2 agent is the FlexRIC
  # E2AP/SCTP one, which requires a real near-RT RIC and does not speak
  # live_kpm_source.py's UDP protobuf protocol.
  git clone https://github.com/wineslab/ORANSlice.git
  # Open5GS core with slicing (the core used on the old rig):
  git clone -b open5gs_slicing https://github.com/wineslab/docker_open5gs.git
  echo "[clone] OK. Record the ORANSlice commit hash in your lab notes:"
  git -C ORANSlice log -1 --format="%H %ad"
  ;;
# -----------------------------------------------------------------------------
build)
  # protobuf-c is a hard prerequisite of the E2 agent (README-documented):
  sudo apt-get update
  sudo apt-get install -y protobuf-compiler libprotoc-dev libprotobuf-c-dev \
      protobuf-c-compiler build-essential cmake ninja-build git
  # If libprotobuf-c-dev is too old on your distro, build protobuf-c from
  # source per the ORANSlice README instead.
  cd "$OAI_DIR/cmake_targets"
  ./build_oai -I                      # one-time dependency install
  ./build_oai --ninja --gNB --nrUE    # rfsim needs no -w USRP; add it for OTA
  # Regenerate the *Python* protobuf module from THIS checkout (do not carry
  # the old rig's ran_messages_pb2.py):
  cd "$OAI_DIR/openair2/E2_AGENT/oai-oran-protolib"
  mkdir -p builds && protoc --python_out=builds ran_messages.proto
  echo "[build] OK. export XAPP_OAI_PROTO_DIR=$PROTO_BUILDS"
  ;;
# -----------------------------------------------------------------------------
core)
  cd "$WORK/docker_open5gs"
  echo "[core] Follow this repo's README (docker compose based). Then provision"
  echo "       subscribers per slice — reuse the pattern in"
  echo "       $REPO_FRAMEWORK/drl_slicing/scripts/provision_open5gs_subscribers.sh"
  ;;
# -----------------------------------------------------------------------------
gnb)
  cd "$OAI_DIR/cmake_targets/ran_build/build"
  sudo ./nr-softmodem -O "$GNB_CONF" --sa --rfsim
  ;;
# -----------------------------------------------------------------------------
ue)
  cd "$OAI_DIR/cmake_targets/ran_build/build"
  sudo ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --sa \
    -O "$OAI_DIR/targets/PROJECTS/GENERIC-NR-5GC/CONF/nrUE_slice1.conf" \
    --rfsim --rfsimulator.serveraddr 127.0.0.1
  ;;
# -----------------------------------------------------------------------------
soak)
  # GATE for the old rig's fatal bug: v2.1.0 crashed in nr_dci_size /
  # get_ul_tdalist within seconds of single-UE attach. Those functions were
  # rewritten (sc_info refactor + NULL guards) by 2024.w28, but only a live
  # soak proves it on this rig. Keep ONE UE attached with light traffic for
  # >= 30 min; any segfault -> dmesg + addr2line (remember the PIE offset
  # correction) before proceeding.
  echo "[soak] gNB PID: $(pgrep -f nr-softmodem || echo NOT-RUNNING)"
  echo "[soak] watch: dmesg -w | grep -i 'nr-softmodem\|segfault'"
  ;;
# -----------------------------------------------------------------------------
tests)
  cd "$REPO_FRAMEWORK"
  export XAPP_OAI_PROTO_DIR="$PROTO_BUILDS"
  python3 -m pytest qoe_oran_framework/tests/ -q   # expect 140 passed
  ;;
# -----------------------------------------------------------------------------
probe)
  # Preconditions P2/P3/P4 live (E2 round-trip, KPM field population,
  # real per-UE demand). Run with >=1 UE attached AND real traffic flowing.
  cd "$REPO_FRAMEWORK"
  export XAPP_OAI_PROTO_DIR="$PROTO_BUILDS"
  python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --polls 120
  echo "[probe] P5 control check (values = your configured floor/ceiling only):"
  echo "  python3 -m qoe_oran_framework.scripts.probe_e2_preconditions \\"
  echo "      --send-control --sst 1 --sd 0 --min-ratio <floor> --max-ratio <cap>"
  ;;
# -----------------------------------------------------------------------------
caps)
  echo "[caps] Using the probe's P3 demand output, set per-slice max_ratio in"
  echo "       $REPO_FRAMEWORK/qoe_oran_framework/configs/*.yaml so the ceiling"
  echo "       BINDS (below aggregate offered demand). Handoff §4.4: the old rig"
  echo "       burned weeks on 10-20x headroom caps that never differentiated."
  echo "       Also re-tune Lmax / reward weights before trusting any carried"
  echo "       checkpoints — they encode the OLD rig's dynamics."
  ;;
# -----------------------------------------------------------------------------
smoke)
  # Short lb_only live smoke test — the papers' own heuristic baseline —
  # before any DQN/A2C/QoE-mode work resumes.
  cd "$REPO_FRAMEWORK"
  export XAPP_OAI_PROTO_DIR="$PROTO_BUILDS"
  bash qoe_oran_framework/scripts/run_saclb_live_testbed.sh   # review flags first
  ;;
# -----------------------------------------------------------------------------
*)
  cat <<EOF
Usage: ./rig_bringup.sh <stage>
Stages, in order (each gates the next):
  clone   - ORANSlice main + docker_open5gs(open5gs_slicing)
  build   - deps, gNB+nrUE build, regenerate ran_messages_pb2 from THIS tree
  core    - Open5GS up + per-slice subscriber provisioning
  gnb     - launch gNB (rfsim)             [separate terminal]
  ue      - launch one UE (rfsim)          [separate terminal]
  soak    - 30+ min single-UE attach stability gate (old segfault check)
  tests   - framework suite standalone (expect 140 passed)
  probe   - live P2/P3/P4 (+ optional P5) precondition probe
  caps    - re-tune ratio caps/Lmax from measured demand (manual)
  smoke   - short lb_only live baseline run
Only after ALL stages pass: resume DQN/A2C offline retraining and QoE-mode work.
EOF
  ;;
esac
