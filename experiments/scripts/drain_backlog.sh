#!/usr/bin/env bash
# Resets all 3 slices' ceilings to wide-open (min=0,max=100, the gNB's own
# boot default) and waits for backlog to drain, verified via a probe --
# run BETWEEN arms in the full campaign so no arm inherits the previous
# arm's queue state (per the campaign handover's explicit requirement).
set -uo pipefail
source /home/kmanojp/oranslice_rig/venv/bin/activate
source /home/kmanojp/oranslice_rig/env.sh
cd /home/kmanojp/oranslice_rig/framework

echo "[drain] opening all 3 slices wide (min=0,max=100)"
python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --send-control --sst 1 --sd 16777215 --min-ratio 0 --max-ratio 100 --polls 1 --interval-s 0.1 > /dev/null 2>&1
python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --send-control --sst 1 --sd 1 --min-ratio 0 --max-ratio 100 --polls 1 --interval-s 0.1 > /dev/null 2>&1
python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --send-control --sst 1 --sd 2 --min-ratio 0 --max-ratio 100 --polls 1 --interval-s 0.1 > /dev/null 2>&1

echo "[drain] waiting 20s for backlog to drain"
sleep 20

echo "[drain] verifying via probe"
python3 -m qoe_oran_framework.scripts.probe_e2_preconditions --polls 10 --interval-s 1.0 2>&1
