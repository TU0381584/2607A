#!/usr/bin/env bash
# Paper #5 (M1-M4) full reproduction, from a fresh checkout to final
# numbers. Runs into an ISOLATED output root (default:
# experiments/results/reproduction_check/) -- never touches or overwrites
# the already-committed experiments/results/{m1_recalibration,m2_campaign,
# m3_campaign,m4_campaign}/ directories, so this can be run at any time
# without risking the paper's already-verified data.
#
# See docs/PAPER5_REPRODUCIBILITY.md for the full provenance table (which
# manuscript number comes from which of these commands) and for how to
# compare this script's output against the committed results.
#
# M1 is cheap (grid search + held-out eval against already-frozen,
# already-committed live_campaign_v2 checkpoints -- no training). M2/M3
# are expensive: M2 is a 30-seed x 3-arm x 300-train+50-eval-episode
# campaign (~5-8 hours on an 8-core machine, calibrated empirically --
# one gat_ctde seed alone took 317s), M3 is a 10-seed x 5-sigma federated
# campaign of similar or greater scale. M4 (eval-only, no training) is
# fast (~18 minutes for the full 330-cell sweep) and, critically, is run
# HERE against the freshly-produced M2/M3 checkpoints (via
# --m2-campaign-dir/--m3-campaign-dir), not the committed ones -- so a
# clean run of this script all the way through is a genuine, complete,
# from-scratch verification of the entire M1-M4 chain, not just M4 in
# isolation against pre-existing checkpoints.
#
# Every M2/M3/M4 sub-stage is merge-safe/resumable (campaign_results.json
# skips already-completed cells) -- if this script is interrupted, re-run
# it with the same OUT_ROOT and it picks up where it left off.
#
# Usage:
#   experiments/scripts/reproduce_paper5_full.sh [OUT_ROOT]
#   OUT_ROOT defaults to experiments/results/reproduction_check

set -euo pipefail
# -e: a killed/failed stage (e.g. an interrupted python3 | tee pipeline --
# real incident this session: a partial kill of one stage's python process
# left the wrapper script silently continuing into the NEXT stage against
# incomplete prior output, since pipefail alone does not stop the script,
# only marks that one pipeline's exit status) must stop this script, not
# silently fall through to the next stage against incomplete data.

REPO_ROOT="/home/kmanojp/oranslice_rig"
OUT_ROOT="${1:-${REPO_ROOT}/experiments/results/reproduction_check}"
LOG_DIR="${REPO_ROOT}/experiments/logs/reproduction"
PY="${REPO_ROOT}/venv/bin/python3"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
cd "${REPO_ROOT}/framework"

echo "[reproduce] OUT_ROOT=${OUT_ROOT}"
echo "[reproduce] logs under ${LOG_DIR}"
date

# ---------------------------------------------------------------------
# M1: recalibration grid search + held-out eval (cheap -- no training,
# evaluates the already-frozen, already-committed live checkpoints)
# ---------------------------------------------------------------------
echo "[reproduce] === M1: extract live traces + grid search + held-out eval ==="
"${PY}" ../experiments/scripts/m1_extract_live_traces.py \
    --out "${OUT_ROOT}/m1/live_traces.json" \
    2>&1 | tee "${LOG_DIR}/m1_extract_live_traces.log"

"${PY}" ../experiments/scripts/m1_fit_recalibration.py \
    --live-traces "${OUT_ROOT}/m1/live_traces.json" \
    --out "${OUT_ROOT}/m1/fit_search.json" \
    --scratch-root /tmp/m1_repro_scratch \
    2>&1 | tee "${LOG_DIR}/m1_fit_recalibration.log"

# Best-fit config per docs/PAPER5_M1_recalibration.md's Result section:
# backlog_capacity=3200, drift_coef=0.1, offered_volatility=0.04, ar1_coef=0.0.
# Baseline config reproduces the frozen ClosedLoopKpmSource bit-for-bit
# (bc=2000, the class default) -- both re-run so the comparison in this
# doc's own final step is apples-to-apples, not read off the old run.
"${PY}" ../experiments/scripts/m1_run_held_out_eval.py \
    --backlog-capacity 3200 --drift-coef 0.1 --offered-volatility 0.04 --ar1-coef 0.0 \
    --out-dir "${OUT_ROOT}/m1/held_out_recalibrated" --tag recalibrated \
    2>&1 | tee "${LOG_DIR}/m1_held_out_recalibrated.log"

"${PY}" ../experiments/scripts/m1_run_held_out_eval.py \
    --backlog-capacity 2000 --drift-coef 0.1 --offered-volatility 0.04 --ar1-coef 0.0 \
    --out-dir "${OUT_ROOT}/m1/held_out_baseline" --tag baseline \
    2>&1 | tee "${LOG_DIR}/m1_held_out_baseline.log"

echo "[reproduce] M1 done"
date

# ---------------------------------------------------------------------
# M2: full 30-seed x 3-arm campaign (gat_ctde, independent_dqn,
# single_agent_dqn) -- SLOW, this is the real training campaign.
# ---------------------------------------------------------------------
echo "[reproduce] === M2: 30-seed x 3-arm campaign ==="
"${PY}" ../experiments/scripts/m2_seed_campaign.py \
    --out-dir "${OUT_ROOT}/m2_campaign" \
    --arms gat_ctde independent_dqn single_agent_dqn \
    2>&1 | tee "${LOG_DIR}/m2_campaign.log"
echo "[reproduce] M2 done"
date

# ---------------------------------------------------------------------
# M3: full 10-seed x 5-sigma federated + DP privacy sweep -- SLOW.
# ---------------------------------------------------------------------
echo "[reproduce] === M3: 10-seed x 5-sigma privacy sweep ==="
"${PY}" ../experiments/scripts/m3_privacy_sweep.py \
    --out-dir "${OUT_ROOT}/m3_campaign" \
    2>&1 | tee "${LOG_DIR}/m3_campaign.log"
echo "[reproduce] M3 done"
date

# ---------------------------------------------------------------------
# M4: full 33-condition x 10-seed disruption campaign -- fast (eval
# only), and run against the FRESH M2/M3 checkpoints just produced
# above, not the committed ones, for a genuine end-to-end verification.
# ---------------------------------------------------------------------
echo "[reproduce] === M4: disruption-resilience campaign (against fresh M2/M3 checkpoints) ==="
"${PY}" ../experiments/scripts/m4_seed_campaign.py \
    --out-dir "${OUT_ROOT}/m4_campaign" \
    --m2-campaign-dir "${OUT_ROOT}/m2_campaign" \
    --m3-campaign-dir "${OUT_ROOT}/m3_campaign" \
    2>&1 | tee "${LOG_DIR}/m4_campaign.log"
echo "[reproduce] M4 done"
date

echo "[reproduce] === ALL STAGES COMPLETE ==="
echo "[reproduce] Compare against committed results with:"
echo "  python3 experiments/scripts/compare_reproduction.py --repro-root ${OUT_ROOT}"
