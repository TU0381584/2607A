#!/usr/bin/env bash
# Paper #5 independent-seed replication: retrains GAT-CTDE/independent-DQN/
# single-agent-DQN (M2), the federated+DP sweep (M3), and the disruption
# campaign (M4) from scratch under a DISJOINT seed range from the one
# already used (900-929 / 900-909), rather than re-running the same
# seeds. This is a stronger check than reproduce_paper5_full.sh's
# same-seed determinism verification -- it tests whether the paper's
# statistical findings (paired significance, threshold-like severity
# curves) hold up under an independent sample, not just that the
# pipeline is deterministic.
#
# M1 is deliberately NOT part of this script: it evaluates paper #4's
# real, historical live-hardware checkpoints (seeds 256-261) against
# already-recorded live traffic -- there is no "fresh seed" retraining
# of a live testbed run to redo. M1's own reproducibility is already
# covered by reproduce_paper5_full.sh.
#
# Runs into an ISOLATED output root (default
# experiments/results/fresh_seed_retrain/), never touching the committed
# results or the same-seed reproduction_check/ directory. Every
# sub-stage is merge-safe/resumable exactly like reproduce_paper5_full.sh.
#
# Usage:
#   experiments/scripts/reproduce_paper5_fresh_seeds.sh [OUT_ROOT] [SEED_BASE]
#   OUT_ROOT defaults to experiments/results/fresh_seed_retrain
#   SEED_BASE defaults to 1000 (disjoint from 900-929)

set -uo pipefail

REPO_ROOT="/home/kmanojp/oranslice_rig"
OUT_ROOT="${1:-${REPO_ROOT}/experiments/results/fresh_seed_retrain}"
SEED_BASE="${2:-1000}"
LOG_DIR="${REPO_ROOT}/experiments/logs/fresh_seed_retrain"
PY="${REPO_ROOT}/venv/bin/python3"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
cd "${REPO_ROOT}/framework"

echo "[fresh-seed] OUT_ROOT=${OUT_ROOT} SEED_BASE=${SEED_BASE} (M2: ${SEED_BASE}-$((SEED_BASE+29)), M3/M4: ${SEED_BASE}-$((SEED_BASE+9)))"
date

# ---------------------------------------------------------------------
# M2: full 30-seed x 3-arm campaign, fresh seed range.
# ---------------------------------------------------------------------
echo "[fresh-seed] === M2: 30-seed x 3-arm campaign, seeds ${SEED_BASE}-$((SEED_BASE+29)) ==="
"${PY}" ../experiments/scripts/m2_seed_campaign.py \
    --out-dir "${OUT_ROOT}/m2_campaign" \
    --arms gat_ctde independent_dqn single_agent_dqn \
    --seed-base "${SEED_BASE}" \
    2>&1 | tee "${LOG_DIR}/m2_campaign.log"
echo "[fresh-seed] M2 done"
date

# ---------------------------------------------------------------------
# M3: full 10-seed x 5-sigma sweep, same fresh seed base's first 10 --
# needed so the Federation Cost comparison can pair against this same
# run's own M2 gat_ctde seeds, not the original 900-909 centralized data.
# ---------------------------------------------------------------------
echo "[fresh-seed] === M3: 10-seed x 5-sigma privacy sweep, seeds ${SEED_BASE}-$((SEED_BASE+9)) ==="
"${PY}" ../experiments/scripts/m3_privacy_sweep.py \
    --out-dir "${OUT_ROOT}/m3_campaign" \
    --seed-base "${SEED_BASE}" \
    2>&1 | tee "${LOG_DIR}/m3_campaign.log"
echo "[fresh-seed] M3 done"
date

# ---------------------------------------------------------------------
# M4: disruption campaign against this run's OWN fresh M2/M3 checkpoints.
# ---------------------------------------------------------------------
FRESH_SEEDS=$(seq "${SEED_BASE}" $((SEED_BASE+9)))
echo "[fresh-seed] === M4: disruption-resilience campaign, seeds ${SEED_BASE}-$((SEED_BASE+9)) ==="
"${PY}" ../experiments/scripts/m4_seed_campaign.py \
    --out-dir "${OUT_ROOT}/m4_campaign" \
    --seeds ${FRESH_SEEDS} \
    --m2-campaign-dir "${OUT_ROOT}/m2_campaign" \
    --m3-campaign-dir "${OUT_ROOT}/m3_campaign" \
    2>&1 | tee "${LOG_DIR}/m4_campaign.log"
echo "[fresh-seed] M4 done"
date

echo "[fresh-seed] === ALL STAGES COMPLETE ==="
echo "[fresh-seed] Analyze with:"
echo "  python3 experiments/scripts/m2_correctness_metrics.py --campaign-dir ${OUT_ROOT}/m2_campaign --results ${OUT_ROOT}/m2_campaign/campaign_results.json"
echo "  python3 experiments/scripts/m4_correctness_metrics.py --m4-results ${OUT_ROOT}/m4_campaign/campaign_results.json --m4-campaign-dir ${OUT_ROOT}/m4_campaign --m2-campaign-dir ${OUT_ROOT}/m2_campaign --m3-campaign-dir ${OUT_ROOT}/m3_campaign"
