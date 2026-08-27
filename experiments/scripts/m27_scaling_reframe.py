#!/usr/bin/env python3
"""M27: rerun M6's N=7/19 topology-scaling campaign (GAT-CTDE /
independent DQN / single-agent DQN, fully-connected / ring / hex)
against RealisticServedKpmSource instead of the frozen ClosedLoopKpmSource,
to check whether M6's findings (single-agent DQN collapses totally at
N=19; GAT-CTDE collapses partially, ~35%; independent DQN never fully
collapses but never cleanly differentiates) hold, strengthen, or change
once each simulated gNB's congestion actually resembles this rig's own
measured live range (0.23-0.26 at 3-UE-like load, 0.60-0.69 at 6-UE-like
load) rather than ClosedLoopKpmSource's structurally-capped 0.03-0.09
(see M32-M34, commits 78e11a8/bc36bae/639ec89, and
docs/PAPER5_M27_M28_scope.md for why this is the reading of "offline
scaling reframe" this script proceeds on).

RealisticServedKpmSource was already written generic over gnb_ids (each
gNB gets its own independent load_alpha random walk, offered/backlog
state), so it needed no change to extend from N=1 (M34) to N=7/19 here.
Each simulated gNB reuses the SAME single-cell served-PRB anchors M34
measured on the real rig -- not a new, invented multi-gNB number, the
same already-validated per-cell calibration M6 itself already applies
per-gNB via gnb_load_multiplier for heterogeneity.

Does not touch qoe_oran_framework/ (frozen) or edit m6_run_experiment.py
-- imports it as a module and monkeypatches its make_kpm_source_factory
name (a plain module-global reassignment resolved at call time, the same
technique m32/m33's scripts already used for MEAN_OFFERED_RATIO), then
delegates to its own main() via sys.argv so all of its existing
orchestration, resumability (--resume-seeds), and results-writing is
reused unchanged, not reimplemented.

Usage:
    python3 experiments/scripts/m27_scaling_reframe.py \
        --config-path qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml \
        --topology fully_connected \
        --seeds 900 901 902 \
        --train-episodes 100 --eval-episodes 20 \
        --out-dir experiments/results/m27_scaling_reframe/n19_fully_connected \
        --resume-seeds
"""
import sys
from typing import Dict, List, Optional

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import m6_run_experiment as m6  # noqa: E402
from realistic_served_kpm_source import (  # noqa: E402
    RealisticServedKpmSource, SERVED_PRB_3UE, SERVED_PRB_6UE,
)


def _default_gnb_load_multiplier(gnb_ids: List[str], seed: int = 0) -> Dict[str, float]:
    return RealisticServedKpmSource._default_gnb_load_multiplier(gnb_ids, seed)


def make_realistic_kpm_source_factory(cfg, sd_for_slice, gnb_load_multiplier_mode: str = "default"):
    """Same gnb_load_multiplier_mode semantics as m6_run_experiment's own
    make_kpm_source_factory (default=seeded [0.6,1.4] per gNB via
    RealisticServedKpmSource's own static method; homogeneous=every gNB
    forced to 1.0), for direct comparability with M6/M7's established
    heterogeneity axis."""
    def factory(seed):
        kwargs = dict(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, sd_for_slice=sd_for_slice,
            served_prb_lo=dict(SERVED_PRB_3UE), served_prb_hi=dict(SERVED_PRB_6UE),
        )
        if gnb_load_multiplier_mode == "homogeneous":
            kwargs["gnb_load_multiplier"] = {g: 1.0 for g in cfg.gnb_ids}
        elif gnb_load_multiplier_mode != "default":
            raise ValueError(f"unknown gnb_load_multiplier_mode {gnb_load_multiplier_mode!r}")
        return RealisticServedKpmSource(**kwargs)
    return factory


if __name__ == "__main__":
    m6.make_kpm_source_factory = make_realistic_kpm_source_factory
    m6.main()
