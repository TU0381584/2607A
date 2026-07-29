#!/usr/bin/env python3
"""Stage 5: offline training environment factory with demand scaled to
match REAL, live-observed traffic (not qoe_oran_framework/scripts/
train_offline.py's frozen OVERSUBSCRIPTION_FACTOR * nominal_ratio
formula, which for this campaign's nominal_ratio=3/2/2 produces a mean
offered ratio of ~0.0375/0.025/0.025 -- 4-6x smaller than what this rig's
real traffic actually produces).

mean_offered_ratio here is set directly from repeated, direct
probe_e2_preconditions.py measurements on the real rig this session
(eMBB mean ~14-15.6 PRB / B=100 -> ~0.15; URLLC/mMTC pinned at exactly
5.0 PRB every single probe -> ~0.05), NOT derived from nominal_ratio or
max_ratio_cap by any formula -- ClosedLoopKpmSource's constructor accepts
mean_offered_ratio directly (a dict), so this is the one honest way to
make offline demand track real demand without touching the frozen
formula or the config's ceiling/nominal semantics (which must stay at
their real, live-matching values -- saclb_campaign_v2.yaml's
nominal_ratio=3/2/2, max_ratio_cap=12/4/3 -- for a policy trained here to
be about the SAME MDP the live rig actually presents, not a rescaled
stand-in).

Mirrors admission_efficiency_env.py's precedent (non-frozen factory,
explicit sd_for_slice from the real config, not ClosedLoopKpmSource's
{embb:0,urllc:1,mmtc:2} default).

Usage (as a library):
    from live_scale_offline_env import make_env
    env = make_env(seed=256, reward_mode="qoe")
"""
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import SacLbExperimentConfig, load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2.yaml"

# Real, live-observed prb_used_ratio (this rig, this session, repeated
# probe_e2_preconditions.py runs -- see docs/STAGE5_recalibration.md):
MEAN_OFFERED_RATIO: Dict[str, float] = {"embb": 0.15, "urllc": 0.05, "mmtc": 0.05}

# Defaults matching saclb_campaign_v2.yaml (Lmax=10, ClosedLoopKpmSource's
# own backlog_capacity default=200) -- overridable per validate_offline_env.py's
# sweep if these turn out to saturate at the REAL cap scale (12/4/3), same
# validation discipline the admission-efficiency workstream used before
# freezing its own (very different, rescaled-cap) config.
DEFAULT_BACKLOG_CAPACITY = 200.0


def load_config(config_path: Optional[str] = None) -> SacLbExperimentConfig:
    return load_saclb_config(config_path or CONFIG_PATH)


def make_env(
    seed: int,
    reward_mode: str = "qoe",
    backlog_capacity: float = DEFAULT_BACKLOG_CAPACITY,
    mean_offered_ratio: Optional[Dict[str, float]] = None,
    config_path: Optional[str] = None,
) -> RANEnv:
    cfg = load_saclb_config(config_path or CONFIG_PATH)
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
    kpm = ClosedLoopKpmSource(
        seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        mean_offered_ratio=mean_offered_ratio or MEAN_OFFERED_RATIO,
        backlog_capacity=backlog_capacity, sd_for_slice=sd_for_slice,
    )
    return RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)
