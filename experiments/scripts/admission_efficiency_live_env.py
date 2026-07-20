#!/usr/bin/env python3
"""Shared factory for the LIVE-TRANSFERABLE admission-efficiency training
environment (experiments/configs/saclb_admission_efficiency_live_v1.yaml).

backlog_capacity=30.0, oversub_of_cap=1.2 validated via
live_scale_diagnostic.py (CAMPAIGN_LOG, 2026-07-20): accept_all's offline
eMBB compliance (22.9%) closely matches its ACTUAL measured live
compliance (23.3%, same session's live baseline run) -- the closest
offline-to-live match found in this entire workstream, and the first
config in this workstream validated against real rig data, not just
internal offline consistency.
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import SacLbExperimentConfig, load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent / "configs" / "saclb_admission_efficiency_live_v1.yaml"
)

BACKLOG_CAPACITY = 30.0
OVERSUB_OF_CAP = 1.2


def load_config() -> SacLbExperimentConfig:
    return load_saclb_config(CONFIG_PATH)


def make_env(
    seed: int,
    reward_mode: str = "qoe",
    backlog_capacity: float = BACKLOG_CAPACITY,
    oversub_of_cap: float = OVERSUB_OF_CAP,
    config_path: Optional[str] = None,
) -> RANEnv:
    cfg = load_saclb_config(config_path or CONFIG_PATH)
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
    mean_offered_ratio = {
        slice_id: min(0.98, oversub_of_cap * cfg.slice_by_id[slice_id].max_ratio_cap / 100.0)
        for slice_id in cfg.slice_by_id
    }
    kpm = ClosedLoopKpmSource(
        seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        mean_offered_ratio=mean_offered_ratio, backlog_capacity=backlog_capacity,
        sd_for_slice=sd_for_slice,
    )
    return RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)
