#!/usr/bin/env python3
"""Shared factory for the "admission efficiency under overload" objective's
offline environment (experiments/configs/saclb_admission_efficiency_v1.yaml).

Does NOT modify any frozen qoe_oran_framework/ source -- reuses
RANEnv/ClosedLoopKpmSource exactly as-is, supplying the two constructor
kwargs that aren't (and structurally can't be, per ClosedLoopKpmSource's
own signature) expressed in the YAML config file itself:
  - mean_offered_ratio: derived here as OVERSUB_OF_CAP x each slice's
    max_ratio_cap (NOT nominal_ratio -- see the config file's own
    docstring for why this differs from qoe_oran_framework/scripts/
    train_offline.py's OVERSUBSCRIPTION_FACTOR convention).
  - sd_for_slice: the config's REAL SliceSpec.sd values. Required as of
    the 2026-07-20 fix (commit c523e02) -- ClosedLoopKpmSource's default
    {embb:0,urllc:1,mmtc:2} map does NOT match this rig's real embb sd
    (16777215), and omitting this silently makes embb's ceiling a no-op
    (see CAMPAIGN_LOG.md).
  - backlog_capacity: ClosedLoopKpmSource has no YAML-configurable
    equivalent; validated at 1000.0 (see CONFIG's own header comment).

Usage (as a library):
    from admission_efficiency_env import make_env
    env = make_env(seed=256, reward_mode="qoe")
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import SacLbExperimentConfig, load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CONFIG_PATH = str(
    Path(__file__).resolve().parent.parent / "configs" / "saclb_admission_efficiency_v1.yaml"
)

# Validated in CAMPAIGN_LOG.md (2026-07-20, post sd-bug-fix sweep) as the
# combination that differentiates all 3 slices simultaneously and
# monotonically (accept_all > threshold_like > reject_all on compliance).
# NOT yet re-validated against a wider sweep -- see the config file's
# own "future work" notes on ceiling_step_ratio / a real beta sweep.
BACKLOG_CAPACITY = 1000.0
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
