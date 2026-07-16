"""Paper #1 (MECON) comparator: Slice Admission Control only -- no LB term,
single gNB, no mMTC slice.

Papers #1/#2 compare A2C/DQN/Rainbow as "SAC" variants ("SAC" = Slice
Admission Control, the task, not an algorithm -- see Stage Zero plan).
Paper #1's contribution relative to paper #2 is dropping the LB reward term
and the mMTC slice / multi-gNB cluster, not a different algorithm. That
means the comparator needs no new environment or policy code: it is
DQNAdmissionPolicy run against configs/saclb_paper1_sac_only.yaml
(paper_variant: paper1, single gNB, urllc+embb only, reward.lb_coeff: 0.0),
which already turns off the LB term inside RANEnv/reward.py.
"""

from ..config import SacLbExperimentConfig, load_saclb_config
from ..env import request_state_dim
from ..policies.dqn_admission import DQNAdmissionPolicy


def load_sac_only_config(config_path: str) -> SacLbExperimentConfig:
    cfg = load_saclb_config(config_path)
    if cfg.paper_variant != "paper1":
        raise ValueError(
            f"sac_only comparator expects a paper_variant: paper1 config, got {cfg.paper_variant!r}"
        )
    if len(cfg.gnbs) != 1:
        raise ValueError(f"paper #1's SAC comparator is single-gNB; got {len(cfg.gnbs)} gNBs")
    if "mmtc" in cfg.slice_by_id:
        raise ValueError("paper #1's SAC comparator has no mMTC slice")
    return cfg


def build_sac_only_comparator(config_path: str, **policy_overrides) -> DQNAdmissionPolicy:
    cfg = load_sac_only_config(config_path)
    return DQNAdmissionPolicy(state_dim=request_state_dim(cfg), **policy_overrides)
