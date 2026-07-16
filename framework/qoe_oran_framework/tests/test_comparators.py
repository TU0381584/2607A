from pathlib import Path

import pytest

from qoe_oran_framework.comparators.lb_only_baseline import LbOnlyHeuristic
from qoe_oran_framework.comparators.sac_only import build_sac_only_comparator, load_sac_only_config
from qoe_oran_framework.config import (
    ArrivalConfig,
    EpisodeConfig,
    GnbSpec,
    RewardWeights,
    SacLbExperimentConfig,
    SliceSpec,
    load_saclb_config,
)
from qoe_oran_framework.env import request_state_dim
from qoe_oran_framework.types import AdmissionRequest, ClusterState, SliceAggState

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_sac_only_config_accepts_paper1_config():
    cfg = load_sac_only_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    assert cfg.paper_variant == "paper1"


def test_load_sac_only_config_rejects_paper2_config():
    with pytest.raises(ValueError):
        load_sac_only_config(str(CONFIGS_DIR / "saclb_offline_dqn.yaml"))


def test_build_sac_only_comparator_is_dqn_admission_policy():
    cfg = load_sac_only_config(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    policy = build_sac_only_comparator(str(CONFIGS_DIR / "saclb_paper1_sac_only.yaml"))
    assert policy.action_dim == 2
    assert policy.state_dim == request_state_dim(cfg)


def _lb_only_cfg(nominal_ratio=30):
    """Standalone config, independent of the real yaml files (which are
    subject to ongoing live-calibration tuning) -- gives full control over
    nominal_ratio for isolating each of LbOnlyHeuristic's two reject
    conditions."""
    return SacLbExperimentConfig(
        name="test", random_seed=1, paper_variant="paper1", B=100, Lmax=100,
        gnbs=[GnbSpec(gnb_id="g0", prb_capacity=100)],
        slices=[
            SliceSpec(
                slice_id="urllc", sst=1, sd=1, nominal_ratio=nominal_ratio,
                min_ratio_floor=5, max_ratio_cap=60, latency_budget_ms=15,
                loss_budget_pct=0.8, priority_weight=4.0, accept_reward=1.0,
                violation_penalty=2.0,
            )
        ],
        reward=RewardWeights(congestion_coeff=1.0, lb_coeff=0.0),
        arrivals=ArrivalConfig(synthetic_arrivals_per_step=2, max_pending_per_step=8, ceiling_step_ratio=5),
        episode=EpisodeConfig(step_seconds=5.0, steps_per_episode=60),
    )


def _cluster_state(prb_used_ratio, congestion_level=None, gnb_id="g0", slice_id="urllc"):
    if congestion_level is None:
        congestion_level = prb_used_ratio
    return ClusterState(
        timestamp_s=0.0,
        per_gnb={gnb_id: {slice_id: SliceAggState(slice_id, gnb_id, prb_used_ratio=prb_used_ratio,
                                                    congestion_level=congestion_level, queue_len_norm=0.0)}},
    )


def test_lb_only_admits_when_gnb_and_slice_both_below_thresholds():
    cfg = _lb_only_cfg(nominal_ratio=30)  # quota_ratio = 0.30
    heuristic = LbOnlyHeuristic(cfg, utilization_threshold=0.97)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    state = _cluster_state(prb_used_ratio=0.10, congestion_level=0.10)
    assert heuristic.decide([req], state) == [1]


def test_lb_only_rejects_when_gnb_aggregate_saturated():
    """Original condition, preserved: whole-gNB saturation still rejects,
    even if the specific slice's own usage is nowhere near its quota."""
    cfg = _lb_only_cfg(nominal_ratio=30)
    heuristic = LbOnlyHeuristic(cfg, utilization_threshold=0.97)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    state = _cluster_state(prb_used_ratio=0.05, congestion_level=0.98)
    assert heuristic.decide([req], state) == [0]


def test_lb_only_rejects_when_slice_at_its_own_quota_even_if_gnb_aggregate_is_low():
    """The fix: a slice already at/over its own nominal quota is rejected
    even though gNB-aggregate utilisation stays far below the saturation
    threshold -- this is exactly the blind spot the redesign closes (see
    lb_only_baseline.py docstring)."""
    cfg = _lb_only_cfg(nominal_ratio=3)  # quota_ratio = 0.03, matches live-calibrated tight quotas
    heuristic = LbOnlyHeuristic(cfg, utilization_threshold=0.97)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    state = _cluster_state(prb_used_ratio=0.05, congestion_level=0.15)  # real observed-demand scale
    assert heuristic.decide([req], state) == [0]


def test_lb_only_capacity_margin_scales_the_quota():
    cfg = _lb_only_cfg(nominal_ratio=3)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    state = _cluster_state(prb_used_ratio=0.05, congestion_level=0.15)

    strict = LbOnlyHeuristic(cfg, capacity_margin=1.0)  # quota_ratio 0.03 -> rejects
    assert strict.decide([req], state) == [0]

    generous = LbOnlyHeuristic(cfg, capacity_margin=3.0)  # quota_ratio 0.09 -> admits
    assert generous.decide([req], state) == [1]


def test_lb_only_handles_unknown_slice_gnb_gracefully():
    cfg = _lb_only_cfg()
    heuristic = LbOnlyHeuristic(cfg)
    req = AdmissionRequest("r1", "urllc", "unknown-gnb", arrival_step=1)
    actions = heuristic.decide([req], ClusterState(timestamp_s=0.0))
    assert actions == [1]  # nothing observed for this (slice, gNB) -> nothing to reject against
