from qoe_oran_framework.config import QoeRewardWeights, RewardWeights, SliceSpec
from qoe_oran_framework.reward import check_violations, compute_qoe_reward, compute_step_reward
from qoe_oran_framework.types import ClusterState, SliceAggState

SPECS = {
    "urllc": SliceSpec("urllc", 1, 1, 30, 5, 60, 15, 0.8, 4.0, 1.0, 2.0),
    "embb": SliceSpec("embb", 1, 0, 60, 10, 90, 45, 2.0, 2.0, 1.0, 1.0),
}


def _state(urllc_queue=0.0, urllc_loss=0.0, fairness=1.0):
    return ClusterState(
        timestamp_s=0.0,
        per_gnb={
            "g0": {
                "urllc": SliceAggState("urllc", "g0", prb_used_ratio=0.3, congestion_level=0.3,
                                        queue_len_norm=min(2.0, urllc_queue), loss_proxy=urllc_loss,
                                        raw_queue_len_norm=urllc_queue),
                "embb": SliceAggState("embb", "g0", prb_used_ratio=0.2, congestion_level=0.3,
                                       queue_len_norm=0.0, loss_proxy=0.0, raw_queue_len_norm=0.0),
            }
        },
        fairness_ratio=fairness,
    )


def test_no_violation_when_within_budget():
    v = check_violations(_state(urllc_queue=0.5, urllc_loss=0.1), SPECS)
    assert v.violated["urllc"] is False


def test_queue_violation_flags_slice():
    v = check_violations(_state(urllc_queue=1.5), SPECS)
    assert v.violated["urllc"] is True


def test_loss_violation_flags_slice():
    v = check_violations(_state(urllc_loss=5.0), SPECS)  # > loss_budget_pct=0.8
    assert v.violated["urllc"] is True


def test_margin_is_graded_not_just_binary():
    """The whole point of margin: two slices can both be 'violated' (queue
    over budget, and both clipped to the same queue_len_norm=2.0 -- see
    kpm_adapter.py) while one was actually much closer to the budget the
    whole time. margin uses the unclipped raw_queue_len_norm specifically
    so it can still distinguish them where the clipped state value can't."""
    v_deep = check_violations(_state(urllc_queue=20.0), SPECS)  # way over budget
    v_barely = check_violations(_state(urllc_queue=1.01), SPECS)  # just over budget
    assert v_deep.violated["urllc"] is True
    assert v_barely.violated["urllc"] is True  # both read as violated, and both
    # clip to the SAME queue_len_norm=2.0 (min(2.0, 20.0) == min(2.0, would-be
    # 2.0 if urllc_queue were >=2))... but margin still tells them apart:
    assert v_deep.margin["urllc"] == 1.0 - 20.0
    assert abs(v_barely.margin["urllc"] - (1.0 - 1.01)) < 1e-9
    assert v_deep.margin["urllc"] < v_barely.margin["urllc"]


def test_margin_within_budget_is_graded_too():
    v = check_violations(_state(urllc_queue=0.5), SPECS)
    assert v.violated["urllc"] is False
    assert v.margin["urllc"] == 0.5  # 1.0 - 0.5


def test_margin_takes_worst_of_queue_and_loss_channels():
    # queue margin = 1-0.3=0.7, loss margin = 1-(5.0/0.8)=-5.25 -> worst is loss's
    v = check_violations(_state(urllc_queue=0.3, urllc_loss=5.0), SPECS)
    assert abs(v.margin["urllc"] - (1.0 - 5.0 / 0.8)) < 1e-9


def test_margin_defaults_to_one_for_slice_with_no_observed_state():
    v = check_violations(_state(), SPECS)
    assert v.margin["embb"] == 1.0


def test_paper1_reward_has_no_lb_term():
    state = _state(fairness=0.2)  # badly imbalanced, must not matter for paper1
    weights = RewardWeights(congestion_coeff=1.0, lb_coeff=5.0)
    reward, info = compute_step_reward(
        state, SPECS, accepted_counts={"urllc": 1, "embb": 0}, weights=weights, include_lb_term=False
    )
    assert info["lb_term"] == 0.0
    assert info["imbalance"] is None
    assert "limitation" not in info


def test_paper2_reward_penalizes_imbalance_not_fairness_directly():
    weights = RewardWeights(congestion_coeff=0.0, lb_coeff=1.0)
    balanced = _state(fairness=1.0)
    imbalanced = _state(fairness=0.2)
    r_balanced, info_balanced = compute_step_reward(
        balanced, SPECS, accepted_counts={}, weights=weights, include_lb_term=True
    )
    r_imbalanced, info_imbalanced = compute_step_reward(
        imbalanced, SPECS, accepted_counts={}, weights=weights, include_lb_term=True
    )
    # perfectly balanced (fairness=1.0) must score >= imbalanced (fairness=0.2):
    # this is the guard against the eq.2 ambiguity noted in reward.py.
    assert r_balanced > r_imbalanced
    assert info_balanced["imbalance"] == 0.0
    assert info_imbalanced["imbalance"] == 0.8
    assert "limitation" in info_imbalanced


def test_service_term_scales_with_priority_weight_and_accepted_count():
    reward, info = compute_step_reward(
        _state(), SPECS, accepted_counts={"urllc": 2, "embb": 0},
        weights=RewardWeights(congestion_coeff=0.0, lb_coeff=0.0), include_lb_term=False,
    )
    # urllc: omega=4.0, R_k=1.0, n=2 -> service_term=8.0
    assert info["service_term"] == 8.0


def test_congestion_term_scales_with_accepted_count_and_mean_congestion():
    reward, info = compute_step_reward(
        _state(), SPECS, accepted_counts={"urllc": 3, "embb": 0},
        weights=RewardWeights(congestion_coeff=2.0, lb_coeff=0.0), include_lb_term=False,
    )
    # mean_congestion over the 2 slice entries = 0.3, total_accepted = 3
    assert info["congestion_term"] == 2.0 * 0.3 * 3


# ---- compute_qoe_reward (eq.9, Stage One) ----

def test_qoe_reward_higher_mos_gives_higher_reward():
    weights = QoeRewardWeights(alpha=1.0, beta=0.0, gamma=0.0)
    state = _state()
    r_low, _ = compute_qoe_reward(
        state, SPECS, accepted_counts={}, mos_by_slice={"urllc": 1.0, "embb": 1.0}, qoe_weights=weights,
    )
    r_high, _ = compute_qoe_reward(
        state, SPECS, accepted_counts={}, mos_by_slice={"urllc": 5.0, "embb": 5.0}, qoe_weights=weights,
    )
    assert r_high > r_low


def test_qoe_reward_mos_term_is_normalised_one_to_five_scale():
    weights = QoeRewardWeights(alpha=1.0, beta=0.0, gamma=0.0)
    # mean_mos=3.0 -> mos_norm=(3-1)/4=0.5 -> reward=alpha*0.5=0.5, no cost/viol terms
    reward, info = compute_qoe_reward(
        _state(), SPECS, accepted_counts={}, mos_by_slice={"urllc": 3.0, "embb": 3.0}, qoe_weights=weights,
    )
    assert info["mean_mos"] == 3.0
    assert abs(reward - 0.5) < 1e-9


def test_qoe_reward_cost_term_scales_with_congestion_and_accepted_count():
    weights = QoeRewardWeights(alpha=0.0, beta=1.0, gamma=0.0)
    reward, info = compute_qoe_reward(
        _state(), SPECS, accepted_counts={"urllc": 3, "embb": 0},
        mos_by_slice={"urllc": 3.0, "embb": 3.0}, qoe_weights=weights,
    )
    # mean_congestion=0.3 (both slices' congestion_level=0.3), total_accepted=3
    assert info["cost"] == 0.3 * 3
    assert abs(reward - (-1.0 * info["cost"])) < 1e-9


def test_qoe_reward_sla_viol_uses_continuous_severity_not_flat_charge():
    weights = QoeRewardWeights(alpha=0.0, beta=0.0, gamma=1.0)
    # urllc deeply violated (queue=20 -> margin=1-20=-19, severity=19, clipped
    # per-slice contribution before averaging), embb fully compliant (severity=0)
    state = _state(urllc_queue=20.0)
    reward, info = compute_qoe_reward(
        state, SPECS, accepted_counts={}, mos_by_slice={"urllc": 3.0, "embb": 3.0}, qoe_weights=weights,
    )
    # mean severity over 2 slices = (19+0)/2 = 9.5, clipped to 1.0 (sla_viol is bounded [0,1])
    assert info["sla_viol"] == 1.0
    assert abs(reward - (-1.0)) < 1e-9


def test_qoe_reward_missing_slice_in_mos_dict_excluded_from_mean():
    weights = QoeRewardWeights(alpha=1.0, beta=0.0, gamma=0.0)
    reward, info = compute_qoe_reward(
        _state(), SPECS, accepted_counts={}, mos_by_slice={"urllc": 5.0}, qoe_weights=weights,
    )
    assert info["mean_mos"] == 5.0  # only urllc contributes, embb absent from mos_by_slice


def test_qoe_reward_carries_limitation():
    weights = QoeRewardWeights()
    _, info = compute_qoe_reward(
        _state(), SPECS, accepted_counts={}, mos_by_slice={"urllc": 3.0, "embb": 3.0}, qoe_weights=weights,
    )
    assert "limitation" in info and info["limitation"]
