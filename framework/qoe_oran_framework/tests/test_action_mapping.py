from qoe_oran_framework.action_mapping import AdmissionGate
from qoe_oran_framework.config import SliceSpec
from qoe_oran_framework.types import AdmissionRequest, ClusterState, SliceAggState

SPECS = {
    "urllc": SliceSpec("urllc", 1, 1, 30, 5, 60, 15, 0.8, 4.0, 1.0, 2.0),
    "embb": SliceSpec("embb", 1, 0, 60, 10, 90, 45, 2.0, 2.0, 1.0, 1.0),
}


def _empty_cluster_state(prb_used_ratio_urllc=0.0):
    return ClusterState(
        timestamp_s=0.0,
        per_gnb={
            "g0": {
                "urllc": SliceAggState("urllc", "g0", prb_used_ratio=prb_used_ratio_urllc,
                                        congestion_level=0.0, queue_len_norm=0.0),
                "embb": SliceAggState("embb", "g0", prb_used_ratio=0.0, congestion_level=0.0,
                                       queue_len_norm=0.0),
            }
        },
    )


def test_initial_ceilings_are_nominal_ratio():
    gate = AdmissionGate(SPECS, ["g0"])
    c = gate.ceiling_for("g0", "urllc")
    assert c.max_ratio == 30
    assert c.min_ratio == 5


def test_accept_nudges_ceiling_up_bounded_by_cap():
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    for _ in range(20):
        gate.apply([req], [1], _empty_cluster_state(), step=1)
    assert gate.ceiling_for("g0", "urllc").max_ratio == 60  # max_ratio_cap


def test_reject_nudges_ceiling_down_bounded_by_floor_and_counts_primary_block():
    """Reject both lowers the ceiling toward the floor AND (via
    KpmSource.notify_rejected(), wired in env.py) relieves backlog on the
    demand side. Tried decoupling these (reject only relieves backlog,
    never touches the ceiling) to see if it would help SLA compliance
    differentiate DQN from A2C -- it didn't move SLA compliance
    meaningfully, and it destabilized A2C's training into rejecting
    urllc/eMBB too (previously a robust, verified 0 blocks) -- so this
    stays coupled. See action_mapping.py's apply() comment."""
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    result = gate.apply([req], [0], _empty_cluster_state(), step=1)
    assert gate.ceiling_for("g0", "urllc").max_ratio == 25
    assert len(result.primary_blocks) == 1
    assert result.primary_blocks[0].kind == "primary_reject"
    assert result.accepted_counts["urllc"] == 0


def test_reject_never_goes_below_floor():
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    for _ in range(20):
        gate.apply([req], [0], _empty_cluster_state(), step=1)
    assert gate.ceiling_for("g0", "urllc").max_ratio == 5  # min_ratio_floor


def test_accept_counts_toward_accepted_counts_not_blocks():
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    result = gate.apply([req], [1], _empty_cluster_state(), step=1)
    assert result.accepted_counts["urllc"] == 1
    assert len(result.primary_blocks) == 0


def test_secondary_block_flags_demand_over_previous_ceiling():
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)  # ceiling starts at 30 -> ratio 0.30
    state_over_ceiling = _empty_cluster_state(prb_used_ratio_urllc=0.50)  # demand 50 > ceiling 30
    result = gate.apply([], [], state_over_ceiling, step=1)
    assert any(b.kind == "secondary_over_ceiling" and b.slice_id == "urllc" for b in result.secondary_blocks)


def test_secondary_blocks_never_included_in_primary_count():
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    state_over_ceiling = _empty_cluster_state(prb_used_ratio_urllc=0.50)
    result = gate.apply([], [], state_over_ceiling, step=1)
    assert len(result.primary_blocks) == 0
    assert len(result.secondary_blocks) >= 1


def test_reset_ceilings_restores_nominal_defaults():
    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    gate.apply([req], [1], _empty_cluster_state(), step=1)
    assert gate.ceiling_for("g0", "urllc").max_ratio == 35
    gate.reset_ceilings()
    assert gate.ceiling_for("g0", "urllc").max_ratio == 30


def test_mismatched_lengths_raise():
    import pytest

    gate = AdmissionGate(SPECS, ["g0"], step_ratio=5)
    req = AdmissionRequest("r1", "urllc", "g0", arrival_step=1)
    with pytest.raises(ValueError):
        gate.apply([req], [1, 0], _empty_cluster_state(), step=1)
