import numpy as np

from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource


def _served_and_backlog(source, gnb_id, slice_id, n_steps):
    served, backlog = [], []
    for _ in range(n_steps):
        samples = source.poll()
        group = [s for s in samples if s.gnb_id == gnb_id and s.nssai_sd == {"embb": 0, "urllc": 1, "mmtc": 2}[slice_id]]
        served.append(sum(s.avg_prbs_dl for s in group))
        backlog.append(sum(s.dl_mac_buffer_occupation for s in group))
    return served, backlog


def test_served_never_exceeds_ceiling():
    source = ClosedLoopKpmSource(
        seed=1, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.9}, initial_ceiling_ratio=100.0,
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=20)  # sd=1 -> urllc
    served, _ = _served_and_backlog(source, "g0", "urllc", 20)
    assert max(served) <= 20.0 + 1e-6  # ceiling_prb = 20% of B=100


def test_low_ceiling_causes_backlog_growth():
    source = ClosedLoopKpmSource(
        seed=2, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.8}, offered_volatility=0.01,
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=10)  # far below offered demand (~80)
    _, backlog = _served_and_backlog(source, "g0", "urllc", 30)
    assert backlog[-1] > backlog[0]
    assert backlog[-1] > 10  # genuinely accumulating, not noise


def test_raising_ceiling_relieves_backlog():
    source = ClosedLoopKpmSource(
        seed=3, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.3}, offered_volatility=0.01,
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=10)  # below demand -> backlog builds
    _, backlog_low = _served_and_backlog(source, "g0", "urllc", 20)
    assert backlog_low[-1] > 0

    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=100)  # relieve constraint
    _, backlog_relieved = _served_and_backlog(source, "g0", "urllc", 30)
    assert backlog_relieved[-1] < backlog_low[-1]


def test_notify_rejected_relieves_backlog_beyond_what_ceiling_alone_would():
    """Reject must give admission decisions real leverage over a slice's own
    SLA compliance, not just block rate -- confirmed missing before this
    fix (DQN/A2C showed byte-identical SLA-compliance numbers across a 10x
    Lmax sweep, because send_control()'s ceiling-only path can only ever
    make backlog worse or leave it unchanged, never actively relieve it)."""
    source_a = ClosedLoopKpmSource(
        seed=4, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.3}, offered_volatility=0.01,
    )
    source_a.send_control("g0", 1, 1, min_ratio=5, max_ratio=10)  # below offered demand (~30)
    _, backlog_a = _served_and_backlog(source_a, "g0", "urllc", 10)

    source_b = ClosedLoopKpmSource(
        seed=4, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.3}, offered_volatility=0.01,
    )
    source_b.send_control("g0", 1, 1, min_ratio=5, max_ratio=10)
    for _ in range(10):
        source_b.notify_rejected("g0", "urllc", n_rejected=1)
        source_b.poll()
    backlog_b = source_b._backlog[("g0", "urllc")]

    assert backlog_b < backlog_a[-1]


def test_notify_rejected_ignores_unknown_key_and_nonpositive_count():
    source = ClosedLoopKpmSource(seed=5, gnb_ids=["g0"], slice_ids=["urllc"])
    source.notify_rejected("unknown-gnb", "urllc", n_rejected=5)
    source.notify_rejected("g0", "urllc", n_rejected=0)
    source.poll()  # must not raise, and backlog stays at its natural value
    assert source._pending_relief[("g0", "urllc")] == 0.0


def test_bler_rises_with_backlog():
    source = ClosedLoopKpmSource(
        seed=4, gnb_ids=["g0"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.9}, offered_volatility=0.01,
    )
    samples_before = source.poll()
    bler_before = np.mean([s.dl_bler for s in samples_before if s.nssai_sd == 1])

    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=5)  # heavily constrain
    for _ in range(20):
        samples = source.poll()
    bler_after = np.mean([s.dl_bler for s in samples if s.nssai_sd == 1])
    assert bler_after > bler_before


def test_send_control_only_affects_matching_slice():
    source = ClosedLoopKpmSource(
        seed=5, gnb_ids=["g0"], slice_ids=["urllc", "embb"],
        mean_offered_ratio={"urllc": 0.9, "embb": 0.2},
    )
    source.send_control("g0", 1, 1, min_ratio=5, max_ratio=15)  # urllc only
    served, _ = _served_and_backlog(source, "g0", "embb", 10)
    # embb ceiling untouched (still 100), so served should track its low offered demand, not be clamped to 15
    assert max(served) > 15


def test_reproducible_with_fixed_seed():
    def run():
        source = ClosedLoopKpmSource(seed=42, gnb_ids=["g0"], slice_ids=["urllc"])
        source.send_control("g0", 1, 1, min_ratio=5, max_ratio=30)
        served, backlog = _served_and_backlog(source, "g0", "urllc", 15)
        return served, backlog

    served_a, backlog_a = run()
    served_b, backlog_b = run()
    assert served_a == served_b
    assert backlog_a == backlog_b


def test_unknown_gnb_or_slice_in_send_control_is_ignored():
    source = ClosedLoopKpmSource(seed=6, gnb_ids=["g0"], slice_ids=["urllc"])
    source.send_control("unknown-gnb", 1, 1, min_ratio=5, max_ratio=10)  # should not raise
    source.poll()


def test_default_gnb_load_multiplier_creates_asymmetry_across_gnbs():
    source = ClosedLoopKpmSource(seed=7, gnb_ids=["g0", "g1", "g2"], slice_ids=["urllc"])
    mult = source._gnb_load_multiplier
    assert mult["g0"] == 1.0  # first gNB pinned, so single-gNB configs are unaffected
    assert len(set(mult.values())) > 1  # not all identical -> genuine heterogeneity


def test_single_gnb_config_unaffected_by_default_asymmetry():
    source = ClosedLoopKpmSource(seed=8, gnb_ids=["g0"], slice_ids=["urllc"])
    assert source._gnb_load_multiplier == {"g0": 1.0}


def test_explicit_gnb_load_multiplier_is_respected():
    source = ClosedLoopKpmSource(
        seed=9, gnb_ids=["g0", "g1"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.5}, gnb_load_multiplier={"g0": 1.0, "g1": 2.0},
        offered_volatility=0.0,
    )
    served_g0, _ = _served_and_backlog(source, "g0", "urllc", 1)
    served_g1, _ = _served_and_backlog(source, "g1", "urllc", 1)
    # both start unconstrained (ceiling=100), so served == offered on step 1
    assert served_g1[0] > served_g0[0]


def test_asymmetric_load_produces_differing_utilisation_across_gnbs():
    source = ClosedLoopKpmSource(
        seed=10, gnb_ids=["g0", "g1"], slice_ids=["urllc"],
        mean_offered_ratio={"urllc": 0.4}, gnb_load_multiplier={"g0": 0.5, "g1": 1.3},
        offered_volatility=0.01,
    )
    served_g0, _ = _served_and_backlog(source, "g0", "urllc", 15)
    served_g1, _ = _served_and_backlog(source, "g1", "urllc", 15)
    assert sum(served_g1[-5:]) > sum(served_g0[-5:])  # g1's higher load shows up in served PRBs
