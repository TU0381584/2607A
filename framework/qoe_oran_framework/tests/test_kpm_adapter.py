from qoe_oran_framework.config import SliceSpec
from qoe_oran_framework.kpm_adapter import (
    aggregate_slice_state,
    build_cluster_state,
    compute_fairness_ratio,
    resolve_slice_id,
)
from qoe_oran_framework.types import UeSample

SPECS = {
    "urllc": SliceSpec("urllc", 1, 1, 30, 5, 60, 15, 0.8, 4.0, 1.0, 2.0),
    "embb": SliceSpec("embb", 1, 0, 60, 10, 90, 45, 2.0, 2.0, 1.0, 1.0),
    "mmtc": SliceSpec("mmtc", 1, 2, 10, 5, 40, 65, 3.5, 1.0, 1.0, 0.5),
}


def test_resolve_slice_id_matches_oranslice_drl_convention():
    assert resolve_slice_id(0) == "embb"
    assert resolve_slice_id(1) == "urllc"
    assert resolve_slice_id(2) == "mmtc"
    assert resolve_slice_id(99) == "embb"  # documented fallback


def test_prb_used_ratio_is_direct_sum_over_b():
    ues = [
        UeSample(rnti=1, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=10.0, gnb_id="g0"),
        UeSample(rnti=2, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=5.0, gnb_id="g0"),
    ]
    agg, limitations = aggregate_slice_state(ues, "g0", SPECS, B=100, Lmax=100, timestamp_s=0.0)
    assert agg["urllc"].prb_used_ratio == (10.0 + 5.0) / 100
    assert agg["embb"].prb_used_ratio == 0.0
    assert agg["urllc"].n_ues == 2


def test_queue_len_uses_buffer_occupation_when_present():
    ues = [
        UeSample(
            rnti=1, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=1.0,
            gnb_id="g0", dl_mac_buffer_occupation=20.0,
        )
    ]
    agg, limitations = aggregate_slice_state(ues, "g0", SPECS, B=100, Lmax=100, timestamp_s=0.0)
    assert agg["urllc"].queue_len_norm == 0.2
    assert agg["urllc"].queue_used_fallback is False
    assert not limitations


def test_queue_len_falls_back_and_flags_limitation_when_buffer_occupation_is_zero():
    ues = [
        UeSample(
            rnti=1, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=1.0,
            gnb_id="g0", dl_mac_buffer_occupation=0.0, dl_errors=1.0, dl_bler=2.0,
        )
    ]
    agg, limitations = aggregate_slice_state(ues, "g0", SPECS, B=100, Lmax=100, timestamp_s=0.0)
    assert agg["urllc"].queue_used_fallback is True
    assert agg["urllc"].queue_len_norm == (1.0 + 2.0) / 100
    assert any("backlog proxy" in msg for msg in limitations)


def test_congestion_level_is_clipped_cluster_wide_utilisation():
    ues = [
        UeSample(rnti=1, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=60.0, gnb_id="g0"),
        UeSample(rnti=2, timestamp_s=0.0, nssai_sst=1, nssai_sd=0, avg_prbs_dl=60.0, gnb_id="g0"),
    ]
    agg, _ = aggregate_slice_state(ues, "g0", SPECS, B=100, Lmax=100, timestamp_s=0.0)
    # raw sum would be 1.2, must clip to 1.0
    assert agg["urllc"].congestion_level == 1.0
    assert agg["embb"].congestion_level == 1.0  # duplicated across slices at the same gNB


def test_empty_slice_groups_still_produce_zeroed_entries():
    agg, _ = aggregate_slice_state([], "g0", SPECS, B=100, Lmax=100, timestamp_s=0.0)
    assert set(agg.keys()) == set(SPECS.keys())
    assert all(a.prb_used_ratio == 0.0 for a in agg.values())


def test_fairness_ratio_single_gnb_is_one():
    per_gnb = {"g0": {"urllc": type("A", (), {"raw_congestion_level": 0.5})()}}
    assert compute_fairness_ratio(per_gnb) == 1.0


def test_fairness_ratio_min_over_max_across_gnbs():
    per_gnb = {
        "g0": {"urllc": type("A", (), {"raw_congestion_level": 0.3})()},
        "g1": {"urllc": type("A", (), {"raw_congestion_level": 0.6})()},
    }
    assert compute_fairness_ratio(per_gnb) == 0.5


def test_fairness_ratio_uses_unclipped_congestion_not_the_state_feature():
    """Two gNBs both individually oversubscribed past 100% (congestion_level
    clipped to 1.0 for both, indistinguishable) must still show up as
    distinct via raw_congestion_level -- this is the fix for fairness_ratio
    degenerating to a single-gNB signal once >=2 gNBs saturate the clip."""
    per_gnb = {
        "g0": {"urllc": type("A", (), {"congestion_level": 1.0, "raw_congestion_level": 1.10})()},
        "g1": {"urllc": type("A", (), {"congestion_level": 1.0, "raw_congestion_level": 1.375})()},
    }
    assert compute_fairness_ratio(per_gnb) == 1.10 / 1.375


def test_build_cluster_state_routes_samples_by_gnb_id():
    ues = [
        UeSample(rnti=1, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=30.0, gnb_id="g0"),
        UeSample(rnti=2, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=60.0, gnb_id="g1"),
    ]
    cs = build_cluster_state(ues, ["g0", "g1"], SPECS, B=100, Lmax=100, timestamp_s=0.0)
    assert cs.per_gnb["g0"]["urllc"].prb_used_ratio == 0.3
    assert cs.per_gnb["g1"]["urllc"].prb_used_ratio == 0.6
    assert cs.fairness_ratio == 0.5


def test_build_cluster_state_defaults_untagged_samples_to_first_gnb():
    ues = [UeSample(rnti=1, timestamp_s=0.0, nssai_sst=1, nssai_sd=1, avg_prbs_dl=10.0, gnb_id="")]
    cs = build_cluster_state(ues, ["g0", "g1"], SPECS, B=100, Lmax=100, timestamp_s=0.0)
    assert cs.per_gnb["g0"]["urllc"].n_ues == 1
    assert cs.per_gnb["g1"]["urllc"].n_ues == 0
