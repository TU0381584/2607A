import numpy as np

from qoe_oran_framework.calibration.acr_scoring import acr_score_mmtc, acr_score_urllc
from qoe_oran_framework.calibration.units import backlog_bytes_to_latency_s, prb_to_kbps


def test_acr_urllc_saturates_near_five_well_within_deadline():
    score = acr_score_urllc(latency_s=0.0, packet_loss=0.0)
    assert score > 4.9


def test_acr_urllc_saturates_near_one_well_beyond_deadline():
    score = acr_score_urllc(latency_s=0.05, packet_loss=0.01)
    assert score < 1.1


def test_acr_urllc_borderline_at_deadline_is_near_midpoint():
    score = acr_score_urllc(latency_s=0.005, packet_loss=0.0)  # exactly at default deadline
    assert 2.5 < score < 3.5


def test_acr_urllc_monotonic_in_latency():
    lo = acr_score_urllc(latency_s=0.001, packet_loss=0.0)
    hi = acr_score_urllc(latency_s=0.02, packet_loss=0.0)
    assert lo > hi


def test_acr_urllc_is_product_of_both_channels_both_must_pass():
    # good latency, bad loss -> should NOT score near 5 despite latency being fine
    score = acr_score_urllc(latency_s=0.0, packet_loss=0.01)
    assert score < 2.0


def test_acr_mmtc_saturates_near_five_when_clean():
    score = acr_score_mmtc(packet_loss=0.0, latency_s=0.0)
    assert score > 4.9


def test_acr_mmtc_loss_dominated_not_latency_dominated():
    # high latency, zero loss should still score reasonably (mMTC is delay-tolerant)
    high_latency_ok_loss = acr_score_mmtc(packet_loss=0.0, latency_s=5.0)
    low_latency_bad_loss = acr_score_mmtc(packet_loss=0.3, latency_s=0.0)
    assert high_latency_ok_loss > low_latency_bad_loss


def test_acr_scores_are_vectorised():
    scores = acr_score_urllc(
        latency_s=np.array([0.0, 0.05]), packet_loss=np.array([0.0, 0.01]),
    )
    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_prb_to_kbps_scales_linearly():
    assert prb_to_kbps(0.0) == 0.0
    assert prb_to_kbps(1.0) == prb_to_kbps(0.5) * 2


def test_backlog_bytes_to_latency_s_is_queueing_delay():
    # 1000 bytes = 8000 bits, at 8 kbps (8000 bps) -> 1 second
    latency = backlog_bytes_to_latency_s(backlog_bytes=1000.0, throughput_kbps=8.0)
    assert abs(float(latency) - 1.0) < 1e-6


def test_backlog_bytes_to_latency_s_handles_zero_throughput_without_error():
    latency = backlog_bytes_to_latency_s(backlog_bytes=1000.0, throughput_kbps=0.0)
    assert np.isfinite(latency)
