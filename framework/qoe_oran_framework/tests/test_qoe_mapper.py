import numpy as np
import torch

from qoe_oran_framework.qoe_mapper import (
    DEFAULT_IQX_COEFFS, QOE_FEATURE_DIM, IqxCoeffs, LatencyProxy, QoEMapper,
    RollingKpmWindow, build_qoe_features, iqx_mos,
)


def test_iqx_mos_better_conditions_score_higher():
    coeffs = DEFAULT_IQX_COEFFS["urllc"]
    good = iqx_mos(latency=0.001, packet_loss=0.0, throughput=5.0, coeffs=coeffs)
    bad = iqx_mos(latency=0.5, packet_loss=0.3, throughput=0.1, coeffs=coeffs)
    assert good > bad


def test_iqx_mos_clipped_to_acr_scale():
    coeffs = IqxCoeffs(alpha=4.5, beta=0.6, gamma=1.0, delta=8.0, epsilon=2.0)
    extreme_bad = iqx_mos(latency=1000.0, packet_loss=1.0, throughput=1e-6, coeffs=coeffs)
    extreme_good = iqx_mos(latency=0.0, packet_loss=0.0, throughput=1000.0, coeffs=coeffs)
    assert 1.0 <= float(extreme_bad) <= 5.0
    assert 1.0 <= float(extreme_good) <= 5.0


def test_iqx_mos_vectorised():
    coeffs = DEFAULT_IQX_COEFFS["embb"]
    out = iqx_mos(
        latency=np.array([0.0, 0.1]), packet_loss=np.array([0.0, 0.02]),
        throughput=np.array([5.0, 1.0]), coeffs=coeffs,
    )
    assert out.shape == (2,)
    assert out[0] > out[1]


def test_latency_proxy_holds_last_real_reading():
    proxy = LatencyProxy(max_staleness=5)
    v0, s0 = proxy.update("g0", "urllc", 0.0)
    assert v0 == 0.0 and s0 == 1
    v1, s1 = proxy.update("g0", "urllc", 12.0)
    assert v1 == 12.0 and s1 == 0
    v2, s2 = proxy.update("g0", "urllc", 0.0)
    assert v2 == 12.0  # held, not reset to 0
    assert s2 == 1


def test_latency_proxy_staleness_caps_at_max():
    proxy = LatencyProxy(max_staleness=3)
    proxy.update("g0", "urllc", 5.0)
    for _ in range(10):
        _, staleness = proxy.update("g0", "urllc", 0.0)
    assert staleness == 3


def test_latency_proxy_tracks_keys_independently():
    proxy = LatencyProxy()
    proxy.update("g0", "urllc", 10.0)
    v_mmtc, s_mmtc = proxy.update("g0", "mmtc", 0.0)
    assert v_mmtc == 0.0 and s_mmtc == 1  # unaffected by urllc's own reading


def test_rolling_kpm_window_left_pads_until_full():
    w = RollingKpmWindow(window=4, feature_dim=3)
    first = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = w.push("g0", "urllc", first)
    assert out.shape == (4, 3)
    assert (out == first).all()  # left-padded with the same first sample


def test_rolling_kpm_window_slides_once_full():
    w = RollingKpmWindow(window=3, feature_dim=1)
    for v in [1.0, 2.0, 3.0, 4.0]:
        out = w.push("g0", "urllc", np.array([v], dtype=np.float32))
    assert list(out.flatten()) == [2.0, 3.0, 4.0]


def test_build_qoe_features_shape_and_staleness_normalisation():
    feat = build_qoe_features(
        latency_norm=0.5, staleness=10, max_staleness=20,
        packet_loss=0.01, throughput_norm=0.3, iqx_prior_mos=4.0,
    )
    assert feat.shape == (QOE_FEATURE_DIM,)
    assert feat[1] == 0.5  # staleness_norm = 10/20


def test_qoe_mapper_forward_shape_and_bounds():
    model = QoEMapper(window=8, hidden=16)
    x = torch.zeros((2, 8, QOE_FEATURE_DIM), dtype=torch.float32)
    x[:, -1, -1] = 3.0  # iqx_prior_mos column
    out = model(x)
    assert out.shape == (2,)
    assert torch.all(out >= 1.0) and torch.all(out <= 5.0)


def test_qoe_mapper_output_is_residual_on_prior_at_init():
    """A freshly-initialised (untrained) QoEMapper's correction should stay
    small relative to the prior it's meant to refine, not swamp it --
    sanity-checks the residual formulation itself, not learned behaviour."""
    model = QoEMapper(window=8, hidden=16)
    x = torch.zeros((1, 8, QOE_FEATURE_DIM), dtype=torch.float32)
    x[:, -1, -1] = 4.0
    with torch.no_grad():
        out = model(x)
    assert abs(float(out.item()) - 4.0) <= 1.0  # tanh-bounded correction, |correction|<=1
