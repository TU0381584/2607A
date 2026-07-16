"""QoE mapper: IQX closed-form MOS estimate + LSTM temporal refinement.

Stage One (paper #4, tier-3 realisation of the survey's framework, paper #3
Section V-B / Table 12). Implements eq.(3):

    MOS_k = alpha_k - beta_k * ln(1 + Q_deg_k)
    Q_deg_k = gamma_k * L_k + delta_k * P_loss_k + epsilon_k / R_k

with PER-SLICE coefficients (alpha, beta, gamma, delta, epsilon) -- the same
QoS degradation has very different perceptual impact per traffic type (e.g.
eMBB users tolerate throughput dips far better than URLLC users tolerate
latency), so slices are calibrated independently (see calibration/).

===========================================================================
LATENCY INPUT LIMITATION (Stage One precondition-3 finding, verified before
any of this was written -- not assumed):
===========================================================================
Real OAI's E2SM-KPM (ran_messages.proto's ue_info_m) has NO literal per-flow
latency field anywhere -- checked directly against the .proto and the MAC
scheduler C source. The only latency-adjacent signal is
dl_mac_buffer_occupation, wired to a real scheduler counter
(sched_ctrl->num_total_bytes = sum of RLC TX buffer occupancy across logical
channels) -- not synthetic, not a stub. But it is INSTANTANEOUS and sparse
for low-rate slices: across a 3000-step live campaign, real (nonzero)
readings occurred on 99.9% of steps for eMBB, 53.3% for URLLC, only 13.5%
for mMTC (low-rate bursts are usually drained before the next poll catches
them mid-buffer). Treating "0 this step" as "confirmed zero latency" would
be wrong far more often than it's right for URLLC/mMTC.

LatencyProxy below holds the last observed nonzero reading across a rolling
window and reports staleness explicitly, so every consumer (the IQX closed
form and the LSTM alike) sees "last known value, N steps stale" rather than
a silently-fabricated zero. This is the same fallback discipline Stage Zero
already used for L_k(t) (see kpm_adapter.py's own dl_mac_buffer_occupation
fallback), extended with explicit staleness tracking because Stage One's
QoE mapper needs the temporal context, not just a point value.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

LATENCY_PROXY_LIMITATION = (
    "latency has no literal E2SM-KPM field on this OAI build (confirmed by "
    "direct proto + MAC-scheduler source inspection); dl_mac_buffer_occupation "
    "(a real scheduler counter, not synthetic) is used as a held, "
    "staleness-tagged proxy -- see qoe_mapper.py module docstring for the "
    "measured per-slice observation-rate evidence (eMBB 99.9%, URLLC 53.3%, "
    "mMTC 13.5% of steps)."
)


@dataclass
class IqxCoeffs:
    """Per-slice IQX coefficients, eq.(3). Defaults mirror the Stage One
    starter config.yaml's global values; calibration/fit_iqx.py overwrites
    these per slice against objective MOS labels."""

    alpha: float = 4.5
    beta: float = 0.6
    gamma: float = 1.0     # latency weight
    delta: float = 8.0     # packet-loss weight
    epsilon: float = 2.0   # inverse-throughput weight

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.alpha, self.beta, self.gamma, self.delta, self.epsilon)


DEFAULT_IQX_COEFFS: Dict[str, IqxCoeffs] = {
    "urllc": IqxCoeffs(alpha=4.5, beta=0.6, gamma=3.0, delta=6.0, epsilon=1.0),
    "embb": IqxCoeffs(alpha=4.5, beta=0.6, gamma=0.5, delta=4.0, epsilon=4.0),
    "mmtc": IqxCoeffs(alpha=4.5, beta=0.6, gamma=1.0, delta=8.0, epsilon=1.5),
}
# Priors only (latency/loss/throughput weighted per slice type, matching each
# slice's actual QoS sensitivity profile), NOT calibrated -- see
# calibration/fit_iqx.py for the coefficients actually fit against P.1203/ACR
# labels and their reported alignment metric. Using these un-calibrated
# priors directly (without running calibration first) must never be reported
# as a validated MOS mapping.


def iqx_mos(
    latency: "np.ndarray | float",
    packet_loss: "np.ndarray | float",
    throughput: "np.ndarray | float",
    coeffs: IqxCoeffs,
) -> np.ndarray:
    """Closed-form IQX MOS estimate, eq.(3).

    latency: seconds (or any consistent normalised unit -- calibration
      determines what unit the fitted gamma actually corresponds to).
    packet_loss: fraction in [0, 1].
    throughput: normalised, > 0 (clipped away from 0 to avoid blow-up).
    Returns MOS clipped to the standard [1, 5] ACR scale.
    """
    latency = np.maximum(np.asarray(latency, dtype=np.float64), 0.0)
    packet_loss = np.clip(np.asarray(packet_loss, dtype=np.float64), 0.0, 1.0)
    throughput = np.maximum(np.asarray(throughput, dtype=np.float64), 1e-3)
    q_deg = coeffs.gamma * latency + coeffs.delta * packet_loss + coeffs.epsilon / throughput
    mos = coeffs.alpha - coeffs.beta * np.log1p(np.maximum(q_deg, 0.0))
    return np.clip(mos, 1.0, 5.0)


@dataclass
class _LatencyProxyState:
    last_value: float = 0.0
    steps_since_observed: int = 0


class LatencyProxy:
    """Per-(gNB, slice) held latency-proxy tracker -- see module docstring.

    update() must be called every step with the raw, instantaneous
    dl_mac_buffer_occupation reading (0.0 when the fallback would trigger).
    Returns (held_value, steps_since_observed). steps_since_observed==0
    means this step's reading was itself real; a nonzero value tells the
    caller (IQX or the LSTM) exactly how stale the held estimate is, rather
    than hiding that behind a silently-reused zero.
    """

    def __init__(self, max_staleness: int = 20):
        self.max_staleness = max_staleness
        self._state: Dict[Tuple[str, str], _LatencyProxyState] = {}

    def update(self, gnb_id: str, slice_id: str, raw_buffer_occupation: float) -> Tuple[float, int]:
        key = (gnb_id, slice_id)
        st = self._state.setdefault(key, _LatencyProxyState())
        if raw_buffer_occupation > 0.0:
            st.last_value = float(raw_buffer_occupation)
            st.steps_since_observed = 0
        else:
            st.steps_since_observed = min(st.steps_since_observed + 1, self.max_staleness)
        return st.last_value, st.steps_since_observed

    def reset(self) -> None:
        self._state.clear()


class RollingKpmWindow:
    """Maintains the last `window` feature vectors per (gNB, slice) for
    QoEMapper's LSTM input. Left-pads with the earliest available sample
    (not zeros) until the window fills, so early-episode steps don't feed
    the LSTM a misleadingly all-zero history."""

    def __init__(self, window: int, feature_dim: int):
        self.window = window
        self.feature_dim = feature_dim
        self._buffers: Dict[Tuple[str, str], "collections.deque[np.ndarray]"] = {}

    def push(self, gnb_id: str, slice_id: str, features: np.ndarray) -> np.ndarray:
        key = (gnb_id, slice_id)
        buf = self._buffers.setdefault(key, collections.deque(maxlen=self.window))
        buf.append(np.asarray(features, dtype=np.float32))
        arr = list(buf)
        while len(arr) < self.window:
            arr.insert(0, arr[0])
        return np.stack(arr, axis=0)  # [window, feature_dim]

    def reset(self) -> None:
        self._buffers.clear()


QOE_FEATURE_NAMES = [
    "latency_proxy_norm", "staleness_norm", "packet_loss", "throughput_norm", "iqx_prior_mos",
]
QOE_FEATURE_DIM = len(QOE_FEATURE_NAMES)


if nn is not None:

    class QoEMapper(nn.Module):
        """Temporal refinement of the IQX prior.

        Ingests a window of [latency_proxy_norm, staleness_norm,
        packet_loss, throughput_norm, iqx_prior_mos] and outputs a refined
        MOS as a BOUNDED RESIDUAL correction on top of the IQX prior (the
        last feature column), not a from-scratch regression -- keeps the
        network's job tractable (learn where/why the closed form is wrong,
        not relearn the whole mapping) and keeps the closed form as a
        documented, inspectable fallback if the LSTM is ever unavailable
        or untrained (e.g. before calibration has run).
        """

        def __init__(self, window: int = 8, hidden: int = 32, input_dim: int = QOE_FEATURE_DIM):
            super().__init__()
            self.window = window
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: [batch, window, input_dim]. Returns MOS: [batch]."""
            out, _ = self.lstm(x)
            last_hidden = out[:, -1, :]
            correction = torch.tanh(self.head(last_hidden).squeeze(-1))  # bounded [-1, 1]
            prior = x[:, -1, -1]  # iqx_prior_mos is the last feature column
            mos = prior + correction
            return torch.clamp(mos, 1.0, 5.0)

else:  # pragma: no cover
    QoEMapper = None  # type: ignore


def build_qoe_features(
    latency_norm: float, staleness: int, max_staleness: int,
    packet_loss: float, throughput_norm: float, iqx_prior_mos: float,
) -> np.ndarray:
    """Assembles one step's QOE_FEATURE_DIM-length feature vector, in
    QOE_FEATURE_NAMES order. staleness is normalised to [0,1] by
    max_staleness so the network sees a bounded input regardless of how
    long a slice has gone unobserved."""
    staleness_norm = min(1.0, staleness / max(1, max_staleness))
    return np.array(
        [latency_norm, staleness_norm, packet_loss, throughput_norm, iqx_prior_mos],
        dtype=np.float32,
    )
