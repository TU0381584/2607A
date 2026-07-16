"""Application-specific, ACR-style (1-5) calibration scoring for URLLC and
mMTC traffic, where no ITU-T perceptual-quality standard applies (P.1203 is
scoped to progressive-download/adaptive audiovisual streaming; URLLC/mMTC
traffic in this framework is task-oriented -- control loops, telemetry --
not perceptual media).

These are DOCUMENTED MODELLING ASSUMPTIONS, not a standard, and are
reported as such in every Omega-tuple that uses them. The design principle
for each slice type follows how its actual application class experiences
QoS degradation, not a generic smooth QoS-to-MOS curve:

  URLLC: control-loop / real-time applications (industrial control, haptic/
  tactile feedback, remote-operation robotics) experience a missed latency
  deadline as a near-BINARY functional failure, not a gradual degradation --
  "the packet arrived 2ms late" and "the packet arrived 50ms late" are both
  simply "too late" to a closed control loop. Scored with a steep sigmoid
  centred on the configured deadline, not a linear ramp.

  mMTC: periodic, delay-tolerant sensor/telemetry traffic. Successful
  delivery (loss) dominates perceived quality; latency is heavily
  discounted since these applications are not time-critical. Scored with a
  loss-dominated, latency-lightly-weighted sigmoid.
"""
from __future__ import annotations

import numpy as np

ACR_SCORING_LIMITATION = (
    "URLLC/mMTC calibration labels are application-specific, threshold-based "
    "ACR-style scores (steep sigmoid around each slice's own latency/loss "
    "deadline), not an ITU-T standard measurement -- P.1203 does not apply "
    "to non-perceptual, task-oriented traffic. See "
    "calibration/acr_scoring.py module docstring for the full rationale."
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def acr_score_urllc(
    latency_s: "np.ndarray | float",
    packet_loss: "np.ndarray | float",
    deadline_s: float = 0.005,
    loss_budget: float = 0.001,
    latency_transition_s: float = 0.001,
    loss_transition: float = 0.0005,
) -> np.ndarray:
    """URLLC: steep sigmoid around the latency deadline (a missed deadline
    is a near-binary functional failure for a control loop), a similarly
    steep sigmoid around the loss budget (a dropped control packet is
    rarely "partially" tolerable). Each channel's sharpness is set from its
    own transition width (roughly "how far from the deadline before the
    verdict is decided"), not a single shared constant -- latency and loss
    live on very different numeric scales (milliseconds vs fractions of a
    percent) and reusing one sharpness value across both was found to
    under-saturate the score even well inside the deadline. Combined score
    is the PRODUCT of the two per-channel success probabilities scaled to
    [1,5] -- reflects that BOTH must be met for the application to actually
    function correctly, not an additive/average tradeoff.
    """
    latency_s = np.asarray(latency_s, dtype=np.float64)
    packet_loss = np.clip(np.asarray(packet_loss, dtype=np.float64), 0.0, 1.0)
    lat_sharpness = 4.0 / max(latency_transition_s, 1e-9)
    loss_sharpness = 4.0 / max(loss_transition, 1e-9)
    latency_ok = 1.0 - _sigmoid(lat_sharpness * (latency_s - deadline_s))
    loss_ok = 1.0 - _sigmoid(loss_sharpness * (packet_loss - loss_budget))
    success = latency_ok * loss_ok
    return np.clip(1.0 + 4.0 * success, 1.0, 5.0)


def acr_score_mmtc(
    packet_loss: "np.ndarray | float",
    latency_s: "np.ndarray | float",
    loss_tolerance: float = 0.05,
    loss_transition: float = 0.02,
    deadline_s: float = 1.0,
    latency_weight: float = 0.15,
) -> np.ndarray:
    """mMTC: loss-dominated. delivery_success uses a gentler sigmoid (loss
    tolerance is much wider than URLLC's, and its transition width is set
    from loss_transition the same way acr_score_urllc's channels are, so
    near-zero loss actually saturates near 1.0 rather than sitting at ~0.8).
    latency contributes only a small, linear discount (delay-tolerant
    applications barely notice it unless it's extreme) rather than another
    hard gate.
    """
    packet_loss = np.clip(np.asarray(packet_loss, dtype=np.float64), 0.0, 1.0)
    latency_s = np.asarray(latency_s, dtype=np.float64)
    loss_sharpness = 4.0 / max(loss_transition, 1e-9)
    delivery_success = 1.0 - _sigmoid(loss_sharpness * (packet_loss - loss_tolerance))
    latency_discount = np.clip(latency_s / max(deadline_s, 1e-6), 0.0, 1.0) * latency_weight
    score = delivery_success * (1.0 - latency_discount)
    return np.clip(1.0 + 4.0 * score, 1.0, 5.0)
