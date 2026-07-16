"""Unit conversions between Stage Zero's normalised, PRB-based KPM
representation and the real units (kbps, seconds) the calibration
references (P.1203, ACR scoring) need.

Stage Zero's environment/IQX-inference path works in normalised units
(avg_prbs_dl as a PRB count, dl_mac_buffer_occupation as raw scheduler
bytes, dl_bler as a fraction) because that's what's directly available from
E2SM-KPM and consistent between offline/live sources. The calibration
references need physically-meaningful units (a video ABR client reasons in
kbps; a control-loop deadline is stated in seconds) to produce a
DEFENSIBLE objective label. This module is the one place that conversion
happens, so it's never done inconsistently in two different places.

PRB_TO_KBPS is a DOCUMENTED APPROXIMATION, not a measured constant: actual
PRB-to-bitrate depends on MCS/numerology/channel conditions, which aren't
directly available per-step from this OAI build's E2SM-KPM. 100 kbps/PRB is
a representative order-of-magnitude figure for NR at a moderate MCS on a
106-PRB carrier (matches this rig's actual config) -- adequate for
generating calibration labels whose ordering (better conditions -> higher
MOS) is what actually matters for fitting IQX coefficients, not for
claiming an exact bitrate measurement.
"""
from __future__ import annotations

import numpy as np

UNIT_CONVERSION_LIMITATION = (
    "throughput (kbps) and latency (seconds) used for calibration labels "
    "are derived from Stage Zero's normalised PRB/scheduler-byte KPM "
    "representation via a documented approximate conversion "
    "(PRB_TO_KBPS=100, Little's-Law-style backlog/throughput latency "
    "estimate), not measured bitrate/delay -- see calibration/units.py "
    "module docstring."
)

PRB_TO_KBPS = 100.0


def prb_to_kbps(avg_prbs: "np.ndarray | float") -> np.ndarray:
    return np.asarray(avg_prbs, dtype=np.float64) * PRB_TO_KBPS


def backlog_bytes_to_latency_s(
    backlog_bytes: "np.ndarray | float", throughput_kbps: "np.ndarray | float",
) -> np.ndarray:
    """Little's-Law-style queueing delay estimate: time to drain the
    currently-buffered backlog at the current service rate."""
    backlog_bytes = np.asarray(backlog_bytes, dtype=np.float64)
    throughput_bps = np.maximum(np.asarray(throughput_kbps, dtype=np.float64) * 1000.0, 1e-6)
    return (backlog_bytes * 8.0) / throughput_bps
