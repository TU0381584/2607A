"""Network KPI -> simulated DASH/ABR video-client behaviour -> P.1203
mode-0 input, for generating objective eMBB MOS calibration labels.

P.1203 mode 0 needs video-BITSTREAM-level inputs (per-segment bitrate,
resolution, fps) plus stalling events -- not raw network KPIs. This module
is the documented bridge: it simulates a standard throughput-based ABR
client's segment-selection and playout-buffer behaviour, driven by our
measured/simulated per-slice throughput, then hands the result to
itu_p1203 (the official ITU-T P.1203 reference implementation,
itu-p1203/itu-p1203 on GitHub, installed per this stage's explicit
approval) to get an objective MOS score (O46).

THIS IS A MODELLING ASSUMPTION, NOT A MEASUREMENT. The "objective" MOS
label it produces is only as good as the ABR/buffer model below (bitrate
ladder, throughput-based bitrate selection rule, segment duration). It is
a standard, documented simplification (comparable to virtual-player models
used in ABR research, e.g. Pensieve-style buffer simulation), not a novel
ABR contribution -- sufficient for generating monotonic, defensible
calibration labels (worse network conditions -> lower resulting MOS), not
a claim of matching any specific real video service's exact behaviour.
This limitation is carried into every Omega-tuple that uses these labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from itu_p1203 import P1203Standalone

VIDEO_CLIENT_MODEL_LIMITATION = (
    "eMBB calibration MOS labels come from a simulated throughput-based "
    "ABR video client (documented bitrate ladder + simplified playout-buffer "
    "model) feeding the official itu_p1203 P.1203 mode-0 reference "
    "implementation -- not a measurement of any real video session. See "
    "calibration/video_client_model.py module docstring for the full "
    "modelling-assumption rationale."
)

# DASH-IF-style bitrate ladder (kbps) and matching resolution rungs --
# a standard, widely-used reference ladder (comparable to the one used in
# the P.1203 project's own test datasets), not tuned to any specific codec
# or service.
BITRATE_LADDER_KBPS = [235, 375, 560, 750, 1050, 1750, 2350, 3000, 4300, 5800]
RESOLUTION_LADDER = [
    "320x240", "384x288", "512x384", "640x480", "768x576",
    "1024x768", "1280x720", "1280x720", "1920x1080", "1920x1080",
]
SEGMENT_DURATION_S = 4.0
FPS = 25.0
CODEC = "h264"
PLAYOUT_BUFFER_CAP_S = 30.0


@dataclass
class AbrSession:
    segments: List[dict]
    stalling: List[Tuple[float, float]]  # (media-time start, duration)
    mean_bitrate_kbps: float
    stall_fraction: float  # total stall time / total session duration


def _select_bitrate(throughput_kbps: float) -> int:
    """Throughput-based bitrate selection: pick the highest ladder rung the
    client believes it can sustain. A documented simplification of real ABR
    heuristics (which also use buffer occupancy, e.g. BOLA/BB) -- sufficient
    for calibration-label purposes, not a research contribution in ABR
    algorithm design."""
    idx = 0
    for i, br in enumerate(BITRATE_LADDER_KBPS):
        if br <= throughput_kbps:
            idx = i
    return idx


def simulate_abr_session(
    throughput_kbps_trace: List[float], segment_duration_s: float = SEGMENT_DURATION_S,
) -> AbrSession:
    """throughput_kbps_trace: one average-throughput sample per segment
    download window. Simplified virtual-player buffer model: buffer drains
    by (download_time - segment_duration) each segment; a stall occurs
    whenever download_time exceeds the buffer available at that point."""
    segments: List[dict] = []
    stalling: List[Tuple[float, float]] = []
    buffer_s = 0.0
    media_t = 0.0
    bitrates_selected: List[float] = []

    for throughput_kbps in throughput_kbps_trace:
        throughput_kbps = max(throughput_kbps, 1.0)
        idx = _select_bitrate(throughput_kbps)
        bitrate_kbps = BITRATE_LADDER_KBPS[idx]
        resolution = RESOLUTION_LADDER[idx]
        bitrates_selected.append(bitrate_kbps)

        segment_bits = bitrate_kbps * segment_duration_s * 1000.0
        download_time_s = segment_bits / (throughput_kbps * 1000.0)

        if download_time_s > buffer_s:
            stall_duration = download_time_s - buffer_s
            stalling.append((media_t + buffer_s, stall_duration))
            buffer_s = 0.0
        else:
            buffer_s -= download_time_s

        segments.append({
            "bitrate": bitrate_kbps, "codec": CODEC, "duration": segment_duration_s,
            "fps": FPS, "resolution": resolution, "start": media_t,
        })
        buffer_s = min(buffer_s + segment_duration_s, PLAYOUT_BUFFER_CAP_S)
        media_t += segment_duration_s

    total_stall = sum(d for _, d in stalling)
    total_duration = media_t + total_stall
    return AbrSession(
        segments=segments, stalling=stalling,
        mean_bitrate_kbps=(sum(bitrates_selected) / len(bitrates_selected)) if bitrates_selected else 0.0,
        stall_fraction=(total_stall / total_duration) if total_duration > 0 else 0.0,
    )


def p1203_mos_from_throughput_trace(
    throughput_kbps_trace: List[float], stream_id: int = 42,
) -> Tuple[float, AbrSession]:
    """End-to-end: throughput trace -> simulated ABR session -> P.1203
    mode-0 O46 (overall quality score, 1-5 MOS scale). Returns (mos,
    session) so callers can also inspect stall_fraction / mean_bitrate for
    diagnostic/reporting purposes."""
    session = simulate_abr_session(throughput_kbps_trace)
    # Constant, robust-codec audio track (AAC-LC, 128kbps) alongside the
    # throughput-driven video track: audio is typically low-bitrate and far
    # less sensitive to the PRB-scale throughput variations this bridge is
    # built to reflect, so holding it constant isolates the video-quality
    # signal our IQX calibration actually cares about, without needing a
    # second, unmodelled audio-degradation pathway.
    audio_segments = [
        {"bitrate": 128.0, "codec": "aaclc", "duration": seg["duration"], "start": seg["start"]}
        for seg in session.segments
    ]
    report = {
        "I11": {"streamId": stream_id, "segments": audio_segments},
        "I13": {"streamId": stream_id, "segments": session.segments},
        "I23": {"streamId": stream_id, "stalling": [list(s) for s in session.stalling]},
        "IGen": {"device": "pc", "displaySize": "1920x1080"},
    }
    result = P1203Standalone(report, quiet=True).calculate_complete()
    mos = float(result["O46"])
    return mos, session
