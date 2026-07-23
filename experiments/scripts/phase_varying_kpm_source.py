#!/usr/bin/env python3
"""A ClosedLoopKpmSource subclass whose mean_offered_ratio changes at fixed
step boundaries within an episode, instead of staying constant for the
whole run -- a small, OFFLINE-ONLY preliminary probe of whether the
frozen checkpoints trained under constant demand (saclb_offline_campaign.yaml)
generalize to demand that varies within an episode. This does NOT touch
qoe_oran_framework/ source; it only subclasses the existing closed-loop
source from experiments/ code, per-slice mean_offered_ratio being a plain
instance attribute ClosedLoopKpmSource already reads fresh every poll().

Three phases per 60-step episode (matching this project's existing
episode length), each phase scaling every slice's OWN already-configured
mean_offered_ratio by a multiplier -- NOT replacing it with new absolute
Mbps/packet-size numbers (this offline synthetic source has no such
concept; see saclb_offline_campaign.yaml's own oversubscription-by-ratio
model). This keeps each slice's relative headroom/scarcity comparable to
its existing offline calibration while still varying demand over time:

  phase 1 (steps  1-20): 0.7x configured mean_offered_ratio ("low")
  phase 2 (steps 21-40): 1.3x configured mean_offered_ratio ("high")
  phase 3 (steps 41-60): 1.0x configured mean_offered_ratio ("medium", baseline level)

Identical across every arm/seed (only the multiplier schedule below
governs demand; policy/env seeding is untouched) -- the same
identical-across-arms requirement Section V's actual live campaign
already follows.
"""
from typing import Dict, List, Tuple

from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource
from qoe_oran_framework.types import UeSample

PHASE_BOUNDARIES = [(1, 20, "low", 0.7), (21, 40, "high", 1.3), (41, 60, "medium", 1.0)]


def phase_for_step(step_in_episode: int) -> Tuple[str, float]:
    for lo, hi, name, mult in PHASE_BOUNDARIES:
        if lo <= step_in_episode <= hi:
            return name, mult
    return PHASE_BOUNDARIES[-1][2], PHASE_BOUNDARIES[-1][3]


class PhaseVaryingClosedLoopKpmSource(ClosedLoopKpmSource):
    def __init__(self, *args, base_mean_offered_ratio: Dict[str, float], **kwargs):
        super().__init__(*args, mean_offered_ratio=dict(base_mean_offered_ratio), **kwargs)
        self._base_mean_offered_ratio = dict(base_mean_offered_ratio)
        self._steps_in_episode = 0
        self.phase_log: List[Tuple[int, str, Dict[str, float]]] = []

    def reset_episode_clock(self) -> None:
        self._steps_in_episode = 0

    def poll(self) -> List[UeSample]:
        self._steps_in_episode += 1
        phase_name, mult = phase_for_step(self._steps_in_episode)
        for slice_id, base in self._base_mean_offered_ratio.items():
            self._mean_offered_ratio[slice_id] = base * mult
        self.phase_log.append((self._steps_in_episode, phase_name, dict(self._mean_offered_ratio)))
        return super().poll()
