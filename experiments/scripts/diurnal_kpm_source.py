#!/usr/bin/env python3
"""SharedPoolCongestedKpmSource subclass that replaces per-episode UNIFORM
RANDOM congestion (CongestedRandomKpmSource._sample_and_apply) with a
DETERMINISTIC diurnal/weekly schedule -- a condensed "one simulated week"
of peak/off-peak usage compressed into N training episodes (default 700,
i.e. 100 episodes/simulated-day), so a training curve over those episodes
shows whether the policy improves at handling recurring peak-hour
contention specifically, not just contention in general.

Schedule model (deliberately simple, not a traffic-engineering claim):
  - time-of-day: bimodal (morning + evening rush-hour) raised-cosine
    humps, matching the textbook double-peak diurnal shape of real
    cellular traffic, NOT a single midday peak.
  - day-of-week: weekday (Mon-Fri) peaks at full amplitude; weekend
    (Sat/Sun) peaks at WEEKEND_SCALE of that (lighter, flatter weekend
    profile).
  - small residual per-episode Gaussian noise on top of the deterministic
    curve (NOISE_STD) so the policy can't just memorize a noise-free
    periodic signal -- keeps some of CongestedRandomKpmSource's original
    intent (genuine variability) while making the DOMINANT signal the
    periodic diurnal/weekly pattern the learning-curve experiment is
    actually about.

The multiplier range [MULT_LO, MULT_HI] is calibrated the same way
CongestedRandomKpmSource's congestion_range is: relative to
nominal_ratio/100, clipped to 0.98. Peak-of-week hits MULT_HI; trough
hits MULT_LO.
"""
import math
from typing import Dict

import numpy as np

from shared_pool_kpm_source import SharedPoolCongestedKpmSource

WEEKEND_SCALE = 0.55
NOISE_STD = 0.04
MORNING_PEAK_HOUR = 8.5 / 24.0   # ~08:30
EVENING_PEAK_HOUR = 19.5 / 24.0  # ~19:30
PEAK_WIDTH = 0.09                # fraction-of-day std of each rush-hour hump


def _rush_hour_shape(time_of_day: float) -> float:
    """Bimodal (morning+evening) shape on [0,1], peak value 1.0."""
    def hump(center: float) -> float:
        d = min(abs(time_of_day - center), 1.0 - abs(time_of_day - center))  # wrap around midnight
        return math.exp(-0.5 * (d / PEAK_WIDTH) ** 2)
    return max(hump(MORNING_PEAK_HOUR), hump(EVENING_PEAK_HOUR))


def diurnal_value(episode_idx: int, episodes_per_week: int) -> Dict[str, float]:
    """Returns {"time_of_day", "day_of_week", "is_weekend", "shape"} for
    episode_idx under a week condensed into episodes_per_week episodes."""
    t = (episode_idx % episodes_per_week) / float(episodes_per_week)  # [0,1) across the week
    day_of_week = int(t * 7) % 7          # 0=Mon ... 6=Sun
    time_of_day = (t * 7) % 1.0           # fraction through that day
    is_weekend = day_of_week >= 5
    shape = _rush_hour_shape(time_of_day)
    if is_weekend:
        shape *= WEEKEND_SCALE
    return {
        "time_of_day": time_of_day, "day_of_week": day_of_week,
        "is_weekend": is_weekend, "shape": shape,
    }


class DiurnalCongestedKpmSource(SharedPoolCongestedKpmSource):
    def __init__(
        self, *args,
        episodes_per_week: int = 700,
        mult_range: tuple = (0.5, 1.8),
        noise_rng: "np.random.RandomState | None" = None,
        **kwargs,
    ):
        self._episodes_per_week = episodes_per_week
        self._mult_lo, self._mult_hi = mult_range
        self._noise_rng = noise_rng if noise_rng is not None else np.random.RandomState(0)
        self._episode_idx = -1  # incremented to 0 on first new_episode_congestion() call
        self.last_diurnal_info: Dict = {}
        super().__init__(*args, **kwargs)

    def _sample_and_apply(self) -> Dict[str, float]:
        self._episode_idx += 1
        info = diurnal_value(self._episode_idx, self._episodes_per_week)
        self.last_diurnal_info = {**info, "episode_idx": self._episode_idx}

        ratio = {}
        for slice_id, nominal in self._nominal_ratio.items():
            noise = float(self._noise_rng.normal(0.0, NOISE_STD))
            shape = min(1.0, max(0.0, info["shape"] + noise))
            mult = self._mult_lo + shape * (self._mult_hi - self._mult_lo)
            self._current_multiplier[slice_id] = mult
            ratio[slice_id] = min(0.98, mult * nominal / 100.0)
        return ratio
