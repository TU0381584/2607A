#!/usr/bin/env python3
"""ClosedLoopKpmSource subclass for genuinely congested, dynamic, AND
randomized offline training/evaluation:

  - RANDOMIZED: each episode independently samples a fresh oversubscription
    multiplier per slice from a congested range (default [1.1, 2.2] x
    nominal_ratio/100, clipped to 0.98) -- episode-to-episode variability
    in how congested the cell is, not one fixed factor for the entire
    training run (train_offline.py's OVERSUBSCRIPTION_FACTOR=1.25 constant).
  - DYNAMIC: within an episode, ClosedLoopKpmSource's own mean-reverting
    random walk (offered_volatility) still drives continuous, non-fixed
    demand around that episode's sampled congestion level -- unchanged,
    reused as-is, not reimplemented.

Congestion range is deliberately independent per slice (embb/urllc/mmtc
each draw their own multiplier) so a training run doesn't only ever see
"all three slices congested together" -- realistic multi-slice
contention includes asymmetric congestion across slices.
"""
from typing import Dict, List

import numpy as np

from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource


class CongestedRandomKpmSource(ClosedLoopKpmSource):
    def __init__(
        self, *args,
        nominal_ratio: Dict[str, float],
        congestion_range: tuple = (1.1, 2.2),
        episode_rng: np.random.RandomState,
        **kwargs,
    ):
        self._nominal_ratio = dict(nominal_ratio)
        self._congestion_range = congestion_range
        self._episode_rng = episode_rng
        self._current_multiplier: Dict[str, float] = {s: 1.0 for s in nominal_ratio}
        init_ratio = self._sample_and_apply()
        super().__init__(*args, mean_offered_ratio=init_ratio, **kwargs)

    def _sample_and_apply(self) -> Dict[str, float]:
        lo, hi = self._congestion_range
        ratio = {}
        for slice_id, nominal in self._nominal_ratio.items():
            mult = float(self._episode_rng.uniform(lo, hi))
            self._current_multiplier[slice_id] = mult
            ratio[slice_id] = min(0.98, mult * nominal / 100.0)
        return ratio

    def new_episode_congestion(self) -> Dict[str, float]:
        """Call once per episode, BEFORE env.reset() (which calls poll()
        immediately) -- resamples each slice's congestion multiplier and
        applies it to self._mean_offered_ratio (already a plain instance
        attribute ClosedLoopKpmSource reads fresh every poll())."""
        ratio = self._sample_and_apply()
        self._mean_offered_ratio.update(ratio)
        return dict(self._current_multiplier)
