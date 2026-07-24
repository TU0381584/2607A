#!/usr/bin/env python3
"""CongestedRandomKpmSource subclass that adds a REAL shared-PRB-pool
constraint across slices at each gNB -- the missing ingredient for
testing "does DRL preserve URLLC over eMBB/mMTC under contention".

ClosedLoopKpmSource (the base class, and CongestedRandomKpmSource which
extends it) computes each slice's served PRBs independently:
served_i = min(demand_i, ceiling_prb_i). There is no constraint that
sum_i(served_i) <= B -- so raising eMBB's ceiling never actually takes
capacity away from URLLC in that model, which is why the first
congested-scenario experiment (2026-07-24) showed real DRL-vs-baseline
gains but no URLLC-specific preservation: that mechanism doesn't exist
without this.

Physical model added here, once per poll() per gNB: compute each slice's
REQUESTED PRBs (min(demand_i, ceiling_prb_i), the same quantity
ClosedLoopKpmSource already calls "served"), and if the sum across the
3 slices at that gNB exceeds B, scale every slice's ACTUAL served amount
down proportionally to its own request share (a standard proportional/
max-min-fair degradation under scarcity -- deliberately NOT hardcoding
URLLC-first priority into the physics itself, since the point is to test
whether the POLICY's own (already URLLC-weighted, priority_weight=5.0/
violation_penalty=8.0) reward-driven ceiling choices are what protect
URLLC, not the simulator silently guaranteeing it regardless of policy).
Unmet (rationed-away) demand carries forward as backlog exactly like the
base class's own unmet-demand handling.
"""
from typing import Dict, List

import numpy as np

from congested_random_kpm_source import CongestedRandomKpmSource
from qoe_oran_framework.types import UeSample


class SharedPoolCongestedKpmSource(CongestedRandomKpmSource):
    """shared_pool_prb: the REAL contended budget these 3 slices compete
    over (default 15) -- deliberately NOT self._B (100), which is only the
    ratio-normalization scale each slice's ceiling_ratio is a percentage
    of. At this config's cap values (embb=12, urllc=4, mmtc=3, summing to
    19), maxing out every slice's ceiling simultaneously requests 19 PRB
    against a 15-PRB shared pool -- genuine, non-trivial scarcity if all
    three try to claim their full ceiling at once, matching the live rig's
    own calibrated scale (real measured demand there: embb ~15 PRB,
    urllc/mmtc ~5 PRB each, caps 12/4/3 against a much bigger B=100 cell
    that has other real overhead these 3 slices don't get to use)."""

    def __init__(self, *args, shared_pool_prb: float = 15.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_pool_prb = shared_pool_prb
        # Read by the training loop AFTER each poll() to build a
        # contention-aware reward shaping term -- see
        # train_offline_congested.py's contention_bonus(). last_contention_ratio
        # in [0,1]: 0 = pool nowhere near saturated, 1 = fully saturated
        # (rationing active). Deliberately exposed here rather than folded
        # into reward.py itself -- keeps the shared-PRB physics and the
        # reward-shaping experiment on top of it as two separable,
        # independently-inspectable pieces.
        self.last_contention_ratio: float = 0.0

    def poll(self) -> List[UeSample]:
        self._t += 1.0
        samples: List[UeSample] = []
        for gnb_id in self._gnb_ids:
            requested: Dict[str, float] = {}
            demand_by_slice: Dict[str, float] = {}
            for slice_id in self._slice_ids:
                key = (gnb_id, slice_id)
                mean = self._mean_offered_ratio[slice_id] * self._B * self._gnb_load_multiplier[gnb_id]
                drift = 0.1 * (mean - self._offered[key])
                noise = self._rng.normal(0.0, self._offered_volatility * self._B)
                self._offered[key] = max(0.0, self._offered[key] + drift + noise)
                offered = self._offered[key]

                relief = self._pending_relief.get(key, 0.0)
                if relief > 0.0:
                    self._backlog[key] = max(0.0, self._backlog[key] - relief)
                    self._pending_relief[key] = 0.0

                ceiling_prb = self._ceiling_ratio[key] / 100.0 * self._B
                demand = offered + self._backlog[key]
                demand_by_slice[slice_id] = demand
                requested[slice_id] = min(demand, ceiling_prb)

            # Shared-pool constraint: this gNB has only B PRBs total,
            # shared across all 3 slices' requested (ceiling-limited)
            # demand -- if the sum exceeds B, ration proportionally.
            total_requested = sum(requested.values())
            if total_requested > self._shared_pool_prb and total_requested > 0:
                scale = self._shared_pool_prb / total_requested
            else:
                scale = 1.0
            self.last_contention_ratio = min(1.0, total_requested / max(self._shared_pool_prb, 1e-6))

            for slice_id in self._slice_ids:
                key = (gnb_id, slice_id)
                served = requested[slice_id] * scale
                unmet = max(0.0, demand_by_slice[slice_id] - served)
                self._backlog[key] = min(self._backlog_capacity, unmet)

                if self._rng.rand() < self._churn_prob and self._ues[key]:
                    idx = self._rng.randint(0, len(self._ues[key]))
                    self._ues[key][idx] = self._next_rnti()

                n_ues = max(1, len(self._ues[key]))
                per_ue_served = served / n_ues
                backlog_frac = self._backlog[key] / max(self._backlog_capacity, 1e-6)
                bler = max(0.0, min(1.0, 0.02 + 0.3 * backlog_frac))
                sd = self._sd_for_slice[slice_id]

                for rnti in self._ues[key]:
                    samples.append(
                        UeSample(
                            rnti=rnti, timestamp_s=self._t, nssai_sst=1, nssai_sd=sd,
                            avg_prbs_dl=per_ue_served, gnb_id=gnb_id,
                            dl_total_bytes=per_ue_served * 1000.0,
                            dl_errors=bler * 2.0, dl_bler=bler,
                            dl_mac_buffer_occupation=self._backlog[key] / n_ues,
                        )
                    )
        return samples
