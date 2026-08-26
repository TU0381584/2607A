#!/usr/bin/env python3
"""RealisticServedKpmSource: an offline KpmSource whose served-PRB model
is grounded in real, measured rig behavior instead of the frozen
ClosedLoopKpmSource's assumption that a slice's admission ceiling
(max_ratio) directly bounds how many PRBs it gets served.

WHY THIS EXISTS. M32-M33 (docs/PAPER5_M32... commit 78e11a8) found that
retraining under two different offline "heavier demand" levers
(arrivals/step, then the real demand-calibrated mean_offered_ratio)
never changed a live-collapsing checkpoint's behavior at all, because
ClosedLoopKpmSource's own congestion_level stays pinned near 0.03-0.09
regardless of either lever -- while live measurement (state-vector
capture, m33) showed real congestion_level at 0.23-0.26 (3 UEs) and
0.60-0.69 (6 UEs), far outside anything the checkpoint was ever trained
on. M34's live ratio-to-PRB sweep (experiments/scripts/
m34_ratio_to_prb_sweep.py, results at experiments/results/live/
m34_ratio_sweep/*_3ue.jsonl) found the deeper reason: served PRB on
this real rig does NOT depend on the admission ceiling at all, anywhere
in the range a trained policy can actually issue (ratio 1-4) --

    slice   ratio=1  ratio=2  ratio=3  ratio=4
    urllc      5.00     5.00     5.00     5.00
    embb      13.00    13.00    13.00    13.00
    mmtc       5.00     5.00     5.00     5.00

-- confirmed independently by M33's live state-vector capture showing
the same order of magnitude (urllc/mmtc ~0.05, embb ~0.13-0.16 at 3
UEs; ~0.10/~0.10/~0.40-0.49 at 6 UEs, all as prb_used_ratio = PRB/B).
So on this hardware the ceiling-ratio lever a policy adjusts every step
is nearly inert; only the accept/reject decision itself matters
(via the already-established notify_rejected() backlog-relief
mechanism, reused here unchanged).

This class keeps ClosedLoopKpmSource's exact backlog/offered/relief
machinery (frozen file, not touched -- this is a new, separate class,
not a subclass, since the one-line change needed is in the middle of
poll() and Python gives no override point for that) but replaces
"served = min(demand, ceiling_prb)" with "served = an empirically
measured per-slice constant" -- ratio-independent, matching what the
real rig actually does. To expose training to the FULL realistic
congestion range (not just one fixed point, which would just move the
OOD boundary rather than removing it), each slice's served-PRB target
interpolates between the measured 3-UE-like and 6-UE-like anchors via
a slowly mean-reverting random walk in [0, 1], recomputed every poll()
-- so across a training run the policy sees congestion drift across
the whole 0.23-0.69 range this rig's real traffic actually produces,
not a single snapshot of it.

Usage (as a library, mirroring live_scale_offline_env.py's pattern):
    from realistic_served_kpm_source import RealisticServedKpmSource, SERVED_PRB_3UE, SERVED_PRB_6UE
    kpm = RealisticServedKpmSource(seed=900, gnb_ids=cfg.gnb_ids,
                                    slice_ids=list(cfg.slice_by_id), B=cfg.B,
                                    sd_for_slice=sd_for_slice)
"""
from typing import Any, Dict, List, Optional

import numpy as np

from qoe_oran_framework.types import UeSample

# Measured 2026-08-26, this rig, real traffic, ratio swept 1-4 (see module
# docstring). n=20 polls/ratio, flat across all 4 ratios in every slice.
SERVED_PRB_3UE: Dict[str, float] = {"urllc": 5.0, "embb": 13.0, "mmtc": 5.0}
# Measured 2026-08-25 via live state-vector capture (m33), 2 UEs/slice,
# prb_used_ratio x B=100: urllc/mmtc ~0.10, embb ~0.40-0.49 -> midpoint 45.
SERVED_PRB_6UE: Dict[str, float] = {"urllc": 10.0, "embb": 45.0, "mmtc": 10.0}

_SD_FOR_SLICE = {"embb": 0, "urllc": 1, "mmtc": 2}


class RealisticServedKpmSource:
    """See module docstring for why this exists and what it measures."""

    def __init__(
        self,
        seed: int,
        gnb_ids: List[str],
        slice_ids: List[str],
        B: float = 100.0,
        served_prb_lo: Optional[Dict[str, float]] = None,
        served_prb_hi: Optional[Dict[str, float]] = None,
        oversubscription_factor: float = 1.3,
        served_prb_noise_std: float = 0.5,
        load_alpha_volatility: float = 0.03,
        gnb_load_multiplier: Optional[Dict[str, float]] = None,
        offered_volatility: float = 0.04,
        ues_per_slice: int = 4,
        backlog_capacity: float = 2000.0,
        churn_prob: float = 0.05,
        initial_ceiling_ratio: float = 100.0,
        sd_for_slice: Optional[Dict[str, int]] = None,
    ):
        self._rng = np.random.RandomState(seed)
        self._gnb_ids = gnb_ids
        self._slice_ids = slice_ids
        self._B = B
        self._served_prb_lo = served_prb_lo or dict(SERVED_PRB_3UE)
        self._served_prb_hi = served_prb_hi or dict(SERVED_PRB_6UE)
        # Offered (organic) demand tracks the SAME alpha interpolation as
        # served, scaled up by oversubscription_factor -- more UEs means
        # more aggregate demand, not just more serving capacity (fixed
        # bug: v1 of this class left offered fixed while served grew with
        # alpha, so high-alpha steps had LESS scarcity, not more). Kept
        # above served at both ends of the range so backlog genuinely
        # accumulates in both the 3-UE-like and 6-UE-like regimes,
        # matching live's own observed behavior (M8: backlog saturates
        # within minutes at 3 UEs too, not just 6).
        self._offered_prb_lo = {s: v * oversubscription_factor for s, v in self._served_prb_lo.items()}
        self._offered_prb_hi = {s: v * oversubscription_factor for s, v in self._served_prb_hi.items()}
        self._served_prb_noise_std = served_prb_noise_std
        self._load_alpha_volatility = load_alpha_volatility
        self._gnb_load_multiplier = gnb_load_multiplier or self._default_gnb_load_multiplier(gnb_ids, seed)
        self._offered_volatility = offered_volatility
        self._ues_per_slice = ues_per_slice
        self._backlog_capacity = backlog_capacity
        self._churn_prob = churn_prob
        self._sd_for_slice = dict(sd_for_slice) if sd_for_slice else dict(_SD_FOR_SLICE)
        self._sd_for_slice_reverse = {v: k for k, v in self._sd_for_slice.items()}

        self._offered: Dict[tuple, float] = {}
        self._backlog: Dict[tuple, float] = {}
        self._ceiling_ratio: Dict[tuple, float] = {}
        self._ues: Dict[tuple, List[int]] = {}
        self._pending_relief: Dict[tuple, float] = {}
        self._load_alpha: Dict[str, float] = {g: self._rng.uniform(0.0, 1.0) for g in gnb_ids}
        self._rnti_counter = 0
        self._t = 0.0
        self.sent_controls: List[Dict[str, Any]] = []

        for gnb_id in gnb_ids:
            for slice_id in slice_ids:
                key = (gnb_id, slice_id)
                self._offered[key] = self._offered_prb_lo[slice_id] * self._gnb_load_multiplier[gnb_id]
                self._backlog[key] = 0.0
                self._ceiling_ratio[key] = initial_ceiling_ratio
                self._ues[key] = [self._next_rnti() for _ in range(ues_per_slice)]
                self._pending_relief[key] = 0.0

    @staticmethod
    def _default_gnb_load_multiplier(gnb_ids: List[str], seed: int = 0) -> Dict[str, float]:
        rng = np.random.RandomState(seed)
        multipliers: Dict[str, float] = {}
        for i, gnb_id in enumerate(gnb_ids):
            multipliers[gnb_id] = 1.0 if i == 0 else float(rng.uniform(0.6, 1.4))
        return multipliers

    def _next_rnti(self) -> int:
        self._rnti_counter += 1
        return self._rnti_counter

    def poll(self) -> List[UeSample]:
        self._t += 1.0
        samples: List[UeSample] = []
        for gnb_id in self._gnb_ids:
            # Slowly mean-reverting random walk in [0,1]: 0 = 3-UE-like
            # measured congestion, 1 = 6-UE-like measured congestion.
            # Recomputed once per gNB per poll (shared across its slices,
            # since real load conditions move all slices together).
            alpha = self._load_alpha[gnb_id]
            drift = 0.05 * (0.5 - alpha)
            noise = self._rng.normal(0.0, self._load_alpha_volatility)
            alpha = float(np.clip(alpha + drift + noise, 0.0, 1.0))
            self._load_alpha[gnb_id] = alpha

            for slice_id in self._slice_ids:
                key = (gnb_id, slice_id)
                sd = self._sd_for_slice[slice_id]

                lo_o = self._offered_prb_lo[slice_id]
                hi_o = self._offered_prb_hi[slice_id]
                target_offered = ((1.0 - alpha) * lo_o + alpha * hi_o) * self._gnb_load_multiplier[gnb_id]
                drift_o = 0.1 * (target_offered - self._offered[key])
                noise_o = self._rng.normal(0.0, self._offered_volatility * target_offered)
                self._offered[key] = max(0.0, self._offered[key] + drift_o + noise_o)
                offered = self._offered[key]

                relief = self._pending_relief.get(key, 0.0)
                if relief > 0.0:
                    self._backlog[key] = max(0.0, self._backlog[key] - relief)
                    self._pending_relief[key] = 0.0

                # THE substantive change vs. ClosedLoopKpmSource: served
                # is an empirically measured constant (interpolated across
                # the real 3-UE/6-UE anchors), NOT min(demand, ceiling_prb)
                # -- matching the real rig's own ratio-independent behavior
                # (m34 sweep, module docstring).
                lo = self._served_prb_lo[slice_id]
                hi = self._served_prb_hi[slice_id]
                target_served = (1.0 - alpha) * lo + alpha * hi
                served = max(0.0, target_served + self._rng.normal(0.0, self._served_prb_noise_std))

                demand = offered + self._backlog[key]
                unmet = max(0.0, demand - served)
                self._backlog[key] = min(self._backlog_capacity, unmet)

                if self._rng.rand() < self._churn_prob and self._ues[key]:
                    idx = self._rng.randint(0, len(self._ues[key]))
                    self._ues[key][idx] = self._next_rnti()

                n_ues = max(1, len(self._ues[key]))
                per_ue_served = served / n_ues
                backlog_frac = self._backlog[key] / max(self._backlog_capacity, 1e-6)
                bler = max(0.0, min(1.0, 0.02 + 0.3 * backlog_frac))

                for rnti in self._ues[key]:
                    samples.append(
                        UeSample(
                            rnti=rnti,
                            timestamp_s=self._t,
                            nssai_sst=1,
                            nssai_sd=sd,
                            avg_prbs_dl=per_ue_served,
                            gnb_id=gnb_id,
                            dl_total_bytes=per_ue_served * 1000.0,
                            dl_errors=bler * 2.0,
                            dl_bler=bler,
                            dl_mac_buffer_occupation=self._backlog[key] / n_ues,
                        )
                    )
        return samples

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        # Recorded (kept for logging/reward-diagnostic parity with
        # ClosedLoopKpmSource) but NOT used to compute served -- see
        # module docstring: ratio has no measured effect on served PRB
        # on this rig, in the range a trained policy can issue.
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )
        slice_id = self._sd_for_slice_reverse.get(int(sd))
        key = (gnb_id, slice_id)
        if key in self._ceiling_ratio:
            self._ceiling_ratio[key] = float(max_ratio)

    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None:
        """Unchanged from ClosedLoopKpmSource: this is the one lever that
        DOES matter on the real rig (accept/reject, not ceiling)."""
        key = (gnb_id, slice_id)
        if key not in self._offered or n_rejected <= 0:
            return
        n_ues = max(1, len(self._ues[key]))
        per_request_demand = self._offered[key] / n_ues
        self._pending_relief[key] = self._pending_relief.get(key, 0.0) + n_rejected * per_request_demand

    def close(self) -> None:
        pass
