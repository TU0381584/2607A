#!/usr/bin/env python3
"""M1: a versioned, additive subclass of the frozen
`qoe_oran_framework.replay_kpm_source.ClosedLoopKpmSource`, NOT a
modification of it (qoe_oran_framework/ is frozen framework source and is
never edited by this project). Instantiating `RecalibratedClosedLoopKpmSource`
with every keyword left at its default reproduces the parent class's
original behaviour exactly (same defaults: drift_coef=0.1, ar1_coef=0.0,
offered_volatility=0.04, backlog_capacity=200.0) -- old behaviour stays
reproducible, per the M1 brief's explicit instruction.

Two new, independently-tunable parameters over the parent:

- `drift_coef`: the parent hardcodes 0.1 as the mean-reversion speed of the
  offered-demand random walk inside poll()'s body (not exposed as a
  constructor argument at all) -- exposed here so it becomes a fittable
  parameter.
- `ar1_coef`: NEW mechanism, absent from the parent entirely. The parent's
  noise term is i.i.d. Gaussian every step (no persistence/burstiness).
  docs/STAGE12_offline_online_gap.md root-caused the offline/live gap
  specifically to the offline demand process's temporal STRUCTURE (it
  matches the live mean but not live traffic's burstiness/autocorrelation),
  and docs/STAGE13_recalibration_attempt.md's own "unexplored next
  hypotheses" section names exactly this gap. `ar1_coef` (in [0, 1))
  turns the i.i.d. noise into an AR(1) process with the SAME marginal
  variance (offered_volatility**2 * B**2) but persistent, bursty
  autocorrelation when ar1_coef > 0 -- ar1_coef=0 is bit-identical to the
  parent's i.i.d. noise.

Everything else in poll() (backlog/serve/relief/churn/bler/UE-sample
construction) is copied verbatim from the parent, unchanged -- this is a
targeted override of the demand-generation step only, not a rewrite.
"""
from typing import Dict, List, Optional

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource, UeSample  # noqa: E402


class RecalibratedClosedLoopKpmSource(ClosedLoopKpmSource):
    def __init__(self, *args, drift_coef: float = 0.1, ar1_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._drift_coef = drift_coef
        self._ar1_coef = ar1_coef
        self._noise_state: Dict[tuple, float] = {key: 0.0 for key in self._offered}

    def poll(self) -> List[UeSample]:
        self._t += 1.0
        samples: List[UeSample] = []
        for gnb_id in self._gnb_ids:
            for slice_id in self._slice_ids:
                key = (gnb_id, slice_id)
                sd = self._sd_for_slice[slice_id]

                mean = self._mean_offered_ratio[slice_id] * self._B * self._gnb_load_multiplier[gnb_id]
                drift = self._drift_coef * (mean - self._offered[key])

                fresh = self._rng.normal(0.0, self._offered_volatility * self._B)
                if self._ar1_coef > 0.0:
                    prev = self._noise_state.get(key, 0.0)
                    # Same marginal std as the i.i.d. case (offered_volatility*B):
                    # var(state) = ar1^2*var(state) + (1-ar1^2)*var(fresh) => var(state)=var(fresh).
                    self._noise_state[key] = self._ar1_coef * prev + ((1.0 - self._ar1_coef ** 2) ** 0.5) * fresh
                    noise = self._noise_state[key]
                else:
                    noise = fresh

                self._offered[key] = max(0.0, self._offered[key] + drift + noise)
                offered = self._offered[key]

                relief = self._pending_relief.get(key, 0.0)
                if relief > 0.0:
                    self._backlog[key] = max(0.0, self._backlog[key] - relief)
                    self._pending_relief[key] = 0.0

                ceiling_prb = self._ceiling_ratio[key] / 100.0 * self._B
                demand = offered + self._backlog[key]
                served = min(demand, ceiling_prb)
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
