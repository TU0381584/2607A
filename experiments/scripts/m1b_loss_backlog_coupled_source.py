#!/usr/bin/env python3
"""M1 Block B: one bounded, time-boxed experiment testing the structural
hypothesis named in docs/PAPER5_M1_recalibration.md's conclusion -- that the
offline env's loss/bler channel is a PURE, zero-variance deterministic
function of backlog fraction (`bler = 0.02 + 0.3*backlog_frac`, the frozen
ClosedLoopKpmSource.poll()), while real live loss presumably has its own
independent noise on top of whatever backlog coupling exists.

Data-availability note, stated plainly rather than silently worked around:
live omega_log.jsonl does NOT log raw backlog occupancy or raw loss/bler
values -- kpm_adapter.py computes `raw_queue_len_norm` and `loss_proxy`
internally (reward.py's check_violations then folds them into a single
`margin = min(queue_margin, loss_margin)`), but only that FINAL combined
per_slice_sla_margin survives to the log (confirmed by direct inspection,
no other live log exists beyond omega_log.jsonl/PROGRESS.log). A literal
raw joint (loss, backlog) scatter cannot be reconstructed from any
artifact this project has. This experiment therefore fits the same
available target as M1's original attempt (the live pooled per-slice
margin distribution, mean+std) -- the only live signal that actually
reflects the loss channel's contribution -- rather than inventing a raw
joint-distribution number that isn't observable in any log.

Additive on top of RecalibratedClosedLoopKpmSource (itself additive over
the frozen ClosedLoopKpmSource): adds `loss_noise_std` (independent
Gaussian noise added to bler, still backlog-coupled since it's added to
the existing backlog-driven formula, not replacing it) and
`loss_noise_ar1` (optional persistence, same AR(1) parameterization as the
demand noise). loss_noise_std=0 reproduces the parent's bit-identical
deterministic bler exactly.
"""
import sys
from typing import Dict, List

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from m1_recalibrated_kpm_source import RecalibratedClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.replay_kpm_source import UeSample  # noqa: E402


class LossBacklogCoupledKpmSource(RecalibratedClosedLoopKpmSource):
    def __init__(self, *args, loss_noise_std: float = 0.0, loss_noise_ar1: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_noise_std = loss_noise_std
        self._loss_noise_ar1 = loss_noise_ar1
        self._loss_noise_state: Dict[tuple, float] = {key: 0.0 for key in self._offered}

    def poll(self) -> List[UeSample]:
        # Reuse the parent's full demand/backlog/serve computation unchanged,
        # then only override the loss (bler) channel below -- everything
        # upstream of bler is byte-for-byte the same as
        # RecalibratedClosedLoopKpmSource / the frozen parent.
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

                # --- the only change relative to the parent: bler gets an
                # independent (optionally AR(1)-persistent) noise term added
                # on top of the existing backlog-driven mean, instead of
                # being a pure zero-variance function of backlog_frac. ---
                loss_fresh = self._rng.normal(0.0, self._loss_noise_std) if self._loss_noise_std > 0 else 0.0
                if self._loss_noise_ar1 > 0.0 and self._loss_noise_std > 0.0:
                    prev_l = self._loss_noise_state.get(key, 0.0)
                    self._loss_noise_state[key] = (
                        self._loss_noise_ar1 * prev_l
                        + ((1.0 - self._loss_noise_ar1 ** 2) ** 0.5) * loss_fresh
                    )
                    loss_noise = self._loss_noise_state[key]
                else:
                    loss_noise = loss_fresh
                bler = max(0.0, min(1.0, 0.02 + 0.3 * backlog_frac + loss_noise))

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
