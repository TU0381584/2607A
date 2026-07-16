"""Offline KpmSource implementations: a JSONL trace player and a synthetic
generator. Both implement the same protocol `live_kpm_source.LiveKpmSource`
will implement, so RANEnv is fully testable with zero network/testbed
dependency (Stage Zero build-order step 4's "critical gate").
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .types import UeSample


class KpmSource(Protocol):
    def poll(self) -> List[UeSample]: ...
    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None: ...
    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None: ...
    def close(self) -> None: ...


class ReplayKpmSource:
    """Plays back a JSONL trace of UE-sample snapshots, one row per poll()."""

    def __init__(self, trace_path: str, loop: bool = True):
        self._rows = self._load_trace(trace_path)
        if not self._rows:
            raise ValueError(f"trace file {trace_path} contains no rows")
        self._idx = 0
        self._loop = loop
        self.sent_controls: List[Dict[str, Any]] = []

    @staticmethod
    def _load_trace(trace_path: str) -> List[dict]:
        rows = []
        with Path(trace_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def poll(self) -> List[UeSample]:
        if self._idx >= len(self._rows):
            if self._loop:
                self._idx = 0
            else:
                return []
        row = self._rows[self._idx]
        self._idx += 1
        return [UeSample(**ue) for ue in row.get("ue_samples", [])]

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )

    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None:
        pass  # a fixed trace replay can't respond to admission decisions at all

    def close(self) -> None:
        pass


class SyntheticKpmSource:
    """Deterministic (seeded) procedural KPM feed for offline dev/testing,
    requiring no hand-authored fixture file.

    IMPORTANT LIMITATION, confirmed by inspection: this source is
    open-loop. poll() samples PRB demand, queue backlog, and loss
    independently every call; send_control() is recorded but never fed
    back into future poll() output. That means admission (accept/reject)
    decisions have literally no effect on future state here, only on the
    step's block-count and service-reward terms -- so "always accept" is
    trivially reward-optimal in this source, and training against it only
    validates the training *loop* (replay buffers, checkpointing, logging),
    not learned SLA-tradeoff behaviour. Use ClosedLoopKpmSource (below) for
    any offline run whose numbers you intend to treat as meaningful; keep
    this one only for fast unit tests / wiring smoke checks that don't
    depend on genuine admission-demand feedback.
    """

    def __init__(
        self,
        seed: int,
        gnb_ids: List[str],
        slice_ids: List[str],
        mean_prb_per_ue: Optional[Dict[str, float]] = None,
        mean_ues_per_slice: Optional[Dict[str, float]] = None,
    ):
        self._rng = np.random.RandomState(seed)
        self._gnb_ids = gnb_ids
        self._slice_ids = slice_ids
        self._mean_prb_per_ue = mean_prb_per_ue or {s: 3.0 for s in slice_ids}
        self._mean_ues_per_slice = mean_ues_per_slice or {s: 4.0 for s in slice_ids}
        self._t = 0.0
        self._rnti_counter = 0
        self.sent_controls: List[Dict[str, Any]] = []

    def poll(self) -> List[UeSample]:
        self._t += 1.0
        samples: List[UeSample] = []
        for gnb_id in self._gnb_ids:
            for slice_id_idx, slice_id in enumerate(self._slice_ids):
                sd = {"embb": 0, "urllc": 1, "mmtc": 2}[slice_id]
                n_ues = self._rng.poisson(self._mean_ues_per_slice[slice_id])
                for _ in range(n_ues):
                    self._rnti_counter += 1
                    prb = max(0.0, self._rng.normal(self._mean_prb_per_ue[slice_id], 1.0))
                    samples.append(
                        UeSample(
                            rnti=self._rnti_counter,
                            timestamp_s=self._t,
                            nssai_sst=1,
                            nssai_sd=sd,
                            avg_prbs_dl=prb,
                            gnb_id=gnb_id,
                            dl_total_bytes=prb * 1000.0,
                            dl_errors=max(0.0, self._rng.normal(0.1, 0.05)),
                            dl_bler=max(0.0, self._rng.normal(0.5, 0.3)),
                            dl_mac_buffer_occupation=max(0.0, self._rng.normal(5.0, 2.0)),
                        )
                    )
        return samples

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )

    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None:
        pass  # open-loop by design -- see class docstring; nothing feeds back regardless

    def close(self) -> None:
        pass


_SD_FOR_SLICE = {"embb": 0, "urllc": 1, "mmtc": 2}
_SD_FOR_SLICE_REVERSE = {v: k for k, v in _SD_FOR_SLICE.items()}


class ClosedLoopKpmSource:
    """Closed-loop synthetic KPM feed: served PRBs are capped by the
    admission ceiling most recently set via send_control(), and unmet
    demand accumulates as persistent queue backlog rather than vanishing --
    so admission decisions have a real, delayed cost/benefit an agent can
    actually learn from. This is the fix for the gap SyntheticKpmSource
    (above) has: there, demand is independent of admission, so accepting
    is costless and "learning" converges trivially.

    Model per (gNB, slice), each poll():
      1. offered demand follows a mean-reverting random walk (exogenous --
         NOT policy-driven; this is the incoming traffic the network can't
         control, matching a real cell's arrival process)
      2. served = min(offered + carried-over backlog, ceiling_prb), where
         ceiling_prb is derived from the *last* max_ratio sent via
         send_control() (default: unconstrained at B, i.e. only the
         slice's own quota cap applies, until the agent's first decision)
      3. unmet demand carries forward as backlog (-> reported via
         dl_mac_buffer_occupation, which is exactly the state's L_k(t) and
         the reward's queue-based SLA-violation signal)
      4. loss (dl_bler/dl_errors) scales with backlog fraction, so
         congestion also shows up on the loss-budget violation channel

    A persistent, slowly-churning UE population is used (rather than a
    fresh rnti every poll, as SyntheticKpmSource does) so RANEnv's
    new-RNTI admission-request channel doesn't fire on every single UE
    every step.

    Cross-gNB asymmetry: each gNB's offered demand is `mean_offered_ratio
    * gnb_load_multiplier[gnb_id]`. Without genuine heterogeneity across
    gNBs there is nothing for the LB (load-balance) term to actually
    resolve -- rho would trend near 1.0 regardless of policy quality,
    which is exactly what an earlier version of this source did (all gNBs
    given identical demand). If gnb_load_multiplier isn't supplied, one is
    auto-generated per gNB from `seed` (uniform in [0.6, 1.4], first gNB
    pinned to 1.0 so single-gNB configs are unaffected) so a caller can't
    silently end up back in the trivial-LB regime by omission.
    """

    def __init__(
        self,
        seed: int,
        gnb_ids: List[str],
        slice_ids: List[str],
        B: float = 100.0,
        mean_offered_ratio: Optional[Dict[str, float]] = None,
        gnb_load_multiplier: Optional[Dict[str, float]] = None,
        offered_volatility: float = 0.04,
        ues_per_slice: int = 4,
        backlog_capacity: float = 200.0,
        churn_prob: float = 0.05,
        initial_ceiling_ratio: float = 100.0,
    ):
        self._rng = np.random.RandomState(seed)
        self._gnb_ids = gnb_ids
        self._slice_ids = slice_ids
        self._B = B
        self._mean_offered_ratio = mean_offered_ratio or {s: 0.5 for s in slice_ids}
        self._gnb_load_multiplier = gnb_load_multiplier or self._default_gnb_load_multiplier(gnb_ids, seed)
        self._offered_volatility = offered_volatility
        self._ues_per_slice = ues_per_slice
        self._backlog_capacity = backlog_capacity
        self._churn_prob = churn_prob

        self._offered: Dict[tuple, float] = {}
        self._backlog: Dict[tuple, float] = {}
        self._ceiling_ratio: Dict[tuple, float] = {}
        self._ues: Dict[tuple, List[int]] = {}
        self._pending_relief: Dict[tuple, float] = {}
        self._rnti_counter = 0
        self._t = 0.0
        self.sent_controls: List[Dict[str, Any]] = []

        for gnb_id in gnb_ids:
            for slice_id in slice_ids:
                key = (gnb_id, slice_id)
                self._offered[key] = self._mean_offered_ratio[slice_id] * B * self._gnb_load_multiplier[gnb_id]
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
            for slice_id in self._slice_ids:
                key = (gnb_id, slice_id)
                sd = _SD_FOR_SLICE[slice_id]

                mean = self._mean_offered_ratio[slice_id] * self._B * self._gnb_load_multiplier[gnb_id]
                drift = 0.1 * (mean - self._offered[key])
                noise = self._rng.normal(0.0, self._offered_volatility * self._B)
                self._offered[key] = max(0.0, self._offered[key] + drift + noise)
                offered = self._offered[key]

                # A rejected admission request's traffic is turned away, not
                # merely capped -- relieve that much of the CARRIED backlog
                # before computing this step's demand, so reject has a real,
                # direct effect on this slice's own future SLA compliance
                # (not just on the serving ceiling, which by itself can only
                # ever make backlog worse, never better -- see
                # notify_rejected()'s docstring for the fuller rationale).
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

    def send_control(self, gnb_id: str, sst: int, sd: int, min_ratio: int, max_ratio: int) -> None:
        self.sent_controls.append(
            {"gnb_id": gnb_id, "sst": sst, "sd": sd, "min_ratio": min_ratio, "max_ratio": max_ratio}
        )
        slice_id = _SD_FOR_SLICE_REVERSE.get(int(sd))
        key = (gnb_id, slice_id)
        if key in self._ceiling_ratio:
            self._ceiling_ratio[key] = float(max_ratio)

    def notify_rejected(self, gnb_id: str, slice_id: str, n_rejected: int) -> None:
        """A rejected admission request's traffic leaves the system rather
        than merely failing to raise the serving ceiling.

        Without this, admission decisions had real leverage over block rate
        but essentially none over that same slice's own backlog-based SLA
        compliance: send_control() only ever adjusts the ceiling, and
        poll()'s demand/served/backlog loop only checks the ceiling against
        (offered + carried backlog) -- rejecting doesn't reduce either of
        those inputs, so a slice's SLA compliance was identical regardless
        of how it was admitted, confirmed empirically (DQN and A2C showed
        byte-identical SLA-compliance numbers despite very different block
        rates, across a 10x Lmax sweep). This method lets a reject relieve
        (queued at self._pending_relief, applied at the next poll(), before
        that step's demand is computed) an amount of backlog proportional to
        the rejected request's own recent per-UE demand -- so a policy that
        actually turns away traffic (not just refuses to expand capacity for
        it) can measurably improve that slice's own future SLA compliance.
        """
        key = (gnb_id, slice_id)
        if key not in self._offered or n_rejected <= 0:
            return
        n_ues = max(1, len(self._ues[key]))
        per_request_demand = self._offered[key] / n_ues
        self._pending_relief[key] = self._pending_relief.get(key, 0.0) + n_rejected * per_request_demand

    def close(self) -> None:
        pass
