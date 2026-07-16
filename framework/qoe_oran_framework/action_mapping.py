"""Maps the papers' binary accept/reject admission action onto the only
control primitive real OAI actually exposes at the xApp layer:
slicing_control_m{sst, sd, min_ratio, max_ratio} (a PRB-ratio ceiling/floor
per NSSAI -- see xapp-oai/base-xapp/oai-oran-protolib/ran_messages.proto).

There is no raw "accept/reject this one flow" hook at the xApp layer, so
admission is realized as an *admission gate*: each accept/reject decision
nudges a per-(gNB,slice) PRB ceiling up or down, bounded by the slice's
configured floor/cap. This mapping -- not a literal per-flow accept at the
scheduler -- is the Stage Zero resolution of that gap, and is surfaced via
BLOCK_MAPPING_LIMITATION on every run.

Block-rate accounting has two tiers, kept separate on purpose:
  - primary_blocks: one per direct reject decision. Honest, requires no
    inference about scheduler behaviour, and is what "URLLC block rate"
    means throughout Stage Zero's acceptance criteria.
  - secondary_blocks: a corroborating signal -- if the *previous* ceiling
    for a slice was already below the demand observed at the *next* KPM
    report, the gNB could not actually have served that demand even had we
    accepted. This is reported alongside the primary count for diagnostic
    purposes, never summed into it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .config import SliceSpec
from .types import AdmissionRequest, BlockEvent, ClusterState

BLOCK_MAPPING_LIMITATION = (
    "admission decisions are realized via PRB-ceiling adjustment "
    "(slicing_control_m min/max_ratio), not a literal per-flow accept at the "
    "scheduler, because real OAI's xApp control API only exposes ratio "
    "ceilings -- see action_mapping.py module docstring."
)


@dataclass
class SliceCeiling:
    gnb_id: str
    slice_id: str
    min_ratio: int
    max_ratio: int


@dataclass
class ApplyResult:
    ceilings: Dict[Tuple[str, str], SliceCeiling]
    primary_blocks: List[BlockEvent] = field(default_factory=list)
    secondary_blocks: List[BlockEvent] = field(default_factory=list)
    accepted_counts: Dict[str, int] = field(default_factory=dict)


class AdmissionGate:
    def __init__(self, slice_specs: Dict[str, SliceSpec], gnb_ids: List[str], step_ratio: int = 5):
        self._specs = slice_specs
        self._step_ratio = step_ratio
        self._ceilings: Dict[Tuple[str, str], SliceCeiling] = {}
        for gnb_id in gnb_ids:
            for slice_id, spec in slice_specs.items():
                self._ceilings[(gnb_id, slice_id)] = SliceCeiling(
                    gnb_id=gnb_id,
                    slice_id=slice_id,
                    min_ratio=spec.min_ratio_floor,
                    max_ratio=spec.nominal_ratio,
                )

    def reset_ceilings(self) -> None:
        for (gnb_id, slice_id), spec in [
            (key, self._specs[key[1]]) for key in self._ceilings
        ]:
            self._ceilings[(gnb_id, slice_id)] = SliceCeiling(
                gnb_id=gnb_id, slice_id=slice_id,
                min_ratio=spec.min_ratio_floor, max_ratio=spec.nominal_ratio,
            )

    def ceiling_for(self, gnb_id: str, slice_id: str) -> SliceCeiling:
        return self._ceilings[(gnb_id, slice_id)]

    def apply(
        self,
        requests: List[AdmissionRequest],
        actions: List[int],
        previous_cluster_state: ClusterState,
        step: int,
    ) -> ApplyResult:
        if len(requests) != len(actions):
            raise ValueError(f"requests/actions length mismatch: {len(requests)} != {len(actions)}")

        accepted_counts: Dict[str, int] = {slice_id: 0 for slice_id in self._specs}
        primary_blocks: List[BlockEvent] = []
        changed: Dict[Tuple[str, str], SliceCeiling] = {}

        for request, action in zip(requests, actions):
            key = (request.gnb_id, request.slice_id)
            if key not in self._ceilings:
                continue
            spec = self._specs[request.slice_id]
            ceiling = self._ceilings[key]
            if action == 1:
                ceiling.max_ratio = min(spec.max_ratio_cap, ceiling.max_ratio + self._step_ratio)
                accepted_counts[request.slice_id] = accepted_counts.get(request.slice_id, 0) + 1
            else:
                # Reject also lowers the ceiling toward the floor. Tried
                # decoupling this (reject only relieves backlog via
                # KpmSource.notify_rejected(), never touches the ceiling) to
                # see if it would let SLA compliance differentiate DQN from
                # A2C -- it didn't move SLA compliance meaningfully, AND it
                # destabilized A2C's on-policy training into rejecting
                # urllc/eMBB too (previously a robust, verified 0 blocks),
                # so reverted back to this coupled behavior. notify_rejected
                # 's relief still runs alongside this -- it's real,
                # tested infrastructure, just not sufficient on its own to
                # make SLA compliance policy-differentiable in this
                # environment (see saclb_offline_live1gnb.yaml's Lmax
                # comment for the fuller diagnostic chain).
                ceiling.max_ratio = max(spec.min_ratio_floor, ceiling.max_ratio - self._step_ratio)
                primary_blocks.append(
                    BlockEvent(
                        request_id=request.request_id, slice_id=request.slice_id,
                        gnb_id=request.gnb_id, step=step, kind="primary_reject",
                    )
                )
            changed[key] = ceiling

        secondary_blocks = self._secondary_blocks(previous_cluster_state, step)

        return ApplyResult(
            ceilings=changed,
            primary_blocks=primary_blocks,
            secondary_blocks=secondary_blocks,
            accepted_counts=accepted_counts,
        )

    def _secondary_blocks(self, previous_cluster_state: ClusterState, step: int) -> List[BlockEvent]:
        """Flag slices whose observed demand, at the time this state was
        captured, already exceeded the ceiling that was in force -- i.e. the
        gNB could not have fully served that slice even absent any new
        reject decisions this step."""
        events: List[BlockEvent] = []
        for gnb_id, slice_states in previous_cluster_state.per_gnb.items():
            for slice_id, agg in slice_states.items():
                key = (gnb_id, slice_id)
                if key not in self._ceilings:
                    continue
                ceiling = self._ceilings[key]
                demand_ratio = agg.prb_used_ratio * 100.0
                if demand_ratio > ceiling.max_ratio:
                    events.append(
                        BlockEvent(
                            request_id=f"secondary:{gnb_id}:{slice_id}:{step}",
                            slice_id=slice_id, gnb_id=gnb_id, step=step,
                            kind="secondary_over_ceiling",
                        )
                    )
        return events
