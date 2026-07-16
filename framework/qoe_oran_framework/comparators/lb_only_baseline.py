"""Contextless baseline: matches the papers' own stated baseline -- "a
classical LB heuristic ... allocates incoming slice requests to the base
station with the lowest current PRB utilisation ... does not account for
SLA priorities or learning-based optimisation" -- used because Stage
Zero's acceptance numbers are measured against *that* baseline in the
source papers, not an arbitrary random policy.

Simplification, documented: the papers' baseline *routes* a request to
whichever gNB currently has the lowest utilisation. Stage Zero's action
space is a fixed accept/reject decision on a request that already carries
a specific (slice, gNB) assignment (see env.py's _synthesize_requests) --
there is no "choose a different gNB" action. The closest faithful analogue
on this fixed action space is: admit unless the request's assigned gNB is
already at/above a saturation threshold, with zero SLA/priority awareness
-- which preserves the paper's core property (no learning, no SLA
weighting, purely utilisation-driven) even though it cannot literally
re-route.

REDESIGN (was: whole-gNB aggregate utilisation only). The original
version only ever rejected when cluster_state's per-slice
congestion_level -- the SUM of PRB usage across *all* slices at that gNB
-- crossed 0.97. Confirmed live: this is structurally blind to per-slice
scarcity. Real total demand across three slices stayed ~0.15 even after
tightening each slice's own quota well below its individual demand
(secondary_block_count, i.e. demand exceeding the per-slice ceiling, was
a persistent 3/step) -- the heuristic never saw it, because it was never
looking at the right number. A baseline that can only ever say "yes" is
not "contextless", it's inert.

Fixed by adding a second, still-context-blind check: does *this specific
slice's own* observed PRB usage (ClusterState's per-slice prb_used_ratio)
already meet or exceed its configured nominal quota? This is exactly the
kind of raw-capacity check a real "no SLA priority, no learning" admission
heuristic would do -- it's still purely a static-quota/utilisation
comparison, no priority weighting, no learned value function -- just
evaluated at the right granularity (per-slice, not gNB-aggregate). Reject
if *either* the whole gNB is saturated (preserves the original LB-routing
analogue) *or* the specific slice is already at/over its own quota
(closes the blind spot).
"""

from typing import List

from ..config import SacLbExperimentConfig
from ..types import AdmissionRequest, ClusterState

LB_ONLY_ROUTING_LIMITATION = (
    "LbOnlyHeuristic approximates the papers' route-to-lowest-utilisation-gNB "
    "baseline as admit-unless-assigned-gNB-is-saturated-or-slice-is-at-quota, "
    "since Stage Zero's action space is accept/reject on a pre-assigned "
    "(slice, gNB) request, not gNB routing -- see lb_only_baseline.py module "
    "docstring for the two-condition rejection rule and why the original "
    "gNB-aggregate-only check was structurally blind to per-slice scarcity."
)


class LbOnlyHeuristic:
    """Not an RLPolicy (no learning, no checkpoints) -- exposes decide(),
    called directly by mc_runner instead of select_action().

    Rejects if EITHER:
      - the whole gNB's aggregate utilisation is at/above
        utilization_threshold (the original LB-routing analogue), OR
      - the request's own slice's observed PRB usage is at/above
        capacity_margin x its configured nominal_ratio (a raw per-slice
        capacity check -- still no SLA priority weighting, still no
        learning, just evaluated at the granularity that actually matters)
    """

    def __init__(
        self,
        cfg: SacLbExperimentConfig,
        utilization_threshold: float = 0.97,
        capacity_margin: float = 1.0,
    ):
        self.cfg = cfg
        self.utilization_threshold = utilization_threshold
        self.capacity_margin = capacity_margin

    def decide(self, pending: List[AdmissionRequest], cluster_state: ClusterState) -> List[int]:
        actions = []
        for request in pending:
            slice_states = cluster_state.per_gnb.get(request.gnb_id, {})
            agg = slice_states.get(request.slice_id)
            if agg is None:
                actions.append(1)  # nothing observed yet for this slice -> nothing to reject against
                continue

            gnb_saturated = agg.congestion_level >= self.utilization_threshold

            slice_at_quota = False
            spec = self.cfg.slice_by_id.get(request.slice_id)
            if spec is not None:
                quota_ratio = (spec.nominal_ratio / 100.0) * self.capacity_margin
                slice_at_quota = agg.prb_used_ratio >= quota_ratio

            actions.append(0 if (gnb_saturated or slice_at_quota) else 1)
        return actions
