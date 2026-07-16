"""Reward implementing eq. (2) from paper #1 (MECON) and its LB-extended form
from paper #2 (ICRAIE):

  paper #1: r(s_t,a_t) = sum_k(omega_k * R_k - lambda_k) - mu * C_t * ||a_t||_1
  paper #2: r(s_t,a_t) = sum_k(omega_k * R_k - lambda_k) - mu * C_t * ||a_t||_1 - beta * imbalance_t

Interpretation notes (documented, not silent -- also mirrored into the
Omega-tuple `limitation` field of every run):

1. R_k is credited once per accepted request of slice type k this step
   (weighted by omega_k); lambda_k is charged once per step a slice is in
   SLA violation. The papers describe these at the level of "reward for
   serving" and "penalty for violating SLA" without specifying finer
   per-request attribution, so step-level, per-slice-type accounting is the
   most direct reading of the published formula.

2. ||a_t||_1 = number of accepted (admitted) requests this step -- this
   matches the papers' own gloss ("a_t||1 corresponds to the number of
   accepted requests").

3. The papers state F_t as "a fairness indicator ... the min/max PRB
   utilisation ratio", then write the reward term as "- beta * F_t".
   Subtracting the ratio directly would penalize GOOD balance (ratio near
   1.0), which contradicts the papers' own reported result that DRL agents
   achieve a HIGHER ratio than the LB-only baseline. We therefore implement
   the load-imbalance penalty as beta * (1 - fairness_ratio): this is the
   quantity whose minimization is consistent with the papers' narrative and
   results, and is our resolution of an ambiguity in the source text.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .config import QoeRewardWeights, RewardWeights, SliceSpec
from .types import ClusterState

REWARD_INTERPRETATION_LIMITATION = (
    "reward eq.2's '-beta*F_t' term is implemented as beta*(1-fairness_ratio) "
    "(load imbalance), not a literal subtraction of the fairness ratio itself, "
    "because the latter would penalize good load balance -- see reward.py docstring."
)


@dataclass
class ViolationCheck:
    """Per-slice SLA violation flags for one step (cluster-wide, OR'd across gNBs).

    margin is the continuous counterpart to violated: 1.0 = comfortably
    within both the queue and loss budgets, 0.0 = exactly at budget,
    negative = past budget and NOT clamped -- more negative means more
    deeply over. violated is equivalent to `margin <= 0.0`, computed
    independently here as a direct threshold check (kept as its own hard
    gate, e.g. for a pass/fail acceptance bar).

    margin is deliberately unclamped and uses SliceAggState's raw
    (unclipped-by-Lmax) queue_len_norm, not the state/reward-facing clipped
    one: the clipped value bounds backlog reporting to [0,2.0] for NN input
    purposes, but that means two policies whose backlog is both deep past
    the violation threshold (e.g. raw/Lmax of 2.0 vs 20.0) would read back
    as the literal same clipped value -- hiding a real difference in how
    much worse one is doing than the other. Confirmed needed in practice:
    DQN's mean mmtc backlog was measurably lower than A2C's (~170.9 vs
    ~171.3) while both were "0% binary-compliant" after the first episode
    -- an unclamped, unclipped margin is what actually shows that gap.
    """

    violated: Dict[str, bool]
    margin: Dict[str, float]


def check_violations(cluster_state: ClusterState, slice_specs: Dict[str, SliceSpec]) -> ViolationCheck:
    violated: Dict[str, bool] = {slice_id: False for slice_id in slice_specs}
    margin: Dict[str, float] = {slice_id: 1.0 for slice_id in slice_specs}
    for slice_states in cluster_state.per_gnb.values():
        for slice_id, agg in slice_states.items():
            if slice_id not in slice_specs:
                continue
            spec = slice_specs[slice_id]
            queue_violation = agg.queue_len_norm > 1.0
            loss_violation = agg.loss_proxy > spec.loss_budget_pct
            if queue_violation or loss_violation:
                violated[slice_id] = True

            queue_margin = 1.0 - agg.raw_queue_len_norm
            loss_margin = (
                1.0 - agg.loss_proxy / spec.loss_budget_pct if spec.loss_budget_pct > 0 else 1.0
            )
            # Worst of the two channels, mirroring violated's OR: a slice is
            # only as compliant as its closer-to-breaching budget.
            margin[slice_id] = min(margin[slice_id], queue_margin, loss_margin)
    return ViolationCheck(violated=violated, margin=margin)


def _cluster_mean_congestion(cluster_state: ClusterState) -> float:
    values = [
        agg.congestion_level
        for slice_states in cluster_state.per_gnb.values()
        for agg in slice_states.values()
    ]
    return sum(values) / len(values) if values else 0.0


def compute_step_reward(
    cluster_state: ClusterState,
    slice_specs: Dict[str, SliceSpec],
    accepted_counts: Dict[str, int],
    weights: RewardWeights,
    include_lb_term: bool,
) -> "tuple[float, Dict[str, Any]]":
    violations = check_violations(cluster_state, slice_specs)

    service_term = 0.0
    violation_term = 0.0
    for slice_id, spec in slice_specs.items():
        n_accepted = accepted_counts.get(slice_id, 0)
        service_term += spec.priority_weight * spec.accept_reward * n_accepted
        if violations.violated.get(slice_id, False):
            violation_term += spec.violation_penalty

    total_accepted = sum(accepted_counts.values())
    mean_congestion = _cluster_mean_congestion(cluster_state)
    congestion_term = weights.congestion_coeff * mean_congestion * total_accepted

    reward = service_term - violation_term - congestion_term

    imbalance = None
    lb_term = 0.0
    if include_lb_term:
        imbalance = 1.0 - cluster_state.fairness_ratio
        lb_term = weights.lb_coeff * imbalance
        reward -= lb_term

    info: Dict[str, Any] = {
        "service_term": service_term,
        "violation_term": violation_term,
        "congestion_term": congestion_term,
        "lb_term": lb_term,
        "imbalance": imbalance,
        "mean_congestion": mean_congestion,
        "total_accepted": total_accepted,
        "violated_slices": {k: v for k, v in violations.violated.items() if v},
        # Full per-slice compliance for THIS step (True = SLA met), every
        # configured slice present regardless of violation status -- unlike
        # violated_slices above (which only lists the violated ones), this is
        # what SLA-compliance-rate reporting needs: a per-slice denominator
        # over every step, not just the numerator of violations.
        "per_slice_compliant": {k: not v for k, v in violations.violated.items()},
        # Continuous counterpart to per_slice_compliant -- see
        # ViolationCheck's docstring for why this exists alongside the
        # binary flag (it surfaces graded differences the binary flag
        # clips away once both compared runs cross the same threshold).
        "per_slice_sla_margin": dict(violations.margin),
        "reward": reward,
    }
    if include_lb_term:
        info["limitation"] = REWARD_INTERPRETATION_LIMITATION
    return reward, info


QOE_REWARD_LIMITATION = (
    "compute_qoe_reward implements the survey's eq.(9) form "
    "(r=alpha*MOS-beta*cost-gamma*SLA_viol), a DIFFERENT reward SHAPE from "
    "eq.2 above, not eq.2 with one term swapped -- MOS replaces the static "
    "priority_weight-based service_term entirely, cost is a direct PRB-"
    "expenditure term (not folded into a per-slice sum), and SLA_viol is a "
    "continuous severity average (from ViolationCheck.margin) rather than a "
    "flat per-violated-slice charge. See reward.py::compute_qoe_reward "
    "docstring and Stage One's task-4 instruction ('swap the reward's MOS "
    "source ... and ONLY that') for why this coexists with, rather than "
    "replaces, compute_step_reward."
)


QOE_DIAGNOSTIC_ONLY_LIMITATION = (
    "mean_mos/mos_by_slice/cost/sla_viol on this reward_mode='sla' step are "
    "QoE-mapper diagnostics computed for ablation comparability against "
    "reward_mode='qoe' runs -- NOT part of this run's actual reward signal, "
    "which remains eq.2's compute_step_reward, unchanged."
)


def compute_qoe_reward(
    cluster_state: ClusterState,
    slice_specs: Dict[str, SliceSpec],
    accepted_counts: Dict[str, int],
    mos_by_slice: Dict[str, float],
    qoe_weights: QoeRewardWeights,
) -> Tuple[float, Dict[str, Any]]:
    """eq.(9) of the survey (paper #3): r_t = alpha*MOS(QoS) - beta*cost -
    gamma*SLA_viol. This is Stage One's QoE-aware reward mode -- used ONLY
    by the "+QoE-driven-ratio" ablation arm; the frozen ratio-control
    baseline (LbOnlyHeuristic, and DQN/A2C's original SLA-based training)
    keeps using compute_step_reward (eq.2) unchanged.

    mos_by_slice: this step's QoE-mapper-inferred MOS per slice (already
    computed upstream by the caller -- env.py owns the mapper's stateful
    rolling window / latency-proxy tracking; this function stays pure and
    stateless, matching compute_step_reward's design, and doesn't import
    qoe_mapper.py itself).

    cost: realised PRB expenditure this step. Reuses the same
    mean_congestion*total_accepted basis compute_step_reward's
    congestion_term already uses (one definition of "cost" across both
    reward modes, not two inconsistent ones).

    SLA_viol: mean per-slice violation SEVERITY (not a flat per-violation
    charge like eq.2's violation_term) -- uses ViolationCheck.margin,
    clipped to [0, 1] per slice via max(0, -margin), so a slice that's
    deeply over budget contributes more severity than one that just
    crossed the line, then averaged across all configured slices (not just
    the ones with pending requests this step, so SLA_viol reflects the
    whole cluster's standing state, matching eq.9's role as a persistent
    penalty term rather than a per-request one).
    """
    violations = check_violations(cluster_state, slice_specs)

    total_accepted = sum(accepted_counts.values())
    mean_congestion = _cluster_mean_congestion(cluster_state)
    cost = mean_congestion * total_accepted

    mos_values = [mos_by_slice[s] for s in slice_specs if s in mos_by_slice]
    mean_mos = (sum(mos_values) / len(mos_values)) if mos_values else 3.0
    mos_norm = (mean_mos - 1.0) / 4.0  # [1,5] -> [0,1], matching the starter scaffold's own normalisation

    severities = [max(0.0, -violations.margin.get(s, 0.0)) for s in slice_specs]
    sla_viol = min(1.0, (sum(severities) / len(severities)) if severities else 0.0)

    reward = (
        qoe_weights.alpha * mos_norm - qoe_weights.beta * cost - qoe_weights.gamma * sla_viol
    )

    info: Dict[str, Any] = {
        "mean_mos": mean_mos,
        "mos_by_slice": dict(mos_by_slice),
        "cost": cost,
        "sla_viol": sla_viol,
        "violated_slices": {k: v for k, v in violations.violated.items() if v},
        "per_slice_compliant": {k: not v for k, v in violations.violated.items()},
        "per_slice_sla_margin": dict(violations.margin),
        "reward": reward,
        "limitation": QOE_REWARD_LIMITATION,
    }
    return reward, info
