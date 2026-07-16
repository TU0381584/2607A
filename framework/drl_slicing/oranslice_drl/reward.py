from typing import Dict, Tuple

from .config import SliceTarget
from .types import EnvState


PROFILE_PRIORITY_WEIGHTS = {
    "urllc": 4.0,
    "embb": 2.0,
    "mmtc": 1.0,
}


def _infer_profile_from_budgets(latency_budget_ms: float, loss_budget_pct: float) -> str:
    if latency_budget_ms <= 20.0 or loss_budget_pct <= 1.0:
        return "urllc"
    if latency_budget_ms <= 50.0 or loss_budget_pct <= 2.5:
        return "embb"
    return "mmtc"


def _resolve_slice_profile(target: SliceTarget) -> str:
    # Default triad mapping used by provisioning scripts:
    # SD 000000 -> eMBB, SD 000001 -> URLLC, SD 000002 -> mMTC.
    if int(target.sd) == 1:
        return "urllc"
    if int(target.sd) == 0:
        return "embb"
    if int(target.sd) == 2:
        return "mmtc"
    return _infer_profile_from_budgets(target.latency_budget_ms, target.loss_budget_pct)


def compute_sla_reward(state: EnvState, targets: Dict[str, SliceTarget]) -> Tuple[float, int]:
    total_penalty = 0.0
    efficiency_bonus = 0.0
    violations = 0

    for slice_id, metrics in state.slices.items():
        target = targets[slice_id]
        profile = _resolve_slice_profile(target)
        profile_weight = float(PROFILE_PRIORITY_WEIGHTS.get(profile, 1.0))

        latency_over = max(0.0, metrics.latency_ms - target.latency_budget_ms)
        loss_over = max(0.0, metrics.loss_pct - target.loss_budget_pct)
        if latency_over > 0.0 or loss_over > 0.0:
            violations += 1
        penalty = profile_weight * (
            2.0 * (latency_over / max(target.latency_budget_ms, 1e-6))
            + 3.0 * (loss_over / max(target.loss_budget_pct, 1e-6))
        )
        total_penalty += float(penalty)

        offered = max(metrics.offered_load_kbps, 1.0)
        efficiency_bonus += min(metrics.throughput_kbps / offered, 1.0)

    reward = efficiency_bonus - total_penalty
    return reward, violations
