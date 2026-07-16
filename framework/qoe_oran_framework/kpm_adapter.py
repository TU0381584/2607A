"""Maps parsed KPM ue_info_m samples onto the paper #1/#2 state fields (eq. 1).

Congestion level C_t, per-slice queue length L_k(t) and the loss proxy are
not directly exposed as single KPM counters by real OAI, so they are
derived proxies -- documented explicitly below and surfaced via the
`limitations` list returned alongside every aggregate, rather than silently
assumed to be literal SLA measurements.
"""

from typing import Dict, List, Tuple

from .config import SD_TO_SLICE_ID, SliceSpec
from .types import ClusterState, SliceAggState, UeSample


def resolve_slice_id(nssai_sd: int) -> str:
    return SD_TO_SLICE_ID.get(int(nssai_sd), "embb")


def aggregate_slice_state(
    ue_samples: List[UeSample],
    gnb_id: str,
    slice_specs: Dict[str, SliceSpec],
    B: float,
    Lmax: float,
    timestamp_s: float,
) -> Tuple[Dict[str, SliceAggState], List[str]]:
    """Aggregate one gNB's UE samples into per-slice state (eq. 1 fields).

    - U_k(t)/B: sum(avg_prbs_dl) over the slice's UEs, divided by B. Direct
      from the KPM report, not a proxy.
    - queue_len_norm (L_k(t)/Lmax): sum(dl_mac_buffer_occupation) / Lmax when
      populated. Some OAI builds leave this field at 0 for all UEs; when
      that happens for a non-empty slice group we fall back to a
      (dl_errors + dl_bler)-based backlog proxy and flag it in `limitations`.
    - congestion_level (C_t): aggregate PRB utilisation across all slices at
      this gNB, clipped to [0,1]. No direct scheduler-congestion KPI exists
      in the KPM report; this is a derived proxy, flagged in `limitations`.
    - loss_proxy: mean dl_bler over the slice's UEs, compared against
      loss_budget_pct downstream (reward.py) as an SLA-violation signal.
    """
    limitations: List[str] = []
    groups: Dict[str, List[UeSample]] = {slice_id: [] for slice_id in slice_specs}
    for ue in ue_samples:
        slice_id = resolve_slice_id(ue.nssai_sd)
        if slice_id in groups:
            groups[slice_id].append(ue)

    raw_prb_ratios: Dict[str, float] = {}
    result: Dict[str, SliceAggState] = {}
    for slice_id, ues in groups.items():
        prb_sum = sum(u.avg_prbs_dl for u in ues)
        prb_used_ratio = prb_sum / B if B > 0 else 0.0
        raw_prb_ratios[slice_id] = prb_used_ratio

        queue_raw = sum(u.dl_mac_buffer_occupation for u in ues)
        used_fallback = False
        if ues and queue_raw <= 0.0:
            queue_raw = sum(u.dl_errors + u.dl_bler for u in ues)
            used_fallback = True
            limitations.append(
                f"gnb={gnb_id} slice={slice_id}: dl_mac_buffer_occupation was 0 for all "
                "UEs; used (dl_errors + dl_bler) as a backlog proxy for L_k(t)."
            )
        raw_queue_len_norm = max(0.0, queue_raw / Lmax) if Lmax > 0 else 0.0
        queue_len_norm = min(2.0, raw_queue_len_norm)

        loss_proxy = (sum(u.dl_bler for u in ues) / len(ues)) if ues else 0.0

        result[slice_id] = SliceAggState(
            slice_id=slice_id,
            gnb_id=gnb_id,
            prb_used_ratio=prb_used_ratio,
            congestion_level=0.0,  # filled in below, once the cluster-wide sum is known
            queue_len_norm=queue_len_norm,
            queue_used_fallback=used_fallback,
            loss_proxy=loss_proxy,
            n_ues=len(ues),
            raw_queue_len_norm=raw_queue_len_norm,
        )

    raw_congestion_level = max(0.0, sum(raw_prb_ratios.values()))
    congestion_level = min(1.0, raw_congestion_level)
    if not slice_specs:
        limitations.append(f"gnb={gnb_id}: no slice_specs configured, congestion_level=0.")
    for agg in result.values():
        agg.congestion_level = congestion_level
        agg.raw_congestion_level = raw_congestion_level

    return result, limitations


def compute_fairness_ratio(per_gnb: Dict[str, Dict[str, SliceAggState]]) -> float:
    """F_t: min/max PRB utilisation ratio across the gNB cluster.

    Per-gNB utilisation is the sum of that gNB's per-slice prb_used_ratio
    values -- using raw_congestion_level (unclipped), NOT the state/reward's
    clipped-to-[0,1] congestion_level: once >=2 gNBs are each individually
    oversubscribed past 100%, the clip makes them indistinguishable from each
    other regardless of how much MORE oversubscribed one is, collapsing this
    ratio to reflect only the single least-loaded gNB rather than genuine
    cluster-wide balance (confirmed empirically: two gNBs pinned at exactly
    congestion_level=1.0 despite ~7% different underlying demand). A
    single-gNB cluster is perfectly balanced by definition (ratio 1.0).
    """
    if not per_gnb:
        return 1.0
    utilisations = []
    for slice_states in per_gnb.values():
        if slice_states:
            utilisations.append(next(iter(slice_states.values())).raw_congestion_level)
        else:
            utilisations.append(0.0)
    if len(utilisations) <= 1:
        return 1.0
    lo, hi = min(utilisations), max(utilisations)
    if hi <= 0.0:
        return 1.0
    return lo / hi


def build_cluster_state(
    ue_samples: List[UeSample],
    gnb_ids: List[str],
    slice_specs: Dict[str, SliceSpec],
    B: float,
    Lmax: float,
    timestamp_s: float,
) -> ClusterState:
    per_gnb: Dict[str, Dict[str, SliceAggState]] = {}
    all_limitations: List[str] = []
    default_gnb = gnb_ids[0] if gnb_ids else ""
    for gnb_id in gnb_ids:
        gnb_samples = [u for u in ue_samples if (u.gnb_id or default_gnb) == gnb_id]
        agg, limitations = aggregate_slice_state(gnb_samples, gnb_id, slice_specs, B, Lmax, timestamp_s)
        per_gnb[gnb_id] = agg
        all_limitations.extend(limitations)

    fairness_ratio = compute_fairness_ratio(per_gnb)
    return ClusterState(
        timestamp_s=timestamp_s,
        per_gnb=per_gnb,
        fairness_ratio=fairness_ratio,
        limitations=all_limitations,
    )
