from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class UeSample:
    """One parsed ue_info_m record from a RAN_indication_response.

    Field names mirror xapp-oai/base-xapp/oai-oran-protolib/ran_messages.proto
    (ue_info_m) so parsing code can construct these with **kwargs from the
    protobuf message directly.
    """

    rnti: int
    timestamp_s: float
    nssai_sst: int
    nssai_sd: int
    avg_prbs_dl: float
    gnb_id: str = ""
    dl_total_bytes: float = 0.0
    dl_errors: float = 0.0
    dl_bler: float = 0.0
    ul_total_bytes: float = 0.0
    ul_errors: float = 0.0
    dl_mac_buffer_occupation: float = 0.0


@dataclass
class SliceAggState:
    """Per-slice, per-gNB aggregate state for one step (eq. 1 fields)."""

    slice_id: str
    gnb_id: str
    prb_used_ratio: float          # U_k(t) / B
    congestion_level: float        # C_t (duplicated per slice for encoding convenience)
    queue_len_norm: float          # L_k(t) / Lmax
    queue_used_fallback: bool = False
    loss_proxy: float = 0.0        # derived from dl_bler, compared against loss_budget_pct
    n_ues: int = 0
    # Unclipped version of congestion_level, used only by compute_fairness_ratio.
    # congestion_level is clipped to [0,1] for state/reward purposes (bounded NN
    # input, bounded reward magnitude); but once >=2 gNBs are each individually
    # oversubscribed past 100% (this codebase's per-slice caps deliberately sum
    # to 110% of one gNB's PRB budget to guarantee persistent scarcity -- see
    # saclb_offline_dqn.yaml), the [0,1] clip makes them indistinguishable from
    # each other regardless of how much MORE oversubscribed one is than another
    # -- collapsing fairness_ratio's min/max computation to reflect only
    # whichever single gNB happens to be the least loaded, not genuine
    # cluster-wide balance. raw_congestion_level preserves the real relative
    # asymmetry for that one computation.
    raw_congestion_level: float = 0.0
    # Unclipped version of queue_len_norm, used only by reward.py's continuous
    # SLA margin. queue_len_norm is clipped to [0,2.0] for state/reward
    # purposes (bounded NN input); but that means two runs whose backlog is
    # both deep past the violation threshold (e.g. raw/Lmax of 2.0 vs 20.0)
    # read back as literally the same clipped value, hiding a real
    # difference in how badly one is doing versus the other -- confirmed
    # empirically (DQN's mean mmtc backlog measurably below A2C's, ~170.9 vs
    # ~171.3, invisible through the clipped value). raw_queue_len_norm
    # preserves that difference for margin's sake.
    raw_queue_len_norm: float = 0.0


@dataclass
class ClusterState:
    """RANEnv observation container (pre-encode)."""

    timestamp_s: float
    per_gnb: Dict[str, Dict[str, SliceAggState]] = field(default_factory=dict)
    fairness_ratio: float = 1.0    # F_t: min/max PRB utilisation ratio across the gNB cluster
    limitations: List[str] = field(default_factory=list)


@dataclass
class AdmissionRequest:
    request_id: str
    slice_id: str
    gnb_id: str
    arrival_step: int
    synthetic: bool = False


@dataclass
class BlockEvent:
    request_id: str
    slice_id: str
    gnb_id: str
    step: int
    kind: str   # "primary_reject" | "secondary_over_ceiling"


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
