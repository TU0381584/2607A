from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SliceMetrics:
    slice_id: str
    throughput_kbps: float
    latency_ms: float
    loss_pct: float
    offered_load_kbps: float = 0.0


@dataclass
class EnvState:
    timestamp_s: float
    slices: Dict[str, SliceMetrics] = field(default_factory=dict)


@dataclass
class SliceAction:
    slice_id: str
    min_ratio: int
    max_ratio: int
    dedicated_ratio: int


@dataclass
class StepRecord:
    step: int
    timestamp_s: float
    reward: float
    sla_violations: int
    action_summary: str
