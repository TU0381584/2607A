"""M4: evaluation-time disruption injection (docs/PAPER5_M4_disruption.md).

Pure, small helper functions -- no dependency on any other new-or-frozen
module beyond numpy -- consumed by `marl_training.run_episodes_marl`'s
optional `disruption=` parameter and by the standalone single-agent eval
loop in `experiments/scripts/m4_run_experiment.py`. Nothing here trains
anything or touches frozen framework source; every function either reads
its inputs and returns a new value, or mutates a plain, non-frozen config
dataclass field (`ArrivalConfig.synthetic_arrivals_per_step`) that this
project's own config.py already exposes as a runtime-mutable knob.

Three disruption kinds, each isolating a different failure mode:
  - "dropout": one gNB goes fully dark for a window -- its row in the
    joint node-feature matrix (or its slice of the flattened single-agent
    observation) is zeroed, AND its own pending requests are forced to
    reject regardless of what the policy would have chosen. Tests whether
    an architecture that attends across the full cluster state (GAT-CTDE,
    federated) degrades gracefully when one neighbor's telemetry goes
    missing, versus one that never looked at neighbors in the first place
    (independent_dqn) or only sees a flattened, corrupted joint vector
    (single_agent_dqn).
  - "spike": a transient multiplier on the arrivals config's
    synthetic_arrivals_per_step for a window, then restored -- tests
    robustness to a burst in offered load, independent of any topology
    assumption, so it applies uniformly to every arm.
  - "churn": one gNB's decisions are made by a freshly-initialized,
    never-trained copy of the same policy class instead of the loaded
    checkpoint, for a window. See docs/PAPER5_M4_disruption.md for why
    this replaces the originally-proposed "client churn" (federation only
    happens during training; a frozen post-aggregation checkpoint has no
    per-client staleness left to reintroduce, and no arm shares
    cross-agent information at INFERENCE time beyond what the encoder
    already reads from the joint state -- so there is nothing FL-specific
    left to churn). Applies only to genuinely multi-agent arms.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DisruptionSpec:
    kind: str  # "dropout" | "spike" | "churn"
    target_agent_idx: int
    start_step: int
    duration_steps: int
    severity_param: float = 0.0  # spike: arrival-rate multiplier. dropout/churn: unused (window length IS the severity).
    severity_label: str = ""  # human-readable tag for filenames/logging, e.g. "dropout_sev1"

    def active_at(self, step_in_episode: int) -> bool:
        return self.start_step <= step_in_episode < self.start_step + self.duration_steps


def corrupt_node_features(node_features: np.ndarray, spec: Optional[DisruptionSpec], step_in_episode: int) -> np.ndarray:
    """Returns a NEW array with spec.target_agent_idx's row zeroed if a
    dropout is active this step; otherwise returns node_features unchanged
    (same object, no copy, so the no-disruption path costs nothing extra)."""
    if spec is None or spec.kind != "dropout" or not spec.active_at(step_in_episode):
        return node_features
    out = node_features.copy()
    out[spec.target_agent_idx, :] = 0.0
    return out


def corrupt_flat_obs(obs: np.ndarray, gnb_ids: Sequence[str], n_slices: int,
                      spec: Optional[DisruptionSpec], step_in_episode: int) -> np.ndarray:
    """single_agent_dqn analogue of corrupt_node_features: obs is the
    flattened joint state env.encode_state() produces (each gNB occupies a
    contiguous n_slices*3 block, in gnb_ids order, per env.py's
    encode_state docstring) -- zero the dropped gNB's block."""
    if spec is None or spec.kind != "dropout" or not spec.active_at(step_in_episode):
        return obs
    if spec.target_agent_idx >= len(gnb_ids):
        return obs
    out = obs.copy()
    block = n_slices * 3
    start = spec.target_agent_idx * block
    out[start:start + block] = 0.0
    return out


def force_reject_actions(actions: List[int], requests_ctx: List[Tuple[int, np.ndarray]],
                          spec: Optional[DisruptionSpec], step_in_episode: int) -> List[int]:
    """MARL-arm version: requests_ctx is the [(agent_idx, slice_onehot), ...]
    list run_episodes_marl already builds via requests_to_agent_contexts."""
    if spec is None or spec.kind != "dropout" or not spec.active_at(step_in_episode):
        return actions
    out = list(actions)
    for i, (agent_idx, _ctx) in enumerate(requests_ctx):
        if agent_idx == spec.target_agent_idx:
            out[i] = 0
    return out


def force_reject_actions_single_agent(actions: List[int], pending_gnb_ids: List[str], target_gnb_id: str,
                                       spec: Optional[DisruptionSpec], step_in_episode: int) -> List[int]:
    """single_agent_dqn version: pending is a flat list of AdmissionRequest,
    here reduced by the caller to just their .gnb_id, matched against the
    single target gNB's own id (single_agent_dqn has one shared policy, no
    agent_idx concept)."""
    if spec is None or spec.kind != "dropout" or not spec.active_at(step_in_episode):
        return actions
    out = list(actions)
    for i, gnb_id in enumerate(pending_gnb_ids):
        if gnb_id == target_gnb_id:
            out[i] = 0
    return out


def splice_churn_actions(frozen_actions: List[int], fresh_actions: List[int],
                          requests_ctx: List[Tuple[int, np.ndarray]],
                          spec: Optional[DisruptionSpec], step_in_episode: int) -> List[int]:
    """Replaces frozen_actions[i] with fresh_actions[i] for every request
    belonging to spec.target_agent_idx while a churn window is active --
    every other agent's decisions are untouched."""
    if spec is None or spec.kind != "churn" or not spec.active_at(step_in_episode):
        return frozen_actions
    out = list(frozen_actions)
    for i, (agent_idx, _ctx) in enumerate(requests_ctx):
        if agent_idx == spec.target_agent_idx:
            out[i] = fresh_actions[i]
    return out


def spike_multiplier_for_step(spec: Optional[DisruptionSpec], step_in_episode: int, base_value: int) -> int:
    """Returns the arrivals-per-step value that should be in effect while
    the request list FOR `step_in_episode` is being synthesized -- see
    marl_training.py's call sites for the +1 step-offset this needs
    relative to the loop's own step_idx (env.step() synthesizes the NEXT
    step's pending list internally, at the end of the call)."""
    if spec is None or spec.kind != "spike" or not spec.active_at(step_in_episode):
        return base_value
    return max(1, int(round(base_value * spec.severity_param)))
