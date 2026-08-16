"""Monte-Carlo orchestration for the MARL policies (GatCtdeMarlPolicy,
IndependentPerGnbDqnPolicy) -- the multi-agent analogue of
qoe_oran_framework.mc_runner.run_single/run_mc, reusing that module's
OmegaLogger/_make_omega_tuple (imported, not copied) so the resulting
omega_log.jsonl schema is identical and every existing plotting/compliance
script keeps working unmodified. mc_runner.py itself is not touched --
this is a new, parallel file for the two policy classes it doesn't know
about (both expose a select_actions(node_features, requests, training) /
train_step(batch) interface mc_runner's single-agent _select_actions /
_store_and_train helpers don't match).

Per docs/PAPER5_M1_recalibration.md's conclusion, this always drives an
OFFLINE KpmSource (ClosedLoopKpmSource or a subclass) -- a live-anchored
stress environment for the contention regime, never a live claim.
"""
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import SacLbExperimentConfig
from ..env import RANEnv
from ..mc_runner import _make_omega_tuple  # noqa: SLF001 -- reused, not modified
from ..omega_logger import OmegaLogger
from .disruption import (
    DisruptionSpec, corrupt_node_features, force_reject_actions,
    spike_multiplier_for_step, splice_churn_actions,
)
from .marl_env import extract_node_features, requests_to_agent_contexts


class JointReplayBuffer:
    """Stores one dict per step transition (ragged per-request fields --
    a step can have zero, one, or several pending requests across all
    agents). Sampling returns the same ragged-list-of-arrays shape
    GatCtdeMarlPolicy.train_step / IndependentPerGnbDqnPolicy.train_step
    both expect."""

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.pos = 0
        self._rng = np.random.RandomState(seed)

    def add(self, transition: Dict[str, Any]) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, Any]:
        idxs = self._rng.randint(0, len(self.buffer), size=min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in idxs]
        return {
            "node_features": [b["node_features"] for b in batch],
            "next_node_features": [b["next_node_features"] for b in batch],
            "agent_request_agent_idx": [b["agent_request_agent_idx"] for b in batch],
            "agent_request_context": [b["agent_request_context"] for b in batch],
            "agent_request_action": [b["agent_request_action"] for b in batch],
            "rewards": np.array([b["reward"] for b in batch], dtype=np.float32),
            "dones": np.array([b["done"] for b in batch], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.buffer)


def run_episodes_marl(
    env: RANEnv,
    policy: Any,
    algorithm_name: str,
    omega: OmegaLogger,
    n_episodes: int,
    seed: int,
    run_id: str,
    mode: str,
    training: bool,
    cfg: SacLbExperimentConfig,
    batch_size: int = 16,
    replay_capacity: int = 10000,
    warmup_transitions: int = 32,
    extra_limitations: Optional[List[str]] = None,
    disruption: Optional[DisruptionSpec] = None,
    disruption_fresh_policy: Optional[Any] = None,
) -> Dict[str, Any]:
    """disruption/disruption_fresh_policy: M4 evaluation-time perturbation
    hooks (see disruption.py's module docstring). Both default to None,
    which is a complete no-op -- every line below that reads `disruption`
    is guarded by `if disruption is not None`, so M2/M3's existing calls
    (which never pass these) run through byte-identically to before this
    parameter was added (verified: docs/PAPER5_M4_disruption.md's
    regression check). disruption_fresh_policy is only consulted for
    kind="churn" and must expose the same select_actions(...) interface as
    `policy`."""
    np.random.seed(seed)
    extra_limitations = extra_limitations or []
    replay_buffer = JointReplayBuffer(capacity=replay_capacity, seed=seed) if training else None
    base_arrivals_per_step = cfg.arrivals.synthetic_arrivals_per_step

    episode_sla_compliance_all_slices: List[float] = []
    episode_sla_compliance_by_slice: List[Dict[str, float]] = []

    try:
        for episode_idx in range(1, n_episodes + 1):
            if disruption is not None:
                # Step 1's pending list is produced by THIS reset() call, not
                # by a step() call inside the loop below -- see
                # spike_multiplier_for_step's docstring for why a spike
                # active at step 1 needs the arrivals knob set before reset()
                # specifically, not just before the loop's first env.step().
                cfg.arrivals.synthetic_arrivals_per_step = spike_multiplier_for_step(
                    disruption, 1, base_arrivals_per_step
                )
            env.reset()
            block_by_slice: Dict[str, int] = {}
            rho_values: List[float] = []
            compliant_steps_by_slice: Dict[str, int] = {s: 0 for s in cfg.slice_by_id}
            all_slices_compliant_steps = 0
            sla_margin_sum_by_slice: Dict[str, float] = {s: 0.0 for s in cfg.slice_by_id}
            step_idx = 0

            while True:
                step_idx += 1
                pending = env.pending_requests()
                cluster_state = env.last_cluster_state
                node_features = extract_node_features(cluster_state, cfg)
                requests_ctx = requests_to_agent_contexts(pending, cfg)

                if disruption is not None:
                    node_features = corrupt_node_features(node_features, disruption, step_idx)

                actions = policy.select_actions(node_features, requests_ctx, training=training)

                if disruption is not None and disruption.kind == "churn" and disruption_fresh_policy is not None \
                        and disruption.active_at(step_idx):
                    fresh_actions = disruption_fresh_policy.select_actions(node_features, requests_ctx, training=False)
                    actions = splice_churn_actions(actions, fresh_actions, requests_ctx, disruption, step_idx)
                if disruption is not None:
                    actions = force_reject_actions(actions, requests_ctx, disruption, step_idx)
                    # The request list env.step() synthesizes internally
                    # (for step_idx+1) reads cfg.arrivals AT THIS CALL --
                    # set it for the step about to be produced, not the one
                    # just processed.
                    cfg.arrivals.synthetic_arrivals_per_step = spike_multiplier_for_step(
                        disruption, step_idx + 1, base_arrivals_per_step
                    )

                result = env.step(actions)
                next_node_features = extract_node_features(env.last_cluster_state, cfg)

                if training and pending:
                    replay_buffer.add({
                        "node_features": node_features,
                        "next_node_features": next_node_features,
                        "agent_request_agent_idx": [a for a, _ in requests_ctx],
                        "agent_request_context": [c for _, c in requests_ctx],
                        "agent_request_action": list(actions),
                        "reward": result.reward,
                        "done": float(result.done),
                    })
                    if len(replay_buffer) >= max(batch_size, warmup_transitions):
                        policy.train_step(replay_buffer.sample(batch_size))

                for block in result.info["primary_blocks"]:
                    block_by_slice[block["slice_id"]] = block_by_slice.get(block["slice_id"], 0) + 1
                rho_values.append(result.info["fairness_ratio"])

                per_slice_compliant = result.info["reward_breakdown"].get("per_slice_compliant", {})
                for slice_id, compliant in per_slice_compliant.items():
                    if compliant:
                        compliant_steps_by_slice[slice_id] = compliant_steps_by_slice.get(slice_id, 0) + 1
                if per_slice_compliant and all(per_slice_compliant.values()):
                    all_slices_compliant_steps += 1

                per_slice_sla_margin = result.info["reward_breakdown"].get("per_slice_sla_margin", {})
                for slice_id, margin in per_slice_sla_margin.items():
                    sla_margin_sum_by_slice[slice_id] = sla_margin_sum_by_slice.get(slice_id, 0.0) + margin

                evidence = {
                    "seed": seed,
                    "reward": result.reward,
                    "primary_block_count": result.info["primary_block_count"],
                    "secondary_block_count": result.info["secondary_block_count"],
                    "accepted_counts": result.info["accepted_counts"],
                    "fairness_ratio": result.info["fairness_ratio"],
                    "n_pending": len(pending),
                    "ceilings": result.info.get("ceilings"),
                    "per_slice_compliant": per_slice_compliant,
                    "per_slice_sla_margin": per_slice_sla_margin,
                }
                omega.log(_make_omega_tuple(
                    algorithm_name, cfg, evidence, list(result.info["limitations"]) + extra_limitations,
                    run_id, episode_idx, step_idx, float(result.info["global_step"]), mode,
                ))

                if result.done:
                    break

            if training and hasattr(policy, "on_episode_end"):
                policy.on_episode_end()

            episode_steps = step_idx
            episode_sla_by_slice = {
                s: (compliant_steps_by_slice.get(s, 0) / episode_steps if episode_steps else 1.0)
                for s in cfg.slice_by_id
            }
            episode_sla_all = all_slices_compliant_steps / episode_steps if episode_steps else 1.0
            episode_sla_compliance_by_slice.append(episode_sla_by_slice)
            episode_sla_compliance_all_slices.append(episode_sla_all)
            episode_margin_by_slice = {
                s: (sla_margin_sum_by_slice.get(s, 0.0) / episode_steps if episode_steps else 1.0)
                for s in cfg.slice_by_id
            }
            episode_block_total = sum(block_by_slice.values())
            episode_rho_mean = float(np.mean(rho_values)) if rho_values else None

            omega.log(_make_omega_tuple(
                algorithm_name, cfg,
                {
                    "seed": seed,
                    "episode_block_total": episode_block_total,
                    "episode_block_by_slice": block_by_slice,
                    "episode_mean_rho": episode_rho_mean,
                    "episode_sla_compliance_by_slice": episode_sla_by_slice,
                    "episode_sla_compliance_all_slices": episode_sla_all,
                    "episode_sla_margin_by_slice": episode_margin_by_slice,
                    "rollup": True,
                },
                extra_limitations, run_id, episode_idx, -1, float(episode_idx), mode,
            ))
    finally:
        # Always restore, even if disruption is None (no-op then, since the
        # value is unchanged) or an exception propagates mid-campaign -- the
        # calling script's cfg object is often reused across seeds/arms.
        cfg.arrivals.synthetic_arrivals_per_step = base_arrivals_per_step

    return {
        "sla_compliance_all_slices": float(np.mean(episode_sla_compliance_all_slices))
        if episode_sla_compliance_all_slices else 1.0,
        "n_episodes": n_episodes,
    }
