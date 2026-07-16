"""Monte-Carlo orchestration: runs a policy through RANEnv for N episodes,
optionally training it (offline) or evaluating frozen weights (live),
fixed seeds throughout, one Omega-tuple per step plus an episode-level
rollup, and a structured drift flag for live runs.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ._oranslice_path import ensure_oranslice_drl_importable
from .comparators.lb_only_baseline import LB_ONLY_ROUTING_LIMITATION, LbOnlyHeuristic
from .config import SacLbExperimentConfig
from .env import RANEnv, encode_full_request_state, request_state_dim
from .omega_logger import OmegaLogger, OmegaTuple
from .policies.a2c_admission import A2CAdmissionPolicy
from .policies.dqn_admission import DQNAdmissionPolicy
from .policies.rainbow_admission import PrioritizedReplayBuffer, RainbowAdmissionPolicy
from .replay_kpm_source import KpmSource
from .types import AdmissionRequest

ensure_oranslice_drl_importable()
from oranslice_drl.drl_training import ReplayBuffer  # noqa: E402

REPLAY_BASED_ALGORITHMS = {"dqn", "rainbow"}
ON_POLICY_ALGORITHMS = {"a2c"}
NO_LEARNING_ALGORITHMS = {"lb_only"}

EPISODE_HORIZON_LIMITATION = (
    "episode horizon and step cadence follow Stage Zero's protocol (60 "
    "steps/episode, 5s/step live; offline training targets the full "
    "300-episode Table I schedule against a synthetic/replayed KPM feed), "
    "not a literal replay of the papers' original simulated training run."
)

REQUEST_BATCHING_LIMITATION = (
    "each pending request in a step is decided independently by "
    "concatenating a one-hot slice/gNB identity onto the shared step "
    "observation (see env.encode_request_context) and reusing the same "
    "policy instance per request, rather than the papers' one-request-per-step "
    "formulation -- see Stage Zero plan."
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


@dataclass
class RunSummary:
    run_id: str
    algorithm: str
    mode: str
    seed: int
    n_episodes: int
    mean_reward_per_step: float
    mean_blocks_per_episode: float
    mean_urllc_blocks_per_episode: Optional[float]
    blocks_per_episode: List[int]
    blocks_per_episode_by_slice: List[Dict[str, int]]
    mean_rho: Optional[float]
    omega_path: str
    checkpoint_path: str = ""
    drift_flag: bool = False
    drift_reason: str = ""
    # SLA compliance: fraction of steps (0.0-1.0) each slice was NOT in
    # violation (queue_len_norm<=1.0 AND loss_proxy<=loss_budget_pct -- see
    # reward.check_violations), averaged across episodes. Distinct from
    # block rate: a request can be accepted (not blocked) yet its slice can
    # still be in SLA violation this step (backlog/loss already over
    # budget from earlier accepts), and vice versa.
    sla_compliance_by_slice: Dict[str, float] = field(default_factory=dict)
    # Stricter joint measure: fraction of steps where EVERY configured
    # slice was simultaneously compliant (not just each slice's own
    # marginal rate) -- the "slice-wide" network-level number.
    sla_compliance_all_slices: float = 0.0
    sla_compliance_by_slice_per_episode: List[Dict[str, float]] = field(default_factory=list)
    # Continuous counterpart to sla_compliance_by_slice: mean per-slice SLA
    # margin (1.0 = comfortably within budget, 0.0 = at/beyond budget,
    # in between = graded distance -- see reward.ViolationCheck's
    # docstring). Exists because the binary compliance rate above can tie
    # two policies that both cross the violation threshold while one was
    # actually managing the slice's backlog/loss meaningfully better the
    # whole time -- confirmed happening in practice (DQN's mean mmtc
    # backlog measurably lower than A2C's, ~171 vs ~171.3, while both read
    # as 0% binary-compliant after the first episode).
    sla_margin_by_slice: Dict[str, float] = field(default_factory=dict)
    sla_margin_by_slice_per_episode: List[Dict[str, float]] = field(default_factory=list)
    # QoE-mode only (reward_mode="qoe"): populated from
    # reward.compute_qoe_reward's info dict (mean_mos/mos_by_slice/cost/
    # sla_viol) -- empty/0.0 for sla-mode runs, since compute_step_reward's
    # info dict has no such keys. mean across all steps of all episodes
    # (not episode-of-episode-means) to match how the qoe reward's cost/
    # sla_viol terms are themselves per-step quantities.
    mean_mos_by_slice: Dict[str, float] = field(default_factory=dict)
    mean_cost: float = 0.0
    mean_sla_viol: float = 0.0


def build_policy(algorithm: str, cfg: SacLbExperimentConfig, **overrides):
    dim = request_state_dim(cfg)
    if algorithm == "dqn":
        return DQNAdmissionPolicy(state_dim=dim, **overrides)
    if algorithm == "a2c":
        return A2CAdmissionPolicy(state_dim=dim, **overrides)
    if algorithm == "rainbow":
        return RainbowAdmissionPolicy(state_dim=dim, **overrides)
    if algorithm == "lb_only":
        return LbOnlyHeuristic(cfg, **overrides)
    raise ValueError(
        f"unknown algorithm {algorithm!r} (paper #1's SAC-only comparator is "
        "algorithm='dqn' run against a paper_variant: paper1 config -- see "
        "comparators/sac_only.py, not a separate algorithm string here)"
    )


def _select_actions(
    policy: Any,
    algorithm: str,
    pending: List[AdmissionRequest],
    obs: np.ndarray,
    cluster_state,
    cfg: SacLbExperimentConfig,
    training: bool,
) -> Tuple[List[int], List[Optional[np.ndarray]]]:
    if algorithm == "lb_only":
        actions = policy.decide(pending, cluster_state)
        return list(actions), [None] * len(pending)

    actions: List[int] = []
    request_states: List[Optional[np.ndarray]] = []
    for request in pending:
        req_state = encode_full_request_state(obs, request, cfg)
        action, _info = policy.select_action(req_state, training=training)
        actions.append(int(action))
        request_states.append(req_state)
    return actions, request_states


def _store_and_train(
    algorithm: str,
    policy: Any,
    replay_buffer: Any,
    pending: List[AdmissionRequest],
    obs: np.ndarray,
    next_obs: np.ndarray,
    actions: List[int],
    request_states: List[Optional[np.ndarray]],
    reward: float,
    done: bool,
    cfg: SacLbExperimentConfig,
    batch_size: int,
    warmup_transitions: int,
) -> None:
    if not pending:
        return

    if algorithm == "a2c":
        next_states = [encode_full_request_state(next_obs, req, cfg) for req in pending]
        batch = {
            "states": np.stack(request_states),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.full(len(pending), reward, dtype=np.float32),
            "next_states": np.stack(next_states),
            "dones": np.full(len(pending), float(done), dtype=np.float32),
        }
        policy.train_step(batch)
        return

    if algorithm not in REPLAY_BASED_ALGORITHMS or replay_buffer is None:
        return

    for request, action, req_state in zip(pending, actions, request_states):
        next_state = encode_full_request_state(next_obs, request, cfg)
        if algorithm == "dqn":
            replay_buffer.add(req_state, action, reward, next_state, done, action_info=None)
        else:  # rainbow
            replay_buffer.add(req_state, action, reward, next_state, done)

    if len(replay_buffer) < max(batch_size, warmup_transitions):
        return

    if algorithm == "dqn":
        batch = replay_buffer.sample(batch_size)
        policy.train_step(batch)
    else:  # rainbow
        batch = replay_buffer.sample(batch_size, beta=policy.per_beta)
        metrics = policy.train_step(batch)
        replay_buffer.update_priorities(batch["indices"], metrics["td_errors"])


def _make_omega_tuple(
    algorithm: str,
    cfg: SacLbExperimentConfig,
    evidence: Dict[str, Any],
    extra_limitations: List[str],
    run_id: str,
    episode: int,
    step: int,
    timestamp_s: float,
    mode: str,
) -> OmegaTuple:
    objective = "minimize URLLC block rate subject to eMBB/mMTC SLA and PRB budget"
    constraint = f"gNB capacity B={cfg.B}, slices={list(cfg.slice_by_id)}"
    if len(cfg.gnbs) > 1:
        objective += ", and minimize cluster load imbalance (paper #2 LB term)"
        constraint += ", cluster load-balance ratio rho target [0.55, 0.60]"
    limitations = list(extra_limitations) + [EPISODE_HORIZON_LIMITATION, REQUEST_BATCHING_LIMITATION]
    if algorithm == "lb_only":
        limitations.append(LB_ONLY_ROUTING_LIMITATION)
    return OmegaTuple(
        role="admission-controller",
        method=algorithm,
        objective=objective,
        constraint=constraint,
        evidence=evidence,
        limitation="; ".join(limitations),
        run_id=run_id,
        episode=episode,
        step=step,
        timestamp_s=timestamp_s,
        mode=mode,
    )


def run_single(
    env: RANEnv,
    policy: Any,
    algorithm: str,
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
) -> RunSummary:
    set_seeds(seed)
    extra_limitations = extra_limitations or []

    replay_buffer = None
    if training and algorithm == "dqn":
        replay_buffer = ReplayBuffer(capacity=replay_capacity)
    elif training and algorithm == "rainbow":
        replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=policy.per_alpha)

    episode_mean_rewards: List[float] = []
    episode_block_totals: List[int] = []
    episode_block_by_slice: List[Dict[str, int]] = []
    episode_rho_means: List[Optional[float]] = []
    episode_sla_compliance_by_slice: List[Dict[str, float]] = []
    episode_sla_compliance_all_slices: List[float] = []
    episode_sla_margin_by_slice: List[Dict[str, float]] = []
    # QoE-mode only -- flat, across every step of every episode (see
    # RunSummary.mean_mos_by_slice's docstring for why this isn't
    # episode-of-episode-means).
    all_step_mos_by_slice: Dict[str, List[float]] = {}
    all_step_cost: List[float] = []
    all_step_sla_viol: List[float] = []

    for episode_idx in range(1, n_episodes + 1):
        # reset() must run at the top of every episode, not just once before
        # the loop: it's what zeroes _step_in_episode and re-arms the
        # admission gate's ceilings. Without a per-episode reset, episode 1
        # runs the full steps_per_episode as intended, but _step_in_episode
        # keeps climbing past steps_per_episode forever after -- so every
        # later "episode" hits done=True on its very first step. UE/KPM
        # state is not reinitialized here (continuous real traffic), only
        # the RL bookkeeping -- matching the live-testbed semantics where a
        # physical gNB can't be reset like a simulator.
        obs = env.reset()
        step_rewards: List[float] = []
        block_by_slice: Dict[str, int] = {}
        rho_values: List[float] = []
        compliant_steps_by_slice: Dict[str, int] = {slice_id: 0 for slice_id in cfg.slice_by_id}
        all_slices_compliant_steps = 0
        sla_margin_sum_by_slice: Dict[str, float] = {slice_id: 0.0 for slice_id in cfg.slice_by_id}
        step_idx = 0

        while True:
            step_idx += 1
            step_start = time.monotonic() if mode == "live_testbed" else None
            pending = env.pending_requests()
            cluster_state = env.last_cluster_state
            actions, request_states = _select_actions(
                policy, algorithm, pending, obs, cluster_state, cfg, training
            )

            result = env.step(actions)
            next_obs = result.obs

            if step_start is not None:
                # Respect the configured live step cadence -- without this,
                # LiveKpmSource's poll()/send_control() round-trip completes
                # in milliseconds and the loop would hammer the gNB's E2
                # agent far faster than the papers' Monte-Carlo protocol
                # (and this run's own episode.step_seconds config) intends.
                elapsed = time.monotonic() - step_start
                remaining = cfg.episode.step_seconds - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            if training and algorithm not in NO_LEARNING_ALGORITHMS:
                _store_and_train(
                    algorithm, policy, replay_buffer, pending, obs, next_obs, actions,
                    request_states, result.reward, result.done, cfg, batch_size, warmup_transitions,
                )

            step_rewards.append(result.reward)
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

            # QoE-mode only -- these keys are absent from compute_step_reward's
            # info dict (sla-mode), so .get() leaves sla-mode runs untouched.
            mos_by_slice = result.info["reward_breakdown"].get("mos_by_slice")
            if mos_by_slice is not None:
                for slice_id, mos in mos_by_slice.items():
                    all_step_mos_by_slice.setdefault(slice_id, []).append(mos)
            step_cost = result.info["reward_breakdown"].get("cost")
            if step_cost is not None:
                all_step_cost.append(step_cost)
            step_sla_viol = result.info["reward_breakdown"].get("sla_viol")
            if step_sla_viol is not None:
                all_step_sla_viol.append(step_sla_viol)

            evidence = {
                "seed": seed,
                "reward": result.reward,
                "primary_block_count": result.info["primary_block_count"],
                "secondary_block_count": result.info["secondary_block_count"],
                "accepted_counts": result.info["accepted_counts"],
                "fairness_ratio": result.info["fairness_ratio"],
                "n_pending": len(pending),
                "ceilings": result.info.get("ceilings"),
                "mean_mos": result.info["reward_breakdown"].get("mean_mos"),
                "mos_by_slice": mos_by_slice,
                "cost": step_cost,
                "sla_viol": step_sla_viol,
                "per_slice_compliant": per_slice_compliant,
                "per_slice_sla_margin": per_slice_sla_margin,
            }
            omega.log(
                _make_omega_tuple(
                    algorithm, cfg, evidence, list(result.info["limitations"]) + extra_limitations,
                    run_id, episode_idx, step_idx, float(result.info["global_step"]), mode,
                )
            )

            obs = next_obs
            if result.done:
                break

        if training and algorithm in ("dqn", "rainbow") and hasattr(policy, "on_episode_end"):
            policy.on_episode_end()
        if training and algorithm == "rainbow":
            policy.anneal_beta(episode_idx / n_episodes)

        episode_block_total = sum(block_by_slice.values())
        episode_mean_rewards.append(float(np.mean(step_rewards)) if step_rewards else 0.0)
        episode_block_totals.append(episode_block_total)
        episode_block_by_slice.append(block_by_slice)
        episode_rho_mean = float(np.mean(rho_values)) if len(cfg.gnbs) > 1 and rho_values else None
        episode_rho_means.append(episode_rho_mean)

        episode_steps = step_idx
        episode_sla_by_slice = {
            slice_id: (compliant_steps_by_slice.get(slice_id, 0) / episode_steps if episode_steps else 1.0)
            for slice_id in cfg.slice_by_id
        }
        episode_sla_all = all_slices_compliant_steps / episode_steps if episode_steps else 1.0
        episode_sla_compliance_by_slice.append(episode_sla_by_slice)
        episode_sla_compliance_all_slices.append(episode_sla_all)
        episode_margin_by_slice = {
            slice_id: (sla_margin_sum_by_slice.get(slice_id, 0.0) / episode_steps if episode_steps else 1.0)
            for slice_id in cfg.slice_by_id
        }
        episode_sla_margin_by_slice.append(episode_margin_by_slice)

        omega.log(
            _make_omega_tuple(
                algorithm, cfg,
                {
                    "seed": seed,
                    "episode_mean_reward": episode_mean_rewards[-1],
                    "episode_block_total": episode_block_total,
                    "episode_block_by_slice": block_by_slice,
                    "episode_mean_rho": episode_rho_mean,
                    "episode_sla_compliance_by_slice": episode_sla_by_slice,
                    "episode_sla_compliance_all_slices": episode_sla_all,
                    "episode_sla_margin_by_slice": episode_margin_by_slice,
                    "rollup": True,
                },
                extra_limitations,
                run_id, episode_idx, -1, float(episode_idx), mode,
            )
        )

    rho_vals = [r for r in episode_rho_means if r is not None]
    urllc_blocks = [d.get("urllc", 0) for d in episode_block_by_slice]

    sla_compliance_by_slice = {
        slice_id: float(np.mean([e.get(slice_id, 1.0) for e in episode_sla_compliance_by_slice]))
        for slice_id in cfg.slice_by_id
    } if episode_sla_compliance_by_slice else {}
    sla_compliance_all_slices = (
        float(np.mean(episode_sla_compliance_all_slices)) if episode_sla_compliance_all_slices else 1.0
    )
    sla_margin_by_slice = {
        slice_id: float(np.mean([e.get(slice_id, 1.0) for e in episode_sla_margin_by_slice]))
        for slice_id in cfg.slice_by_id
    } if episode_sla_margin_by_slice else {}

    mean_mos_by_slice = {
        slice_id: float(np.mean(values)) for slice_id, values in all_step_mos_by_slice.items()
    }
    mean_cost = float(np.mean(all_step_cost)) if all_step_cost else 0.0
    mean_sla_viol = float(np.mean(all_step_sla_viol)) if all_step_sla_viol else 0.0

    summary = RunSummary(
        run_id=run_id,
        algorithm=algorithm,
        mode=mode,
        seed=seed,
        n_episodes=n_episodes,
        mean_reward_per_step=float(np.mean(episode_mean_rewards)) if episode_mean_rewards else 0.0,
        mean_blocks_per_episode=float(np.mean(episode_block_totals)) if episode_block_totals else 0.0,
        mean_urllc_blocks_per_episode=float(np.mean(urllc_blocks)) if urllc_blocks else None,
        blocks_per_episode=episode_block_totals,
        blocks_per_episode_by_slice=episode_block_by_slice,
        mean_rho=float(np.mean(rho_vals)) if rho_vals else None,
        omega_path=str(omega.path),
        sla_compliance_by_slice=sla_compliance_by_slice,
        sla_compliance_all_slices=sla_compliance_all_slices,
        sla_compliance_by_slice_per_episode=episode_sla_compliance_by_slice,
        sla_margin_by_slice=sla_margin_by_slice,
        sla_margin_by_slice_per_episode=episode_sla_margin_by_slice,
        mean_mos_by_slice=mean_mos_by_slice,
        mean_cost=mean_cost,
        mean_sla_viol=mean_sla_viol,
    )
    if mode == "live_testbed":
        summary.drift_flag, summary.drift_reason = flag_drift(summary)
    return summary


def flag_drift(
    summary: RunSummary,
    urllc_block_rate_threshold: float = 1.0,
    rho_band: Tuple[float, float] = (0.45, 0.70),
) -> Tuple[bool, str]:
    """Structured drift flag for live runs: block rate >2x the acceptance
    bar, or rho outside a wide margin around the [0.55,0.60] target."""
    reasons: List[str] = []
    if summary.mean_urllc_blocks_per_episode is not None and (
        summary.mean_urllc_blocks_per_episode > 2 * urllc_block_rate_threshold
    ):
        reasons.append(
            f"mean URLLC block rate {summary.mean_urllc_blocks_per_episode:.2f}/episode "
            f"exceeds 2x the {urllc_block_rate_threshold}/episode acceptance bar"
        )
    if summary.mean_rho is not None and not (rho_band[0] <= summary.mean_rho <= rho_band[1]):
        reasons.append(f"mean rho {summary.mean_rho:.3f} outside drift band {rho_band}")
    return (len(reasons) > 0, "; ".join(reasons))


def run_mc(
    cfg: SacLbExperimentConfig,
    algorithm: str,
    kpm_source_factory: Callable[[int], KpmSource],
    n_reps: int,
    episodes_per_rep: int,
    base_seed: int,
    mode: str,
    training: bool,
    results_dir: str,
    batch_size: int = 16,
    policy_factory: Optional[Callable[[int], Any]] = None,
    extra_limitations: Optional[List[str]] = None,
    reward_mode: str = "sla",
) -> List[RunSummary]:
    import os

    summaries: List[RunSummary] = []
    for rep in range(n_reps):
        seed = base_seed + rep
        run_id = f"{algorithm}_{mode}_seed{seed}_rep{rep}"
        kpm_source = kpm_source_factory(seed)
        env = RANEnv(cfg, kpm_source, seed=seed, reward_mode=reward_mode)
        policy = policy_factory(seed) if policy_factory is not None else build_policy(algorithm, cfg)

        rep_dir = os.path.join(results_dir, algorithm, mode, f"rep_{rep}")
        omega_path = os.path.join(rep_dir, "omega_log.jsonl")
        try:
            with OmegaLogger(omega_path) as omega:
                summary = run_single(
                    env, policy, algorithm, omega, episodes_per_rep, seed, run_id, mode, training, cfg,
                    batch_size=batch_size, extra_limitations=extra_limitations,
                )
        finally:
            # Each rep binds fresh sockets for live mode (LiveKpmSource) --
            # without closing here, a second rep in the same process fails
            # to re-bind the xApp listen port.
            env.close()

        if training and hasattr(policy, "save_checkpoint"):
            ckpt_path = os.path.join(rep_dir, "checkpoint.pt")
            policy.save_checkpoint(ckpt_path)
            summary.checkpoint_path = ckpt_path

        summaries.append(summary)
    return summaries
