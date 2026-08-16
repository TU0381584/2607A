#!/usr/bin/env python3
"""M4: evaluation-time disruption-resilience harness (docs/PAPER5_M4_disruption.md).

Runs FROZEN M2/M3 checkpoints (never retrains) through disrupted eval
episodes -- gNB dropout, demand spikes, agent churn (see
qoe_oran_framework/marl/disruption.py's module docstring for the exact
mechanism and severity axis of each). Writes eval omega logs in the
same schema M2/M3 already use, so m2_correctness_metrics.per_seed_metrics
works unchanged on this arm's output too (m4_correctness_metrics.py
imports it, does not reimplement it).

Arms: gat_ctde, independent_dqn, single_agent_dqn (M2's three, checkpoints
at experiments/results/m2_campaign/<arm>/seed<seed>/train/...) and the
federated no-DP arm (experiments/results/m3_campaign/fl_gat_ctde_sigma0.0/
seed<seed>/train/checkpoint.pt, M3's own clean FL-vs-privacy baseline --
no DP sweep here, keeping disruption and privacy cost separable).

single_agent_dqn runs through a small standalone loop below, NOT
qoe_oran_framework.mc_runner.run_single/run_mc (frozen, not modified) --
mirrors marl_training.py's own precedent of importing mc_runner's private
per-request helpers (encode_full_request_state, _make_omega_tuple)
rather than duplicating their logic.

Usage (from repo root, cwd=framework/ required, matching every other
M2/M3 script's own documented convention):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m4_run_experiment.py \
        --arm gat_ctde --seed 900 --kind dropout --severity 2
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv, encode_full_request_state  # noqa: E402
from qoe_oran_framework.mc_runner import _make_omega_tuple, build_policy  # noqa: E402,SLF001 -- reused, not modified
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.marl.ctde_policy import GatCtdeMarlPolicy  # noqa: E402
from qoe_oran_framework.marl.disruption import (  # noqa: E402
    DisruptionSpec, corrupt_flat_obs, force_reject_actions_single_agent, spike_multiplier_for_step,
)
from qoe_oran_framework.marl.fl_ctde_policy import FederatedGatPolicy  # noqa: E402
from qoe_oran_framework.marl.independent_dqn_ablation import IndependentPerGnbDqnPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml"
BACKLOG_CAPACITY = 2000.0
ACTION_DIM = 2
EVAL_SEED_OFFSET = 5000  # matches M2/M3's own convention
CHURN_FRESH_POLICY_SEED_OFFSET = 800000  # disjoint from every other seed use in this project
EVAL_EPISODES = 50

M2_CKPT = {
    "gat_ctde": f"{REPO_ROOT}/experiments/results/m2_campaign/gat_ctde/seed{{seed}}/train/checkpoint.pt",
    "independent_dqn": f"{REPO_ROOT}/experiments/results/m2_campaign/independent_dqn/seed{{seed}}/train/checkpoint.pt",
    "single_agent_dqn": f"{REPO_ROOT}/experiments/results/m2_campaign/single_agent_dqn/seed{{seed}}/train/dqn/offline_train/rep_0/checkpoint.pt",
}
M3_CKPT = f"{REPO_ROOT}/experiments/results/m3_campaign/fl_gat_ctde_sigma0.0/seed{{seed}}/train/checkpoint.pt"

# Severity axis, matching the approved M4 plan: dropout/churn sweep window
# DURATION (10/30/60% of a 60-step episode); spike sweeps arrival-rate
# MAGNITUDE at a fixed 30%-of-episode window.
DROPOUT_CHURN_DURATION_FRAC = {1: 0.10, 2: 0.30, 3: 0.60}
SPIKE_MULTIPLIER = {1: 2.0, 2: 4.0, 3: 8.0}
SPIKE_DURATION_FRAC = 0.30

MULTI_AGENT_ARMS = ("gat_ctde", "independent_dqn", "fl_gat_ctde_sigma0.0")


def make_kpm_source_factory(cfg, sd_for_slice):
    def factory(seed):
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )
    return factory


def build_disruption_template(kind: str, severity: int, episode_length: int) -> DisruptionSpec:
    if kind in ("dropout", "churn"):
        duration = max(1, round(DROPOUT_CHURN_DURATION_FRAC[severity] * episode_length))
        return DisruptionSpec(kind=kind, target_agent_idx=-1, start_step=-1, duration_steps=duration,
                               severity_label=f"{kind}_sev{severity}")
    if kind == "spike":
        duration = max(1, round(SPIKE_DURATION_FRAC * episode_length))
        return DisruptionSpec(kind="spike", target_agent_idx=-1, start_step=-1, duration_steps=duration,
                               severity_param=SPIKE_MULTIPLIER[severity], severity_label=f"spike_sev{severity}")
    raise ValueError(f"unknown disruption kind {kind!r}")


def load_frozen_policy(arm: str, seed: int, cfg, n_agents: int, node_dim: int, ctx_dim: int, adj):
    """Constructs the class matching `arm` and loads its already-trained
    checkpoint. Never trains. Raises loudly (strict=True internally, or an
    unhandled KeyError/RuntimeError) on any architecture mismatch -- the
    same discipline m2_run_experiment.py's resume logic established."""
    if arm == "gat_ctde":
        policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
        policy.load_checkpoint(M2_CKPT["gat_ctde"].format(seed=seed))
    elif arm == "independent_dqn":
        policy = IndependentPerGnbDqnPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM)
        policy.load_checkpoint(M2_CKPT["independent_dqn"].format(seed=seed))
    elif arm == "single_agent_dqn":
        policy = build_policy("dqn", cfg)
        policy.load_checkpoint(M2_CKPT["single_agent_dqn"].format(seed=seed))
    elif arm == "fl_gat_ctde_sigma0.0":
        policy = FederatedGatPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
        policy.load_checkpoint(M3_CKPT.format(seed=seed))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return policy


def run_marl_arm_condition(arm: str, seed: int, kind: str, severity: int, out_dir: str) -> dict:
    """gat_ctde / independent_dqn / fl_gat_ctde_sigma0.0: all three already
    speak run_episodes_marl's select_actions(node_features, requests, training)
    interface, so this reuses that loop wholesale via its disruption= hook."""
    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents)
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)

    policy = load_frozen_policy(arm, seed, cfg, n_agents, node_dim, ctx_dim, adj)

    fresh_policy = None
    if kind == "churn":
        # A fresh, NEVER-TRAINED instance of the same class -- reproducible
        # per (seed, arm) via a dedicated seed offset, disjoint from every
        # other seed use in this project (training/eval/EVAL_SEED_OFFSET).
        torch.manual_seed(CHURN_FRESH_POLICY_SEED_OFFSET + seed)
        if arm == "gat_ctde":
            fresh_policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
        elif arm == "independent_dqn":
            fresh_policy = IndependentPerGnbDqnPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM)
        elif arm == "fl_gat_ctde_sigma0.0":
            fresh_policy = FederatedGatPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)

    spec = build_disruption_template(kind, severity, cfg.episode.steps_per_episode)

    eval_seed = EVAL_SEED_OFFSET + seed
    env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
    eval_dir = Path(out_dir) / arm / spec.severity_label / f"seed{seed}" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "omega_log.jsonl"
    eval_path.unlink(missing_ok=True)  # never append onto a stale run -- see docs/PAPER5_M2_gat_ctde.md section 14
    with OmegaLogger(str(eval_path)) as omega:
        summary = run_episodes_marl(env, policy, arm, omega, EVAL_EPISODES, eval_seed,
                                     f"m4_{arm}_{spec.severity_label}_seed{seed}", "offline_eval", False, cfg,
                                     disruption=spec, disruption_fresh_policy=fresh_policy)
    env.close()
    return summary


def run_single_agent_condition(seed: int, kind: str, severity: int, out_dir: str) -> dict:
    """single_agent_dqn: no separable per-gNB agent (one shared policy over
    the flattened joint state), so churn does not apply here (see
    docs/PAPER5_M4_disruption.md's scope note) -- only dropout/spike."""
    if kind == "churn":
        raise ValueError("single_agent_dqn has no separable agent to churn -- see module docstring")

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)
    n_slices = len(cfg.slices)

    policy = build_policy("dqn", cfg)
    policy.load_checkpoint(M2_CKPT["single_agent_dqn"].format(seed=seed))

    spec = build_disruption_template(kind, severity, cfg.episode.steps_per_episode)
    base_arrivals = cfg.arrivals.synthetic_arrivals_per_step

    eval_seed = EVAL_SEED_OFFSET + seed
    env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
    eval_dir = Path(out_dir) / "single_agent_dqn" / spec.severity_label / f"seed{seed}" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "omega_log.jsonl"
    eval_path.unlink(missing_ok=True)

    import numpy as np
    disruption_rng = np.random.RandomState(eval_seed)
    n_agents = len(cfg.gnb_ids)
    episode_sla_all = []

    with OmegaLogger(str(eval_path)) as omega:
        try:
            for episode_idx in range(1, EVAL_EPISODES + 1):
                ep_spec = spec.randomized_for_episode(n_agents, cfg.episode.steps_per_episode, disruption_rng)
                target_gnb_id = cfg.gnb_ids[ep_spec.target_agent_idx]
                cfg.arrivals.synthetic_arrivals_per_step = spike_multiplier_for_step(ep_spec, 1, base_arrivals)

                obs = env.reset()
                block_by_slice = {}
                compliant_steps_by_slice = {s: 0 for s in cfg.slice_by_id}
                all_slices_compliant_steps = 0
                sla_margin_sum_by_slice = {s: 0.0 for s in cfg.slice_by_id}
                step_idx = 0

                while True:
                    step_idx += 1
                    pending = env.pending_requests()
                    obs_for_policy = corrupt_flat_obs(obs, cfg.gnb_ids, n_slices, ep_spec, step_idx)

                    actions = []
                    for request in pending:
                        req_state = encode_full_request_state(obs_for_policy, request, cfg)
                        action, _info = policy.select_action(req_state, training=False)
                        actions.append(int(action))
                    pending_gnb_ids = [r.gnb_id for r in pending]
                    actions = force_reject_actions_single_agent(actions, pending_gnb_ids, target_gnb_id,
                                                                  ep_spec, step_idx)
                    cfg.arrivals.synthetic_arrivals_per_step = spike_multiplier_for_step(
                        ep_spec, step_idx + 1, base_arrivals)

                    result = env.step(actions)
                    obs = result.obs

                    for block in result.info["primary_blocks"]:
                        block_by_slice[block["slice_id"]] = block_by_slice.get(block["slice_id"], 0) + 1
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
                        "seed": seed, "reward": result.reward,
                        "primary_block_count": result.info["primary_block_count"],
                        "secondary_block_count": result.info["secondary_block_count"],
                        "accepted_counts": result.info["accepted_counts"],
                        "fairness_ratio": result.info["fairness_ratio"], "n_pending": len(pending),
                        "ceilings": result.info.get("ceilings"),
                        "per_slice_compliant": per_slice_compliant, "per_slice_sla_margin": per_slice_sla_margin,
                    }
                    omega.log(_make_omega_tuple(
                        "single_agent_dqn", cfg, evidence, list(result.info["limitations"]),
                        f"m4_single_agent_dqn_{spec.severity_label}_seed{seed}", episode_idx, step_idx,
                        float(result.info["global_step"]), "offline_eval",
                    ))
                    if result.done:
                        break

                episode_steps = step_idx
                episode_sla_by_slice = {
                    s: (compliant_steps_by_slice.get(s, 0) / episode_steps if episode_steps else 1.0)
                    for s in cfg.slice_by_id
                }
                episode_sla_all_val = all_slices_compliant_steps / episode_steps if episode_steps else 1.0
                episode_sla_all.append(episode_sla_all_val)
                episode_margin_by_slice = {
                    s: (sla_margin_sum_by_slice.get(s, 0.0) / episode_steps if episode_steps else 1.0)
                    for s in cfg.slice_by_id
                }
                omega.log(_make_omega_tuple(
                    "single_agent_dqn", cfg,
                    {
                        "seed": seed, "episode_block_total": sum(block_by_slice.values()),
                        "episode_block_by_slice": block_by_slice, "episode_mean_rho": None,
                        "episode_sla_compliance_by_slice": episode_sla_by_slice,
                        "episode_sla_compliance_all_slices": episode_sla_all_val,
                        "episode_sla_margin_by_slice": episode_margin_by_slice, "rollup": True,
                    },
                    [], f"m4_single_agent_dqn_{spec.severity_label}_seed{seed}", episode_idx, -1,
                    float(episode_idx), "offline_eval",
                ))
        finally:
            cfg.arrivals.synthetic_arrivals_per_step = base_arrivals
    env.close()

    return {
        "sla_compliance_all_slices": float(sum(episode_sla_all) / len(episode_sla_all)) if episode_sla_all else 1.0,
        "n_episodes": EVAL_EPISODES,
    }


def run_condition(arm: str, seed: int, kind: str, severity: int, out_dir: str) -> dict:
    if arm == "single_agent_dqn":
        return run_single_agent_condition(seed, kind, severity, out_dir)
    return run_marl_arm_condition(arm, seed, kind, severity, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                     choices=["gat_ctde", "independent_dqn", "single_agent_dqn", "fl_gat_ctde_sigma0.0"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", required=True, choices=["dropout", "spike", "churn"])
    ap.add_argument("--severity", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--out-dir", default=f"{REPO_ROOT}/experiments/results/m4_campaign")
    args = ap.parse_args()

    summary = run_condition(args.arm, args.seed, args.kind, args.severity, args.out_dir)
    print(f"[m4] arm={args.arm} seed={args.seed} kind={args.kind} severity={args.severity}: "
          f"sla_compliance_all_slices={summary['sla_compliance_all_slices']:.4f}")


if __name__ == "__main__":
    main()
