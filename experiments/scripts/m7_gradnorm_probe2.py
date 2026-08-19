#!/usr/bin/env python3
"""Follow-up to m7_gradnorm_probe.py: that probe hand-injected an
artificial 0.05-std perturbation to make the proximal term's gradient
measurable, and found it comparable to or larger than the TD-loss
gradient at mu>=1.0 -- but fl_ctde_policy.py resets global_snapshot
every local_steps_per_round (default 50) train_step calls
(_aggregate_round), so REAL accumulated drift within one round may be
much smaller than that artificial injection. This measures the
proximal term's ACTUAL contribution during real training (no injected
drift at all) by monkeypatching _local_loss to log both terms' scalar
magnitudes on every real train_step call, at mu=1.0, for enough
episodes to span several real rounds.

Usage (from repo root, cwd=framework/ required):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m7_gradnorm_probe2.py
"""
import sys
import tempfile

import torch

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.marl.fl_ctde_policy import FederatedGatPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml"
BACKLOG_CAPACITY = 2000.0
ACTION_DIM = 2
SEED = 8801
MU = 1.0


def main() -> None:
    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents)

    def kpm_factory(seed):
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )

    torch.manual_seed(SEED)
    policy = FederatedGatPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj,
                                 aggregator="fedprox", fedprox_mu=MU,
                                 local_steps_per_round=50, dp_clip_norm=1.0,
                                 dp_noise_multiplier=0.0, dp_seed=SEED)

    log = []
    orig = policy._local_loss

    def logging_local_loss(agent_idx, chosen, target):
        td = torch.nn.functional.smooth_l1_loss(chosen, target)
        prox_term = torch.zeros(())
        for name, p in policy.clients[agent_idx].named_parameters():
            prox_term = prox_term + (p - policy.global_snapshot[name]).pow(2).sum()
        prox_scaled = 0.5 * policy.fedprox_mu * prox_term
        log.append((policy.round_count, agent_idx, float(td.detach()), float(prox_scaled.detach())))
        return orig(agent_idx, chosen, target)

    policy._local_loss = logging_local_loss

    env = RANEnv(cfg, kpm_factory(SEED), seed=SEED, reward_mode="sla")
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
        with OmegaLogger(tf.name) as omega:
            # 10 episodes: 60 steps/episode * 10 = 600 steps of real
            # env interaction, comfortably spanning several real
            # 50-local-step rounds regardless of how many train_step
            # calls happen per env step.
            run_episodes_marl(env, policy, "probe", omega, 10, SEED, "probe_train", "offline_train", True, cfg)
    env.close()

    print(f"[m7-probe2] mu={MU}, {len(log)} real train_step._local_loss calls logged, "
          f"final round_count={policy.round_count}")
    if not log:
        print("no calls logged -- nothing to report")
        return
    tds = [x[2] for x in log]
    proxes = [x[3] for x in log]
    ratios = [p / t if t > 0 else float("nan") for t, p in zip(tds, proxes)]
    import statistics as stats
    print(f"TD loss:      mean={stats.mean(tds):.4f} max={max(tds):.4f}")
    print(f"prox (scaled, 0.5*mu*sum): mean={stats.mean(proxes):.6f} max={max(proxes):.6f}")
    print(f"prox/TD loss ratio: mean={stats.mean(ratios):.6f} max={max(ratios):.6f}")
    # last few calls of the run = maximum accumulated real drift within a round
    print("last 10 (round, agent, td, prox):")
    for row in log[-10:]:
        print(f"  round={row[0]} agent={row[1]} td={row[2]:.4f} prox={row[3]:.6f}")


if __name__ == "__main__":
    main()
