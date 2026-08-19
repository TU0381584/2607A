#!/usr/bin/env python3
"""One-off diagnostic (not a campaign script): the mu-sweep and the
clipnorm=100 re-run both found FedProx bit-identical to FedAvg at the
eval-decision level, at mu up to 1.0, with or without a binding
grad-norm clip -- ruling out clipping as the mechanism. Before guessing
again, measure directly: what is the ACTUAL gradient norm (TD loss
alone vs. TD+proximal at mu=1.0), and how does the proximal term's own
gradient norm compare to the TD loss's, at a real training step on a
real checkpoint mid-training? If the proximal term's gradient is simply
tiny relative to the TD loss's, no clip setting would ever have made a
visible difference -- the earlier clip-norm hypothesis was a real,
code-grounded mechanism that just was not the actual explanation here.

Usage (from repo root, cwd=framework/ required):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m7_gradnorm_probe.py
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.marl.fl_ctde_policy import FederatedGatPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml"
BACKLOG_CAPACITY = 2000.0
ACTION_DIM = 2
SEED = 8801


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
    # fedprox_mu=0 here -- we build the loss/grad manually below so we can
    # inspect the TD-only and TD+proximal gradients separately from the
    # SAME weights, rather than comparing across two different training runs.
    policy = FederatedGatPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj,
                                 aggregator="fedprox", fedprox_mu=0.0,
                                 local_steps_per_round=50, dp_clip_norm=1.0,
                                 dp_noise_multiplier=0.0, dp_seed=SEED)

    # Warm up with 20 real training episodes so weights are mid-training,
    # not at their random initial values (initial-weight gradients would
    # not be representative of the actual campaign's later dynamics).
    env = RANEnv(cfg, kpm_factory(SEED), seed=SEED, reward_mode="sla")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
        with OmegaLogger(tf.name) as omega:
            run_episodes_marl(env, policy, "probe", omega, 20, SEED, "probe_train", "offline_train", True, cfg)
    env.close()

    # Snapshot pre-step weights as the FedProx "global" reference point,
    # then perturb client 0's weights slightly (simulating post-broadcast
    # local drift a few local steps into a round -- the actual condition
    # under which the proximal term is ever nonzero).
    policy.global_snapshot = {name: p.detach().clone() for name, p in policy.clients[0].named_parameters()}
    with torch.no_grad():
        for p in policy.clients[0].parameters():
            p.add_(torch.randn_like(p) * 0.05)  # a real amount of local drift, not zero

    # Pull one real minibatch the same way train_step does, by running one
    # more training episode and capturing what train_step sees -- simplest
    # correct way to get real, in-distribution (nf, ctx, action, target)
    # tensors without duplicating marl_training's buffer-sampling logic.
    captured = {}
    orig_train_step = policy.train_step

    def capturing_train_step(batch):
        captured["batch"] = batch
        return orig_train_step(batch)
    policy.train_step = capturing_train_step
    env2 = RANEnv(cfg, kpm_factory(SEED + 1), seed=SEED + 1, reward_mode="sla")
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
        with OmegaLogger(tf.name) as omega:
            run_episodes_marl(env2, policy, "probe", omega, 1, SEED + 1, "probe_train2", "offline_train", True, cfg)
    env2.close()
    batch = captured["batch"]

    def forward_loss(agent_idx, mu):
        import numpy as np
        nf = torch.tensor(np.stack(batch["node_features"]), dtype=torch.float32)
        next_nf = torch.tensor(np.stack(batch["next_node_features"]), dtype=torch.float32)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32)
        dones = torch.tensor(batch["dones"], dtype=torch.float32)
        b_idxs = [b for b in range(len(batch["rewards"]))
                  if agent_idx in batch["agent_request_agent_idx"][b]]
        if not b_idxs:
            return None
        b_t = torch.tensor(b_idxs, dtype=torch.long)
        embeds = policy.clients[agent_idx].embed(nf[b_t], policy.adjacency)[:, agent_idx]
        with torch.no_grad():
            next_embeds = policy.targets[agent_idx].embed(next_nf[b_t], policy.adjacency)[:, agent_idx]
        ctx_list, act_list = [], []
        for b in b_idxs:
            idxs = [i for i, a in enumerate(batch["agent_request_agent_idx"][b]) if a == agent_idx]
            ctx_list.append(batch["agent_request_context"][b][idxs[0]])
            act_list.append(batch["agent_request_action"][b][idxs[0]])
        ctx_t = torch.tensor(np.asarray(ctx_list), dtype=torch.float32)
        acts_t = torch.tensor(act_list, dtype=torch.long)
        q_vals = policy.clients[agent_idx].agent_q_values(embeds, ctx_t)
        chosen = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = policy.targets[agent_idx].agent_q_values(next_embeds, ctx_t)
            target = rewards[b_t] + policy.gamma * next_q.max(dim=-1)[0] * (1 - dones[b_t])
        td_loss = F.smooth_l1_loss(chosen, target)
        prox = torch.zeros(())
        for name, p in policy.clients[agent_idx].named_parameters():
            prox = prox + (p - policy.global_snapshot[name]).pow(2).sum()
        return td_loss, prox

    def backward_norm(agent_idx, loss):
        policy.optimizers[agent_idx].zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.clients[agent_idx].parameters(), max_norm=1e9).item()
        policy.optimizers[agent_idx].zero_grad()
        return norm

    def grad_norm_for(agent_idx, mu):
        out = forward_loss(agent_idx, mu)
        if out is None:
            return None, None
        td_loss, prox = out
        td_norm = backward_norm(agent_idx, td_loss)
        out2 = forward_loss(agent_idx, mu)
        _, prox2 = out2
        prox_norm = backward_norm(agent_idx, 0.5 * mu * prox2)
        out3 = forward_loss(agent_idx, mu)
        td_loss3, prox3 = out3
        combined_norm = backward_norm(agent_idx, td_loss3 + 0.5 * mu * prox3)
        return td_norm, (prox_norm, combined_norm)

    for mu in [0.01, 0.1, 1.0, 10.0]:
        td_norm, rest = grad_norm_for(0, mu)
        if td_norm is None:
            print(f"mu={mu}: agent 0 had no requests in this batch, skipping")
            continue
        prox_norm, combined_norm = rest
        print(f"mu={mu:>5}: TD-only grad norm={td_norm:.4f}  "
              f"proximal-only grad norm={prox_norm:.4f}  combined grad norm={combined_norm:.4f}  "
              f"prox/TD ratio={prox_norm/td_norm:.4f}")


if __name__ == "__main__":
    main()
