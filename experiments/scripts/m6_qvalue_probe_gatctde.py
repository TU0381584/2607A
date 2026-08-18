#!/usr/bin/env python3
"""Same direct Q-value probe as m6_qvalue_probe.py, for the gat_ctde arm
-- confirms the N=19 contrast (single-agent DQN fully collapsed,
GAT-CTDE not) is a real difference in what each policy learned, not an
artifact of only having inspected one arm."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.marl.ctde_policy import GatCtdeMarlPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import extract_node_features, node_feature_dim, request_context_dim, requests_to_agent_contexts  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency, hex_grid_edges  # noqa: E402

CFG_PATH = "/home/kmanojp/oranslice_rig/framework/qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml"
CKPT = "/home/kmanojp/oranslice_rig/experiments/results/m6_pilot/n19_hex/gat_ctde/seed900/train/checkpoint.pt"
BACKLOG_CAPACITY = 2000.0
EVAL_SEED = 5000 + 900
ACTION_DIM = 2


def main():
    cfg = load_saclb_config(CFG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    kpm = ClosedLoopKpmSource(
        seed=EVAL_SEED, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
        B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
        backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
    )
    env = RANEnv(cfg, kpm, seed=EVAL_SEED, reward_mode="sla")

    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents, hex_grid_edges(n_agents))
    policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
    ckpt = torch.load(CKPT, map_location=policy.device, weights_only=True)
    policy.online.load_state_dict(ckpt["online"])
    print(f"[probe] loaded checkpoint {CKPT}")

    env.reset()
    gaps_by_slice = {s: [] for s in cfg.slice_by_id}
    n_requests_seen = 0
    for step in range(1, 61):
        pending = env.pending_requests()
        cluster_state = env.last_cluster_state
        node_features = extract_node_features(cluster_state, cfg)
        requests_ctx = requests_to_agent_contexts(pending, cfg)

        with torch.no_grad():
            nf = torch.tensor(node_features, dtype=torch.float32, device=policy.device).unsqueeze(0)
            embeds = policy.online.embed(nf, policy.adjacency).squeeze(0)

        actions = []
        for (req, (agent_idx, ctx)) in zip(pending, requests_ctx):
            with torch.no_grad():
                ctx_t = torch.tensor(ctx, dtype=torch.float32, device=policy.device).unsqueeze(0)
                q = policy.online.agent_q_values(embeds[agent_idx].unsqueeze(0), ctx_t).squeeze(0)
            q_reject, q_accept = float(q[0]), float(q[1])
            gaps_by_slice[req.slice_id].append(q_accept - q_reject)
            n_requests_seen += 1
            actions.append(1 if q_accept > q_reject else 0)
        result = env.step(actions)
        if result.done:
            break

    print(f"\n[probe] {n_requests_seen} pending-request decisions inspected across {step} steps (greedy, training=False)")
    for slice_id, gaps in gaps_by_slice.items():
        if not gaps:
            print(f"  {slice_id}: no requests seen this episode")
            continue
        g = np.array(gaps)
        print(f"  {slice_id}: n={len(g)}, Q(accept)-Q(reject) mean={g.mean():+.3f}, "
              f"min={g.min():+.3f}, max={g.max():+.3f}, frac_positive(=accept-preferred)={float((g>0).mean()):.3f}")
    env.close()


if __name__ == "__main__":
    main()
