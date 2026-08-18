#!/usr/bin/env python3
"""Direct Q-value probe on the M6 pilot's single_agent_dqn N=19 eval
checkpoint, mirroring the exact diagnostic discipline
docs/PAPER5_M2_gat_ctde.md used for the original N=3 collapse (direct
Q(accept) vs Q(reject) inspection on a real, in-episode congested
state -- not an aggregate block-count proxy, which the M2 doc's own
history shows can be misleading on its own). Confirms or denies
whether the eval-time zero-block reading is a genuine learned
always-accept collapse (Q(accept) > Q(reject) for real congested
states) versus some other artifact.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv, encode_full_request_state, request_state_dim  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CFG_PATH = "/home/kmanojp/oranslice_rig/framework/qoe_oran_framework/configs/saclb_offline_dqn_n19.yaml"
CKPT = "/home/kmanojp/oranslice_rig/experiments/results/m6_pilot/n19_hex/single_agent_dqn/seed900/train/dqn/offline_train/rep_0/checkpoint.pt"
BACKLOG_CAPACITY = 2000.0
EVAL_SEED = 5000 + 900


def main():
    cfg = load_saclb_config(CFG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    kpm = ClosedLoopKpmSource(
        seed=EVAL_SEED, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
        B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
        backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
    )
    env = RANEnv(cfg, kpm, seed=EVAL_SEED, reward_mode="sla")

    policy = build_policy("dqn", cfg)
    policy.load_checkpoint(CKPT)
    print(f"[probe] loaded checkpoint {CKPT}")
    print(f"[probe] request_state_dim={request_state_dim(cfg)}, epsilon={policy.epsilon}")

    env.reset()
    gaps_by_slice = {s: [] for s in cfg.slice_by_id}
    n_requests_seen = 0
    for step in range(1, 61):
        pending = env.pending_requests()
        obs = env.encode_state() if hasattr(env, "encode_state") else None
        cluster_state = env.last_cluster_state
        from qoe_oran_framework.env import encode_state
        obs = encode_state(cluster_state, cfg)
        actions = []
        for req in pending:
            state = encode_full_request_state(obs, req, cfg)
            state_t = torch.tensor(state, dtype=torch.float32, device=policy.device).unsqueeze(0)
            with torch.no_grad():
                q = policy.q_network(state_t).squeeze(0)  # [action_dim] or [n_branches, action_dim]
                q = q.squeeze()
            q_reject, q_accept = float(q[0]), float(q[1])
            gaps_by_slice[req.slice_id].append(q_accept - q_reject)
            n_requests_seen += 1
            chosen = 1 if q_accept > q_reject else 0
            actions.append(chosen)
        result = env.step(actions)
        if result.done:
            break

    print(f"\n[probe] {n_requests_seen} pending-request decisions inspected across {step} steps (greedy, epsilon=0 equivalent since eval uses training=False)")
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
