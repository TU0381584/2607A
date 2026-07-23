#!/usr/bin/env python3
"""Trains one (algorithm, reward_mode, seed) arm under the congested,
dynamic, randomized offline scenario (saclb_offline_congested_v1.yaml +
CongestedRandomKpmSource) -- fixes the ceiling-headroom train/eval
mismatch found 2026-07-24 (see that config's own comment) and gives the
policy real congestion variability to learn from, not one fixed
oversubscription factor.

Congestion re-sampled fresh at the START of every episode (before
env.reset(), which itself calls kpm_source.poll() once) -- mirrors
mc_runner.run_single's own per-episode loop structure but adds the
resample call run_mc has no hook for, so this reimplements the training
loop directly (reusing mc_runner's _select_actions/_store_and_train,
not duplicating their logic).

Usage:
    python3 experiments/scripts/train_offline_congested.py \
        --algorithm dqn --reward-mode sla --seed 256 --episodes 300 \
        --results-dir experiments/results/offline_congested
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from shared_pool_kpm_source import SharedPoolCongestedKpmSource  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.mc_runner import (  # noqa: E402
    NO_LEARNING_ALGORITHMS, REPLAY_BASED_ALGORITHMS, _select_actions, _store_and_train,
    build_policy, set_seeds,
)
from qoe_oran_framework.policies.rainbow_admission import PrioritizedReplayBuffer  # noqa: E402
from oranslice_drl.drl_training import ReplayBuffer  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_congested_v1.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow"])
    ap.add_argument("--reward-mode", choices=["sla", "qoe"], default="sla")
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--congestion-lo", type=float, default=0.4)
    ap.add_argument("--congestion-hi", type=float, default=1.3)
    ap.add_argument("--backlog-capacity", type=float, default=15.0)
    ap.add_argument("--shared-pool-prb", type=float, default=8.0)
    ap.add_argument("--results-dir", default="experiments/results/offline_congested")
    args = ap.parse_args()

    set_seeds(args.seed)
    cfg = load_saclb_config(CONFIG_PATH)
    nominal_ratio = {s: spec.nominal_ratio for s, spec in cfg.slice_by_id.items()}
    sd_for_slice = {s: spec.sd for s, spec in cfg.slice_by_id.items()}
    episode_rng = np.random.RandomState(args.seed + 1)

    kpm = SharedPoolCongestedKpmSource(
        seed=args.seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        nominal_ratio=nominal_ratio, congestion_range=(args.congestion_lo, args.congestion_hi),
        episode_rng=episode_rng, sd_for_slice=sd_for_slice, backlog_capacity=args.backlog_capacity,
        shared_pool_prb=args.shared_pool_prb,
    )
    env = RANEnv(cfg, kpm, seed=args.seed, reward_mode=args.reward_mode)
    policy = build_policy(args.algorithm, cfg)

    replay_buffer = None
    if args.algorithm == "dqn":
        replay_buffer = ReplayBuffer(capacity=10000)
    elif args.algorithm == "rainbow":
        replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=policy.per_alpha)

    t0 = time.time()
    episode_rewards = []
    for ep in range(1, args.episodes + 1):
        mult = kpm.new_episode_congestion()
        obs = env.reset()
        done = False
        rewards = []
        while not done:
            pending = env.pending_requests()
            cluster_state = env.last_cluster_state
            actions, request_states = _select_actions(policy, args.algorithm, pending, obs, cluster_state, cfg, training=True)
            result = env.step(actions)
            next_obs = result.obs
            done = result.done
            if args.algorithm not in NO_LEARNING_ALGORITHMS:
                _store_and_train(
                    args.algorithm, policy, replay_buffer, pending, obs, next_obs, actions,
                    request_states, result.reward, done, cfg, batch_size=16, warmup_transitions=32,
                )
            obs = next_obs
            rewards.append(result.reward)
        episode_rewards.append(sum(rewards) / max(1, len(rewards)))
        if ep % 25 == 0 or ep == 1:
            recent = np.mean(episode_rewards[-25:])
            print(f"[train_congested] {args.algorithm}/{args.reward_mode} seed={args.seed} "
                  f"ep={ep}/{args.episodes} mean_reward(last25)={recent:.4f} "
                  f"congestion_mult(this_ep)={ {k: round(v,2) for k,v in mult.items()} } "
                  f"elapsed={time.time()-t0:.0f}s", file=sys.stderr)
    env.close()

    out_dir = Path(args.results_dir) / args.reward_mode / f"seed{args.seed}" / args.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    policy.save_checkpoint(str(ckpt_path))
    print(f"[train_congested] {args.algorithm}/{args.reward_mode} seed={args.seed}: "
          f"wrote {ckpt_path}, final mean_reward(last25)={np.mean(episode_rewards[-25:]):.4f}, "
          f"total_time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
