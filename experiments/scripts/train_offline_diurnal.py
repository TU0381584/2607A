#!/usr/bin/env python3
"""Trains one (algorithm, reward_mode, seed) arm under a condensed
"one simulated week" of peak/off-peak demand (DiurnalCongestedKpmSource)
instead of CongestedRandomKpmSource's per-episode uniform-random
congestion -- so the resulting per-episode reward/compliance curve shows
whether the policy improves specifically at handling RECURRING peak-hour
contention over training, not just contention in general.

Adapted directly from train_offline_congested.py (same shared-pool
contention mechanism, same contention-aware reward shaping, same
mc_runner._select_actions/_store_and_train reuse) -- the only change is
the KPM source and the addition of a per-episode metrics JSONL for the
learning-curve plot this experiment is actually for.

Usage:
    python3 experiments/scripts/train_offline_diurnal.py \
        --algorithm dqn --reward-mode qoe --seed 256 --episodes 700 \
        --results-dir experiments/results/offline_diurnal
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from diurnal_kpm_source import DiurnalCongestedKpmSource  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.mc_runner import (  # noqa: E402
    NO_LEARNING_ALGORITHMS, _select_actions, _store_and_train,
    build_policy, set_seeds,
)
from qoe_oran_framework.policies.rainbow_admission import PrioritizedReplayBuffer  # noqa: E402
from oranslice_drl.drl_training import ReplayBuffer  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_congested_v1.yaml"


def contention_shaped_reward(base_reward: float, contention_ratio: float, urllc_compliant: bool, bonus_scale: float) -> float:
    if urllc_compliant:
        return base_reward
    return base_reward - bonus_scale * contention_ratio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow"])
    ap.add_argument("--reward-mode", choices=["sla", "qoe"], default="sla")
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=700)
    ap.add_argument("--episodes-per-week", type=int, default=700)
    ap.add_argument("--mult-lo", type=float, default=0.5)
    ap.add_argument("--mult-hi", type=float, default=1.8)
    ap.add_argument("--backlog-capacity", type=float, default=15.0)
    ap.add_argument("--shared-pool-prb", type=float, default=8.0)
    ap.add_argument("--contention-bonus-scale", type=float, default=15.0)
    ap.add_argument("--a2c-entropy-coef", type=float, default=0.01)
    ap.add_argument("--results-dir", default="experiments/results/offline_diurnal")
    args = ap.parse_args()

    set_seeds(args.seed)
    cfg = load_saclb_config(CONFIG_PATH)
    nominal_ratio = {s: spec.nominal_ratio for s, spec in cfg.slice_by_id.items()}
    sd_for_slice = {s: spec.sd for s, spec in cfg.slice_by_id.items()}
    noise_rng = np.random.RandomState(args.seed + 1)

    kpm = DiurnalCongestedKpmSource(
        seed=args.seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        nominal_ratio=nominal_ratio, episode_rng=noise_rng, sd_for_slice=sd_for_slice,
        backlog_capacity=args.backlog_capacity, shared_pool_prb=args.shared_pool_prb,
        episodes_per_week=args.episodes_per_week, mult_range=(args.mult_lo, args.mult_hi),
        noise_rng=noise_rng,
    )
    env = RANEnv(cfg, kpm, seed=args.seed, reward_mode=args.reward_mode)
    policy_overrides = {"entropy_coef": args.a2c_entropy_coef} if args.algorithm == "a2c" else {}
    policy = build_policy(args.algorithm, cfg, **policy_overrides)

    replay_buffer = None
    if args.algorithm == "dqn":
        replay_buffer = ReplayBuffer(capacity=10000)
    elif args.algorithm == "rainbow":
        replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=policy.per_alpha)

    out_dir = Path(args.results_dir) / args.reward_mode / f"seed{args.seed}" / args.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "episode_metrics.jsonl"

    t0 = time.time()
    with metrics_path.open("w", encoding="utf-8") as metrics_fh:
        for ep in range(1, args.episodes + 1):
            kpm.new_episode_congestion()
            diurnal_info = dict(kpm.last_diurnal_info)
            obs = env.reset()
            done = False
            rewards = []
            compliant_steps = {s: 0 for s in cfg.slice_by_id}
            total_steps = 0
            while not done:
                pending = env.pending_requests()
                cluster_state = env.last_cluster_state
                contention_ratio = kpm.last_contention_ratio
                actions, request_states = _select_actions(policy, args.algorithm, pending, obs, cluster_state, cfg, training=True)
                result = env.step(actions)
                next_obs = result.obs
                done = result.done
                per_slice_compliant = result.info["reward_breakdown"].get("per_slice_compliant", {})
                urllc_compliant = per_slice_compliant.get("urllc", True)
                shaped_reward = contention_shaped_reward(result.reward, contention_ratio, urllc_compliant, args.contention_bonus_scale)
                if args.algorithm not in NO_LEARNING_ALGORITHMS:
                    _store_and_train(
                        args.algorithm, policy, replay_buffer, pending, obs, next_obs, actions,
                        request_states, shaped_reward, done, cfg, batch_size=16, warmup_transitions=32,
                    )
                obs = next_obs
                rewards.append(shaped_reward)
                total_steps += 1
                for s in cfg.slice_by_id:
                    if per_slice_compliant.get(s, True):
                        compliant_steps[s] += 1

            # mc_runner.run_single's own episode-boundary hook (run_single line
            # ~380-381) -- REQUIRED for DQNAdmissionPolicy: that class freezes
            # oranslice_drl.DQNPolicy's per-train_step epsilon decay
            # (epsilon_decay=1.0 passed to the base class deliberately, see
            # dqn_admission.py's __init__ comment) and instead decays epsilon
            # ONCE PER EPISODE inside on_episode_end(), which nothing calls
            # unless the caller does so explicitly. train_offline_congested.py
            # (this script's basis, and the source of the paper's Section VI
            # congested checkpoints) has the SAME omission -- confirmed via
            # both scripts' saved checkpoints showing epsilon=1.0 after
            # thousands of train_step calls. Without this call, DQN's
            # training-time action selection is pure uniform-random for the
            # entire run (train_step still runs real gradient updates on the
            # randomly-collected, reward-labeled transitions -- valid
            # off-policy learning -- but never transitions to exploiting its
            # own Q-values, so there is no exploration-to-exploitation
            # learning curve to show).
            if args.algorithm in ("dqn", "rainbow") and hasattr(policy, "on_episode_end"):
                policy.on_episode_end()

            compliance = {s: compliant_steps[s] / max(1, total_steps) for s in cfg.slice_by_id}
            row = {
                "episode": ep, "mean_reward": sum(rewards) / max(1, len(rewards)),
                "compliance": compliance, "epsilon": getattr(policy, "epsilon", None), **diurnal_info,
            }
            metrics_fh.write(json.dumps(row) + "\n")
            if ep % 25 == 0 or ep == 1:
                metrics_fh.flush()
                print(f"[train_diurnal] {args.algorithm}/{args.reward_mode} seed={args.seed} "
                      f"ep={ep}/{args.episodes} day={diurnal_info['day_of_week']} "
                      f"weekend={diurnal_info['is_weekend']} shape={diurnal_info['shape']:.2f} "
                      f"mean_reward={row['mean_reward']:.4f} compliance={ {k: round(v,2) for k,v in compliance.items()} } "
                      f"epsilon={row['epsilon']} elapsed={time.time()-t0:.0f}s", file=sys.stderr)
    env.close()

    ckpt_path = out_dir / "checkpoint.pt"
    policy.save_checkpoint(str(ckpt_path))
    print(f"[train_diurnal] {args.algorithm}/{args.reward_mode} seed={args.seed}: "
          f"wrote {ckpt_path}, metrics={metrics_path}, total_time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
