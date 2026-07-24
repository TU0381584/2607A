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


def contention_shaped_reward(base_reward: float, contention_ratio: float, urllc_compliant: bool, bonus_scale: float) -> float:
    """Extra penalty when URLLC is violated AND the shared pool is
    genuinely tight (contention_ratio near 1) -- zero when URLLC is fine,
    or when the pool has slack regardless of URLLC's state. Targets the
    diagnosed failure mode directly (a flat per-violation charge too
    small relative to eMBB's raw accept-volume reward under real
    contention), rather than a blanket reward-weight increase that would
    also fire when there's no actual scarcity to justify it."""
    if urllc_compliant:
        return base_reward
    return base_reward - bonus_scale * contention_ratio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow"])
    ap.add_argument("--reward-mode", choices=["sla", "qoe"], default="sla")
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--congestion-lo", type=float, default=0.4)
    ap.add_argument("--congestion-hi", type=float, default=1.3)
    ap.add_argument("--backlog-capacity", type=float, default=15.0)
    ap.add_argument("--shared-pool-prb", type=float, default=8.0)
    ap.add_argument("--contention-bonus-scale", type=float, default=15.0)
    ap.add_argument(
        "--a2c-explore-eps", type=float, default=0.0,
        help="A2C-only: probability of overriding the sampled action with a uniform-random one "
             "during training. Tried first (2026-07-24) as a fix for A2C collapsing to a fully "
             "deterministic degenerate policy (accept_frac exactly 1.0 or 0.0, every state) -- did "
             "NOT fix it (converged policy was essentially unchanged). CORRECTED diagnosis: "
             "A2CPolicy already has entropy regularization (train_step's -entropy_coef*entropy "
             "term), just a small fixed 0.01 coefficient that's negligible next to this script's "
             "contention-bonus-scale (~15) reward-shaping magnitude -- see --a2c-entropy-coef below, "
             "which is the actual fix. Left at 0.0 (disabled) by default now to avoid confounding "
             "two interventions; kept as an option, not removed.",
    )
    ap.add_argument(
        "--a2c-entropy-coef", type=float, default=0.01,
        help="Passed through to A2CPolicy's (now configurable, previously hardcoded-at-0.01) "
             "entropy_coef -- raise this for A2C runs under large reward-shaping magnitudes "
             "(see --contention-bonus-scale) so entropy regularization isn't swamped by the "
             "shaped reward's scale. No effect on dqn/rainbow.",
    )
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
    policy_overrides = {"entropy_coef": args.a2c_entropy_coef} if args.algorithm == "a2c" else {}
    policy = build_policy(args.algorithm, cfg, **policy_overrides)

    replay_buffer = None
    if args.algorithm == "dqn":
        replay_buffer = ReplayBuffer(capacity=10000)
    elif args.algorithm == "rainbow":
        replay_buffer = PrioritizedReplayBuffer(capacity=10000, alpha=policy.per_alpha)

    explore_rng = np.random.RandomState(args.seed + 2000)

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
            # Captured BEFORE env.step(): last_contention_ratio was set by
            # the PREVIOUS step's trailing poll(), i.e. it describes
            # exactly the cluster_state this upcoming step's reward is
            # computed from (env.step() computes reward from
            # self._last_cluster_state, THEN polls again for next_obs) --
            # this is the correctly-paired contention level for the
            # violation this step's reward_breakdown will report.
            contention_ratio = kpm.last_contention_ratio
            actions, request_states = _select_actions(policy, args.algorithm, pending, obs, cluster_state, cfg, training=True)
            if args.algorithm == "a2c" and args.a2c_explore_eps > 0:
                actions = [
                    int(explore_rng.randint(0, 2)) if explore_rng.rand() < args.a2c_explore_eps else a
                    for a in actions
                ]
            result = env.step(actions)
            next_obs = result.obs
            done = result.done
            urllc_compliant = result.info["reward_breakdown"].get("per_slice_compliant", {}).get("urllc", True)
            shaped_reward = contention_shaped_reward(result.reward, contention_ratio, urllc_compliant, args.contention_bonus_scale)
            if args.algorithm not in NO_LEARNING_ALGORITHMS:
                _store_and_train(
                    args.algorithm, policy, replay_buffer, pending, obs, next_obs, actions,
                    request_states, shaped_reward, done, cfg, batch_size=16, warmup_transitions=32,
                )
            obs = next_obs
            rewards.append(shaped_reward)
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
