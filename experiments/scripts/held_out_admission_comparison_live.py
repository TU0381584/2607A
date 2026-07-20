#!/usr/bin/env python3
"""Held-out comparison of all 7 admission-efficiency arms (accept_all,
reject_all, static_threshold, dqn_sla, a2c_sla, dqn_qoe, a2c_qoe) on seeds
NOT used anywhere else in this workstream:
  - 256/257/258: offline TRAINING seeds for the 4 learned arms.
  - 950: the seed static_threshold's parameters were TUNED on.
  - 960/961/962 (used here): fresh, unseen by anything above.

Learned arms are evaluated with frozen weights (select_action(training=False)
-- greedy, no epsilon exploration) via the SAME frozen mc_runner.run_mc/
run_single harness used for training, just training=False and a
policy_factory that loads each arm's checkpoint instead of building a
fresh one. static_threshold reuses run_mc too (algorithm="lb_only", the
frozen LbOnlyHeuristic, unmodified) with the parameters
tune_static_threshold.py already found (utilization_threshold=0.7,
capacity_margin=0.7). accept_all/reject_all stay as lightweight scripted
loops (no run_mc integration needed for a stateless fixed rule).

SLA-reward arms (dqn_sla, a2c_sla) and QoE-reward arms (dqn_qoe, a2c_qoe)
are NOT directly reward-comparable (different formulas/scales -- same
convention as fig1_training_convergence.py's 2x2 split) -- accept_all/
reject_all/static_threshold are therefore run under BOTH reward modes so
each group has its own fair baseline set.

Usage:
    python3 experiments/scripts/held_out_admission_comparison.py \
        --seeds 960 961 962 --episodes 20 \
        --out experiments/results/admission_efficiency/held_out_comparison.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

import numpy as np  # noqa: E402
from admission_efficiency_live_env import BACKLOG_CAPACITY, CONFIG_PATH, OVERSUB_OF_CAP  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402

SLICE_ORDER = ["embb", "urllc", "mmtc"]
TRAIN_ROOT = Path("/home/kmanojp/oranslice_rig/experiments/results/admission_efficiency_live_offline")
TUNED_STATIC_THRESHOLD = {"utilization_threshold": 0.7, "capacity_margin": 0.7}

LEARNED_ARMS = {
    "dqn_sla": ("dqn", "sla"), "a2c_sla": ("a2c", "sla"),
    "dqn_qoe": ("dqn", "qoe"), "a2c_qoe": ("a2c", "qoe"),
}


def kpm_source_factory(cfg, seed):
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
    mean_offered_ratio = {
        slice_id: min(0.98, OVERSUB_OF_CAP * spec.max_ratio_cap / 100.0)
        for slice_id, spec in cfg.slice_by_id.items()
    }
    return ClosedLoopKpmSource(
        seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        mean_offered_ratio=mean_offered_ratio, backlog_capacity=BACKLOG_CAPACITY,
        sd_for_slice=sd_for_slice,
    )


def scripted_policy_eval(cfg, decide_fn, seed, reward_mode, n_episodes):
    kpm = kpm_source_factory(cfg, seed)
    env = RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)
    compliant = {s: [] for s in SLICE_ORDER}
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        for _ in range(env.cfg.episode.steps_per_episode):
            pending = env.pending_requests()
            actions = decide_fn(pending, env.last_cluster_state)
            result = env.step(actions)
            rewards.append(result.reward)
            rb = result.info.get("reward_breakdown", {})
            for s, c in rb.get("per_slice_compliant", {}).items():
                compliant[s].append(bool(c))
    env.close()
    return compliant, rewards


def accept_all_decide(pending, cluster_state):
    return [1 for _ in pending]


def reject_all_decide(pending, cluster_state):
    return [0 for _ in pending]


def learned_arm_eval(cfg, algo, mode, seed, n_episodes, checkpoint_path):
    def policy_factory(_seed):
        policy = build_policy(algo, cfg)
        policy.load_checkpoint(str(checkpoint_path))
        return policy

    summaries = run_mc(
        cfg, algo, lambda s: kpm_source_factory(cfg, s), n_reps=1,
        episodes_per_rep=n_episodes, base_seed=seed, mode="held_out_eval",
        training=False, results_dir="/tmp/admission_eff_held_out_eval",
        policy_factory=policy_factory, reward_mode=mode,
    )
    s = summaries[0]
    return s.sla_compliance_by_slice, s.mean_reward_per_step


def static_threshold_eval(cfg, seed, reward_mode, n_episodes):
    summaries = run_mc(
        cfg, "lb_only", lambda s: kpm_source_factory(cfg, s), n_reps=1,
        episodes_per_rep=n_episodes, base_seed=seed, mode="held_out_eval",
        training=False, results_dir="/tmp/admission_eff_held_out_eval",
        policy_factory=lambda _s: build_policy("lb_only", cfg, **TUNED_STATIC_THRESHOLD),
        reward_mode=reward_mode,
    )
    s = summaries[0]
    return s.sla_compliance_by_slice, s.mean_reward_per_step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[960, 961, 962])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="experiments/results/admission_efficiency/held_out_comparison.md")
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)

    results = {"sla": {}, "qoe": {}}

    for mode in ["sla", "qoe"]:
        for name, decide_fn in [("accept_all", accept_all_decide), ("reject_all", reject_all_decide)]:
            compl_pooled = {s: [] for s in SLICE_ORDER}
            rewards_pooled = []
            for seed in args.seeds:
                compl, rewards = scripted_policy_eval(cfg, decide_fn, seed, mode, args.episodes)
                for s in SLICE_ORDER:
                    compl_pooled[s].extend(compl[s])
                rewards_pooled.extend(rewards)
            results[mode][name] = (
                {s: float(np.mean(compl_pooled[s])) for s in SLICE_ORDER},
                float(np.mean(rewards_pooled)),
            )

        per_seed_compl = {s: [] for s in SLICE_ORDER}
        per_seed_reward = []
        for seed in args.seeds:
            compl, reward = static_threshold_eval(cfg, seed, mode, args.episodes)
            for s in SLICE_ORDER:
                per_seed_compl[s].append(compl.get(s, float("nan")))
            per_seed_reward.append(reward)
        results[mode]["static_threshold"] = (
            {s: float(np.mean(per_seed_compl[s])) for s in SLICE_ORDER},
            float(np.mean(per_seed_reward)),
        )

    for arm, (algo, mode) in LEARNED_ARMS.items():
        # Use seed 256's checkpoint (arbitrary but fixed choice among the
        # 3 trained seeds -- documented, not cherry-picked post-hoc).
        ckpt = TRAIN_ROOT / mode / "seed256" / algo / "offline_closed_loop" / "rep_0" / "checkpoint.pt"
        per_seed_compl = {s: [] for s in SLICE_ORDER}
        per_seed_reward = []
        for seed in args.seeds:
            compl, reward = learned_arm_eval(cfg, algo, mode, seed, args.episodes, ckpt)
            for s in SLICE_ORDER:
                per_seed_compl[s].append(compl.get(s, float("nan")))
            per_seed_reward.append(reward)
        results[mode][arm] = (
            {s: float(np.mean(per_seed_compl[s])) for s in SLICE_ORDER},
            float(np.mean(per_seed_reward)),
        )

    lines = [
        "# Held-out admission-efficiency comparison",
        "",
        f"Seeds: {args.seeds} (fresh -- distinct from training seeds 256/257/258 "
        f"and the static-threshold tuning seed 950). {args.episodes} episodes/seed.",
        "Learned arms use frozen weights from their seed256 checkpoint, "
        "select_action(training=False) (greedy).",
        "",
    ]
    for mode in ["sla", "qoe"]:
        lines.append(f"## {mode.upper()}-reward group")
        lines.append("")
        lines.append("| Arm | eMBB compliant | URLLC compliant | mMTC compliant | Mean reward |")
        lines.append("|---|---|---|---|---|")
        for name, (compl, reward) in results[mode].items():
            if mode == "qoe" and name in ("dqn_sla", "a2c_sla"):
                continue
            if mode == "sla" and name in ("dqn_qoe", "a2c_qoe"):
                continue
            lines.append(
                f"| {name} | {compl['embb']*100:.1f}% | {compl['urllc']*100:.1f}% | "
                f"{compl['mmtc']*100:.1f}% | {reward:.4f} |"
            )
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[held_out_admission_comparison] wrote {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
