#!/usr/bin/env python3
"""Evaluates baseline (frozen ceiling, ceiling_step_ratio=0) vs. the 4
DRL arms (frozen weights, trained by train_offline_congested.py) under
HELD-OUT congested/dynamic/randomized episodes -- same
CongestedRandomKpmSource mechanism as training, different eval seeds
(950/951/952, matching this project's live-eval seed convention) so
episodes are not the exact ones any arm trained on.

Per-slice SLA compliance is the primary metric, with URLLC broken out
explicitly (this is the slice the reward is designed to protect first --
priority_weight=5.0, violation_penalty=8.0, both highest of the 3 slices
in saclb_offline_congested_v1.yaml).

Usage:
    python3 experiments/scripts/eval_congested_vs_baseline.py \
        --ckpt-root experiments/results/offline_congested \
        --out experiments/results/congested_vs_baseline
"""
import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from shared_pool_kpm_source import SharedPoolCongestedKpmSource  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.mc_runner import _select_actions, build_policy  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_offline_congested_v1.yaml"
ARMS = {
    "baseline": None,  # special-cased: ceiling_step_ratio=0, always-accept
    "dqn_sla": ("dqn", "sla"), "a2c_sla": ("a2c", "sla"),
    "dqn_qoe": ("dqn", "qoe"), "a2c_qoe": ("a2c", "qoe"),
}
SLICE_ORDER = ["urllc", "embb", "mmtc"]  # URLLC first -- the slice under test


def make_env_and_policy(arm: str, algo, reward_mode, seed: int, congestion_range, ckpt_root: str, backlog_capacity: float, shared_pool_prb: float):
    cfg = load_saclb_config(CONFIG_PATH)
    if arm == "baseline":
        cfg.arrivals = dataclasses.replace(cfg.arrivals, ceiling_step_ratio=0)
        reward_mode = "sla"
    nominal_ratio = {s: spec.nominal_ratio for s, spec in cfg.slice_by_id.items()}
    sd_for_slice = {s: spec.sd for s, spec in cfg.slice_by_id.items()}
    episode_rng = np.random.RandomState(seed + 5000)  # distinct stream from training's seed+1

    kpm = SharedPoolCongestedKpmSource(
        seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id), B=cfg.B,
        nominal_ratio=nominal_ratio, congestion_range=congestion_range,
        episode_rng=episode_rng, sd_for_slice=sd_for_slice, backlog_capacity=backlog_capacity,
        shared_pool_prb=shared_pool_prb,
    )
    env = RANEnv(cfg, kpm, seed=seed, reward_mode=reward_mode)

    if arm == "baseline":
        policy = None  # action is always 1 (accept) -- ceiling can't move regardless
    else:
        policy = build_policy(algo, cfg)
        ckpt = Path(ckpt_root) / reward_mode / f"seed256" / algo / "checkpoint.pt"
        policy.load_checkpoint(str(ckpt))
    return env, kpm, policy, cfg


def run_arm(arm: str, seeds: list, episodes_per_seed: int, congestion_range, ckpt_root: str, backlog_capacity: float, shared_pool_prb: float) -> dict:
    spec = ARMS[arm]
    algo, reward_mode = spec if spec else (None, "sla")
    compliance = {s: [] for s in SLICE_ORDER}
    margin = {s: [] for s in SLICE_ORDER}
    rejections = {s: [] for s in SLICE_ORDER}

    for seed in seeds:
        env, kpm, policy, cfg = make_env_and_policy(arm, algo, reward_mode, seed, congestion_range, ckpt_root, backlog_capacity, shared_pool_prb)
        try:
            for _ep in range(episodes_per_seed):
                kpm.new_episode_congestion()
                obs = env.reset()
                done = False
                ep_rejections = {s: 0 for s in SLICE_ORDER}
                while not done:
                    pending = env.pending_requests()
                    cluster_state = env.last_cluster_state
                    if arm == "baseline":
                        actions = [1] * len(pending)
                    else:
                        actions, _ = _select_actions(policy, algo, pending, obs, cluster_state, cfg, training=False)
                    result = env.step(actions)
                    obs = result.obs
                    done = result.done
                    rb = result.info["reward_breakdown"]
                    for s, c in rb.get("per_slice_compliant", {}).items():
                        compliance[s].append(bool(c))
                    for s, m in rb.get("per_slice_sla_margin", {}).items():
                        margin[s].append(m)
                    for block in result.info["primary_blocks"]:
                        if block["slice_id"] in ep_rejections:
                            ep_rejections[block["slice_id"]] += 1
                for s in SLICE_ORDER:
                    rejections[s].append(ep_rejections[s])
        finally:
            env.close()

    return {
        "compliance_pct": {s: float(np.mean(v)) * 100 if v else float("nan") for s, v in compliance.items()},
        "mean_margin": {s: float(np.mean(v)) if v else float("nan") for s, v in margin.items()},
        "mean_rejections_per_episode": {s: float(np.mean(v)) if v else float("nan") for s, v in rejections.items()},
        "n_steps_pooled": {s: len(v) for s, v in compliance.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", default="experiments/results/offline_congested")
    ap.add_argument("--seeds", type=int, nargs="+", default=[950, 951, 952])
    ap.add_argument("--episodes-per-seed", type=int, default=10)
    ap.add_argument("--congestion-lo", type=float, default=0.4)
    ap.add_argument("--congestion-hi", type=float, default=1.3)
    ap.add_argument("--backlog-capacity", type=float, default=15.0)
    ap.add_argument("--shared-pool-prb", type=float, default=8.0)
    ap.add_argument("--out", default="experiments/results/congested_vs_baseline")
    args = ap.parse_args()

    results = {}
    for arm in ARMS:
        print(f"[eval] running {arm} ...", file=sys.stderr)
        results[arm] = run_arm(
            arm, args.seeds, args.episodes_per_seed,
            (args.congestion_lo, args.congestion_hi), args.ckpt_root, args.backlog_capacity,
            args.shared_pool_prb,
        )
        print(f"  {arm}: compliance={results[arm]['compliance_pct']}")

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval] wrote {out_path / 'results.json'}")


if __name__ == "__main__":
    main()
