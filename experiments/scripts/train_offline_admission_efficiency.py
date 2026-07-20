#!/usr/bin/env python3
"""Offline training entrypoint for the "admission efficiency under
overload" objective (experiments/configs/saclb_admission_efficiency_v1.yaml),
mirroring qoe_oran_framework/scripts/train_offline.py's CLI/output layout
exactly (so existing plotting scripts like fig1_training_convergence.py
work unchanged against --offline-root), but using
admission_efficiency_env.py's VALIDATED environment construction
(oversub_of_cap=1.2 x max_ratio_cap, backlog_capacity=1000.0, real
sd_for_slice) instead of train_offline.py's own (nominal_ratio-relative
1.25x, backlog_capacity=200 default) -- see CAMPAIGN_LOG.md's 2026-07-20
entries for why those numbers don't produce a differentiable environment.

Does not modify any frozen qoe_oran_framework/ source -- reuses
mc_runner.run_mc/RANEnv/build_policy exactly as-is via the same
kpm_source_factory extension point train_offline.py itself uses.

Usage:
    python3 experiments/scripts/train_offline_admission_efficiency.py \
        --algorithm dqn --reward-mode qoe --episodes 40 --seed 256 \
        --results-dir experiments/results/admission_efficiency_offline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from admission_efficiency_env import BACKLOG_CAPACITY, CONFIG_PATH, OVERSUB_OF_CAP  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import run_mc  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow"])
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--results-dir", default="experiments/results/admission_efficiency_offline")
    ap.add_argument("--reward-mode", choices=["sla", "qoe"], default="qoe")
    args = ap.parse_args()

    cfg = load_saclb_config(args.config)

    def kpm_source_factory(seed: int):
        sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
        mean_offered_ratio = {
            slice_id: min(0.98, OVERSUB_OF_CAP * spec.max_ratio_cap / 100.0)
            for slice_id, spec in cfg.slice_by_id.items()
        }
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=mean_offered_ratio,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )

    out_dir = f"{args.results_dir}/{args.reward_mode}/seed{args.seed}"
    summaries = run_mc(
        cfg, args.algorithm, kpm_source_factory, n_reps=1,
        episodes_per_rep=args.episodes, base_seed=args.seed,
        mode="offline_closed_loop", training=True,
        results_dir=out_dir, reward_mode=args.reward_mode,
    )
    for s in summaries:
        print(f"[train_offline_admission_efficiency] {args.algorithm}/{args.reward_mode} "
              f"seed={args.seed}: {s}")


if __name__ == "__main__":
    main()
