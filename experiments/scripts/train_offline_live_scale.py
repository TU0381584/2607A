#!/usr/bin/env python3
"""Stage 5 retraining entrypoint: same CLI/output layout as
qoe_oran_framework/scripts/train_offline.py, but using
live_scale_offline_env.py's VALIDATED environment (real cap/nominal
matching saclb_campaign_v2.yaml -- NOT rescaled -- mean_offered_ratio set
to real live-observed demand, Lmax=1000/backlog_capacity=2000 -- see
experiments/results/live_scale_offline/baseline_validity.md for the
zero-training validation that PASSED before this script was used for any
real training run).

Does not modify any frozen qoe_oran_framework/ source -- reuses
mc_runner.run_mc/RANEnv/build_policy via the same kpm_source_factory
extension point train_offline.py itself uses.

Usage:
    python3 experiments/scripts/train_offline_live_scale.py \
        --algorithm dqn --reward-mode qoe --episodes 300 --seed 256 \
        --results-dir experiments/results/offline_v2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import run_mc  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2_offline_train.yaml"
BACKLOG_CAPACITY = 2000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", required=True, choices=["dqn", "a2c", "rainbow"])
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=256)
    ap.add_argument("--results-dir", default="experiments/results/offline_v2")
    ap.add_argument("--reward-mode", choices=["sla", "qoe"], default="qoe")
    ap.add_argument("--backlog-capacity", type=float, default=BACKLOG_CAPACITY,
                     help="Stage 12 finding: the default 2000 was chosen purely so "
                          "accept_all/reject_all/threshold_like differentiate (Stage 5's "
                          "own criterion) and was never checked against real live margin "
                          "magnitude -- at 2000, mean offline SLA margin is ~-0.60 vs real "
                          "live's ~+0.7-0.75 for the same checkpoint. Override for a "
                          "recalibration attempt; see docs/STAGE13_recalibration_attempt.md.")
    args = ap.parse_args()

    cfg = load_saclb_config(args.config)

    def kpm_source_factory(seed: int):
        sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=args.backlog_capacity, sd_for_slice=sd_for_slice,
        )

    out_dir = f"{args.results_dir}/{args.reward_mode}/seed{args.seed}"
    summaries = run_mc(
        cfg, args.algorithm, kpm_source_factory, n_reps=1,
        episodes_per_rep=args.episodes, base_seed=args.seed,
        mode="offline_closed_loop", training=True,
        results_dir=out_dir, reward_mode=args.reward_mode,
    )
    for s in summaries:
        print(f"[train_offline_live_scale] {args.algorithm}/{args.reward_mode} seed={args.seed}: {s}")


if __name__ == "__main__":
    main()
