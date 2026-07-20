#!/usr/bin/env python3
"""Real beta sweep for the admission-efficiency objective: actual short
training runs (not a retroactive recompute on fixed rollouts, unlike the
earlier beta_sensitivity_probe.py) under a grid of candidate beta values,
for both dqn and a2c under reward_mode="qoe" (beta only appears in eq.9).

Motivation: beta_sensitivity_probe.py's retroactive analysis (CAMPAIGN_LOG,
2026-07-20) showed accept_all's cost term (~2.1) so dwarfs its mos term
(~0.02-0.25) that beta in [0.05, 1.0] never changes which extreme wins --
suggesting the interesting range is below 0.05. This script tests that
directly by actually training under each candidate, not just recomputing
old rollouts.

Does not modify frozen qoe_oran_framework/ source -- loads the frozen
config once, overrides cfg.qoe.reward.beta in memory per sweep point
(QoeRewardWeights is a plain mutable dataclass), and reuses
mc_runner.run_mc/RANEnv exactly as train_offline_admission_efficiency.py
does.

Usage:
    python3 experiments/scripts/beta_sweep_training.py \
        --betas 0.001 0.005 0.01 0.02 0.05 0.1 0.2 \
        --episodes 100 --seed 256
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

OUT_ROOT = "/home/kmanojp/oranslice_rig/experiments/results/admission_efficiency_beta_sweep"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--betas", type=float, nargs="+",
                     default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--algorithms", nargs="+", default=["dqn", "a2c"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=256)
    args = ap.parse_args()

    for beta in args.betas:
        cfg = load_saclb_config(CONFIG_PATH)
        cfg.qoe.reward.beta = beta

        def kpm_source_factory(seed, cfg=cfg):
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

        for algo in args.algorithms:
            out_dir = f"{OUT_ROOT}/beta{beta}/seed{args.seed}"
            print(f"=== training algo={algo} beta={beta} seed={args.seed} ===", flush=True)
            summaries = run_mc(
                cfg, algo, kpm_source_factory, n_reps=1,
                episodes_per_rep=args.episodes, base_seed=args.seed,
                mode="offline_closed_loop", training=True,
                results_dir=out_dir, reward_mode="qoe",
            )
            for s in summaries:
                print(f"  {algo} beta={beta}: mean_reward_per_step={s.mean_reward_per_step:.4f} "
                      f"mean_cost={s.mean_cost:.4f} sla_compliance={s.sla_compliance_by_slice}", flush=True)


if __name__ == "__main__":
    main()
