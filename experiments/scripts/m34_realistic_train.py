#!/usr/bin/env python3
"""M34: retrain single_agent_dqn against RealisticServedKpmSource (see
realistic_served_kpm_source.py for why -- ratio-independent, empirically
measured served-PRB per slice, interpolated across the real 3-UE/6-UE
congestion range this rig actually produces, instead of
ClosedLoopKpmSource's ratio-derived served that structurally cannot
reach live congestion levels).

Does not touch qoe_oran_framework/ (frozen) or m6_run_experiment.py
(existing, working script) -- calls build_policy/run_mc directly, same
pattern m32/m33's scripts already established, swapping in only the new
KPM source factory.

Usage:
    python3 experiments/scripts/m34_realistic_train.py --seed 900 \
        --train-episodes 300 --eval-episodes 50 \
        --out-dir experiments/results/m34_realistic_retrain
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402
from realistic_served_kpm_source import RealisticServedKpmSource  # noqa: E402
from m2_correctness_metrics import per_seed_metrics  # noqa: E402

CONFIG_PATH = "qoe_oran_framework/configs/saclb_offline_live1gnb.yaml"
EVAL_SEED_OFFSET = 5000


def make_kpm_factory(cfg, sd_for_slice):
    def factory(seed):
        return RealisticServedKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, sd_for_slice=sd_for_slice,
        )
    return factory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--out-dir", default="experiments/results/m34_realistic_retrain")
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    kpm_factory = make_kpm_factory(cfg, sd_for_slice)

    train_dir = f"{args.out_dir}/seed{args.seed}/train"
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    print(f"[m34-train] training single_agent_dqn seed={args.seed} against RealisticServedKpmSource "
          f"({args.train_episodes} episodes)...")
    run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=args.train_episodes, base_seed=args.seed,
           mode="offline_train", training=True, results_dir=train_dir, reward_mode="sla")
    ckpt_path = f"{train_dir}/dqn/offline_train/rep_0/checkpoint.pt"

    def policy_factory(_s, ckpt=ckpt_path):
        p = build_policy("dqn", cfg)
        p.load_checkpoint(ckpt)
        return p

    eval_dir = f"{args.out_dir}/seed{args.seed}/eval"
    eval_seed = EVAL_SEED_OFFSET + args.seed
    print(f"[m34-train] evaluating ({args.eval_episodes} episodes)...")
    run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=args.eval_episodes,
           base_seed=eval_seed, mode="offline_eval", training=False,
           results_dir=eval_dir, policy_factory=policy_factory, reward_mode="sla")

    eval_omega = f"{eval_dir}/dqn/offline_eval/rep_0/omega_log.jsonl"
    mrps, mmtc_b, total_b = per_seed_metrics(eval_omega)
    precision = mmtc_b / total_b if total_b else float("nan")
    print(f"\n[m34-train] seed={args.seed} eval: mean_reward_per_step={mrps:.4f} "
          f"mmtc_blocks={mmtc_b} total_blocks={total_b} "
          f"precision={'UNDEFINED (collapsed)' if total_b == 0 else f'{precision:.4f}'}")
    print(f"[m34-train] checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
