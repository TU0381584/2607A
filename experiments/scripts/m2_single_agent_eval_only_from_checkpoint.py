#!/usr/bin/env python3
"""Regenerates single_agent_dqn's undisrupted-baseline eval omega log from
its ALREADY-TRAINED, already-committed checkpoint -- no training, no
gradient steps. Same rationale and pattern as
m2_eval_only_from_checkpoint.py (which covers gat_ctde/independent_dqn
only): m2_campaign's raw eval logs were never part of this project's
committed evidence chain and were cleaned up as disposable working data.

Mirrors m2_run_experiment.py's run_single_agent_dqn_arm EVAL block
exactly (same eval_seed offset, same mc_runner.run_mc call, same
policy_factory pattern) but skips the training block entirely, loading
the existing checkpoint's weights instead of training a fresh one. Does
not modify frozen qoe_oran_framework/ source or m2_run_experiment.py.

Usage:
    python3 experiments/scripts/m2_single_agent_eval_only_from_checkpoint.py \
        --seeds 900 901 902 ... \
        --m2-campaign-dir experiments/results/m2_campaign \
        --eval-episodes 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_run_experiment as m2  # noqa: E402

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--m2-campaign-dir", default="experiments/results/m2_campaign")
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()

    cfg = m2.load_saclb_config(m2.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    kpm_factory = m2.make_kpm_source_factory(cfg, sd_for_slice)

    for seed in args.seeds:
        ckpt = f"{args.m2_campaign_dir}/single_agent_dqn/seed{seed}/train/dqn/offline_train/rep_0/checkpoint.pt"
        if not Path(ckpt).exists():
            print(f"[eval-only:single] seed={seed}: no checkpoint at {ckpt}, skipping")
            continue
        eval_omega_path = Path(args.m2_campaign_dir) / "single_agent_dqn" / f"seed{seed}" / "eval" / "dqn" / "offline_eval" / "rep_0" / "omega_log.jsonl"
        if eval_omega_path.exists():
            print(f"[eval-only:single] seed={seed}: eval log already exists, skipping")
            continue

        def policy_factory(_s, ckpt=ckpt):
            p = build_policy("dqn", cfg)
            p.load_checkpoint(ckpt)
            return p

        eval_seed = m2.EVAL_SEED_OFFSET + seed
        eval_dir = f"{args.m2_campaign_dir}/single_agent_dqn/seed{seed}/eval"
        summaries = run_mc(cfg, "dqn", kpm_factory, n_reps=1, episodes_per_rep=args.eval_episodes,
                           base_seed=eval_seed, mode="offline_eval", training=False,
                           results_dir=eval_dir, policy_factory=policy_factory, reward_mode="sla")
        compliance = summaries[0].sla_compliance_all_slices if summaries else float("nan")
        print(f"[eval-only:single] seed={seed}: regenerated, sla_compliance_all_slices={compliance:.3f}")


if __name__ == "__main__":
    main()
