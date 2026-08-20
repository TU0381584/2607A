#!/usr/bin/env python3
"""Regenerates an M2 arm's undisrupted-baseline eval omega log from its
ALREADY-TRAINED, already-committed checkpoint -- no training, no gradient
steps. Needed because m2_campaign's raw eval logs (unlike checkpoint.pt
and campaign_results.json) were never part of this project's committed
evidence chain and were cleaned up as disposable working data (matching
m2_campaign's own established convention -- see BRINGUP disk-cleanup
history) -- but m4_correctness_metrics.py's disruption-cost comparison
needs a real per-seed baseline eval log to pair against, not just the
committed compliance summary.

Mirrors m2_run_experiment.py's run_gat_ctde_arm/run_independent_dqn_arm
EVAL section exactly (same eval_seed offset, same policy construction,
same RANEnv/OmegaLogger/run_episodes_marl call) but skips the training
block entirely, loading the existing checkpoint's weights instead of
initialising and training a fresh policy. Does not modify
m2_run_experiment.py or any frozen qoe_oran_framework/ source.

Usage:
    python3 experiments/scripts/m2_eval_only_from_checkpoint.py \
        --arm gat_ctde --seeds 900 901 902 ... \
        --m2-campaign-dir experiments/results/m2_campaign \
        --eval-episodes 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_run_experiment as m2  # noqa: E402

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.marl.ctde_policy import GatCtdeMarlPolicy  # noqa: E402
from qoe_oran_framework.marl.independent_dqn_ablation import IndependentPerGnbDqnPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["gat_ctde", "independent_dqn"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--m2-campaign-dir", default=m2.DEFAULT_M2_CAMPAIGN_DIR
                     if hasattr(m2, "DEFAULT_M2_CAMPAIGN_DIR") else "experiments/results/m2_campaign")
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()

    cfg = m2.load_saclb_config(m2.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    kpm_factory = m2.make_kpm_source_factory(cfg, sd_for_slice)

    for seed in args.seeds:
        ckpt_path = f"{args.m2_campaign_dir}/{args.arm}/seed{seed}/train/checkpoint.pt"
        if not Path(ckpt_path).exists():
            print(f"[eval-only] seed={seed}: no checkpoint at {ckpt_path}, skipping")
            continue
        eval_omega_path = f"{args.m2_campaign_dir}/{args.arm}/seed{seed}/eval/omega_log.jsonl"
        if Path(eval_omega_path).exists():
            print(f"[eval-only] seed={seed}: eval log already exists, skipping")
            continue

        if args.arm == "gat_ctde":
            adj = build_adjacency(n_agents)
            policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, m2.ACTION_DIM, adj)
        else:
            policy = IndependentPerGnbDqnPolicy(n_agents, node_dim, ctx_dim, m2.ACTION_DIM)
        policy.load_checkpoint(ckpt_path)  # real load_state_dict -- fails loudly on any mismatch

        eval_seed = m2.EVAL_SEED_OFFSET + seed
        eval_env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
        Path(eval_omega_path).parent.mkdir(parents=True, exist_ok=True)
        with OmegaLogger(eval_omega_path) as omega:
            summary = run_episodes_marl(eval_env, policy, args.arm, omega, args.eval_episodes, eval_seed,
                                         f"{args.arm}_seed{seed}_eval_regen", "offline_eval", False, cfg)
        eval_env.close()
        print(f"[eval-only] seed={seed}: regenerated, "
              f"sla_compliance_all_slices={summary['sla_compliance_all_slices']:.3f}")


if __name__ == "__main__":
    main()
