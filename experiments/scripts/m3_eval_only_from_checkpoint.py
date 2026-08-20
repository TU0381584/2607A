#!/usr/bin/env python3
"""Regenerates the M3 federated arm's per-sigma eval omega logs from
their already-trained, already-committed checkpoints -- no training, no
gradient steps, no DP noise re-applied (noise only ever affected
training-time gradients, not eval-time greedy action selection).

Needed for the same reason m2_eval_only_from_checkpoint.py exists:
m3_campaign's raw eval logs (unlike checkpoint.pt and
campaign_results.json) were never part of this project's committed
evidence chain and were cleaned up as disposable working data, but the
privacy-threshold-location investigation
(docs/PAPER5_M3_privacy_threshold_location.md) needs real per-seed
block precision at each sigma level to compare against the independent
replication sample, not just the committed compliance summary.

Mirrors m3_run_experiment.py's run_fl_arm EVAL section exactly (same
eval_seed offset, same policy construction) but skips the training
block, loading the existing checkpoint's weights instead. Does not
modify m3_run_experiment.py or any frozen qoe_oran_framework/ source.

Usage:
    python3 experiments/scripts/m3_eval_only_from_checkpoint.py \
        --sigma 0.0 --seeds 900 901 ... \
        --m3-campaign-dir experiments/results/m3_campaign \
        --eval-episodes 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_run_experiment as m3  # noqa: E402

sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.marl.fl_ctde_policy import FederatedGatPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--m3-campaign-dir", default="experiments/results/m3_campaign")
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()

    cfg = m3.load_saclb_config(m3.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents)
    kpm_factory = m3.make_kpm_source_factory(cfg, sd_for_slice)

    tag = f"fl_gat_ctde_sigma{args.sigma}"
    for seed in args.seeds:
        ckpt_path = f"{args.m3_campaign_dir}/{tag}/seed{seed}/train/checkpoint.pt"
        if not Path(ckpt_path).exists():
            print(f"[eval-only] sigma={args.sigma} seed={seed}: no checkpoint at {ckpt_path}, skipping")
            continue
        eval_omega_path = f"{args.m3_campaign_dir}/{tag}/seed{seed}/eval/omega_log.jsonl"
        if Path(eval_omega_path).exists():
            print(f"[eval-only] sigma={args.sigma} seed={seed}: eval log already exists, skipping")
            continue

        policy = FederatedGatPolicy(n_agents, node_dim, ctx_dim, m3.ACTION_DIM, adj,
                                     aggregator="fedavg", fedprox_mu=0.0,
                                     local_steps_per_round=50, dp_clip_norm=1.0,
                                     dp_noise_multiplier=0.0, dp_seed=seed)
        policy.load_checkpoint(ckpt_path)  # real load_state_dict -- fails loudly on any mismatch

        eval_seed = m3.EVAL_SEED_OFFSET + seed
        eval_env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
        Path(eval_omega_path).parent.mkdir(parents=True, exist_ok=True)
        with OmegaLogger(eval_omega_path) as omega:
            summary = run_episodes_marl(eval_env, policy, tag, omega, args.eval_episodes, eval_seed,
                                         f"{tag}_seed{seed}_eval_regen", "offline_eval", False, cfg)
        eval_env.close()
        print(f"[eval-only] sigma={args.sigma} seed={seed}: regenerated, "
              f"sla_compliance_all_slices={summary['sla_compliance_all_slices']:.3f}")


if __name__ == "__main__":
    main()
