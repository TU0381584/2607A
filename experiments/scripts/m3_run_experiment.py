#!/usr/bin/env python3
"""M3: federated GAT-CTDE arm (FederatedGatPolicy, framework/
qoe_oran_framework/marl/fl_ctde_policy.py) -- each of the 3 gNBs trains
its own local copy of the same GAT+Q-head architecture the centralized
gat_ctde arm uses, synced via periodic FedAvg/FedProx aggregation, with an
optional DP-SGD-style per-client gradient clip+noise step (noise_multiplier
as the swept privacy knob). Mirrors m2_run_experiment.py's structure and
CLI conventions exactly (same config, same env, same offline stress-regime
framing per docs/PAPER5_M1_recalibration.md) so results are directly
comparable to the Block E centralized-CTDE campaign
(experiments/results/m2_campaign/campaign_results.json) without rerunning
that arm.

Does not modify frozen qoe_oran_framework/ source.

Usage (from repo root, cwd=framework/ required):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m3_run_experiment.py \
        --seeds 900 901 902 --train-episodes 100 --eval-episodes 20 \
        --noise-multiplier 1.0 --out-dir ../experiments/results/m3_fl_dp
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.marl.fl_ctde_policy import FederatedGatPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml"
BACKLOG_CAPACITY = 2000.0  # same offline default M1/M2 use, unchanged
ACTION_DIM = 2
EVAL_SEED_OFFSET = 5000  # same disjoint-from-train convention as M1/M2
DP_DELTA = 1e-5  # standard default (< 1/n_transitions_per_client), fixed so it's never swept accidentally


def make_kpm_source_factory(cfg, sd_for_slice):
    def factory(seed):
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )
    return factory


def run_fl_arm(cfg, sd_for_slice, seeds, train_episodes, eval_episodes, out_dir, tag,
                aggregator="fedavg", fedprox_mu=0.0, dp_noise_multiplier=0.0,
                dp_clip_norm=1.0, local_steps_per_round=50):
    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents)
    kpm_factory = make_kpm_source_factory(cfg, sd_for_slice)

    results = {}
    for seed in seeds:
        policy = FederatedGatPolicy(
            n_agents, node_dim, ctx_dim, ACTION_DIM, adj,
            aggregator=aggregator, fedprox_mu=fedprox_mu,
            local_steps_per_round=local_steps_per_round,
            dp_clip_norm=dp_clip_norm, dp_noise_multiplier=dp_noise_multiplier, dp_seed=seed,
        )
        env = RANEnv(cfg, kpm_factory(seed), seed=seed, reward_mode="sla")
        train_dir = f"{out_dir}/{tag}/seed{seed}/train"
        Path(train_dir).mkdir(parents=True, exist_ok=True)
        with OmegaLogger(f"{train_dir}/omega_log.jsonl") as omega:
            run_episodes_marl(env, policy, tag, omega, train_episodes, seed,
                               f"{tag}_seed{seed}_train", "offline_train", True, cfg)
        env.close()
        ckpt_path = f"{train_dir}/checkpoint.pt"
        policy.save_checkpoint(ckpt_path)

        eval_seed = EVAL_SEED_OFFSET + seed
        eval_env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
        eval_dir = f"{out_dir}/{tag}/seed{seed}/eval"
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        with OmegaLogger(f"{eval_dir}/omega_log.jsonl") as omega:
            summary = run_episodes_marl(eval_env, policy, tag, omega, eval_episodes, eval_seed,
                                         f"{tag}_seed{seed}_eval", "offline_eval", False, cfg)
        eval_env.close()
        summary["dp_step_count_per_client"] = list(policy.dp_step_count)
        summary["round_count"] = policy.round_count
        summary["aggregator"] = aggregator
        summary["fedprox_mu"] = fedprox_mu
        summary["dp_noise_multiplier"] = dp_noise_multiplier
        summary["dp_clip_norm"] = dp_clip_norm
        results[seed] = summary
        print(f"[m3:{tag}] seed={seed}: eval sla_compliance_all_slices="
              f"{summary['sla_compliance_all_slices']:.3f} rounds={policy.round_count} "
              f"dp_steps={policy.dp_step_count}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902])
    ap.add_argument("--train-episodes", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out-dir", default=f"{REPO_ROOT}/experiments/results/m3_fl_dp")
    ap.add_argument("--tag", default="fl_gat_ctde")
    ap.add_argument("--aggregator", default="fedavg", choices=["fedavg", "fedprox"])
    ap.add_argument("--fedprox-mu", type=float, default=0.0)
    ap.add_argument("--noise-multiplier", type=float, default=0.0)
    ap.add_argument("--dp-clip-norm", type=float, default=1.0)
    ap.add_argument("--local-steps-per-round", type=int, default=50)
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[m3] {len(cfg.gnb_ids)}-gNB config loaded: {cfg.gnb_ids}, "
          f"aggregator={args.aggregator}, noise_multiplier={args.noise_multiplier}")

    t0 = time.time()
    results = run_fl_arm(
        cfg, sd_for_slice, args.seeds, args.train_episodes, args.eval_episodes, args.out_dir, args.tag,
        aggregator=args.aggregator, fedprox_mu=args.fedprox_mu,
        dp_noise_multiplier=args.noise_multiplier, dp_clip_norm=args.dp_clip_norm,
        local_steps_per_round=args.local_steps_per_round,
    )

    out_path = Path(args.out_dir) / f"{args.tag}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[m3] wrote {out_path}, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
