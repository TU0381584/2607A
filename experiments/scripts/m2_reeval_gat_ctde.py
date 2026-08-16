#!/usr/bin/env python3
"""One-time correction: re-run gat_ctde's EVAL phase only (no retraining)
for all 30 M2 campaign seeds, against the already-existing, already-
verified-correct checkpoints, writing a fresh single-block eval omega log.

Why: OmegaLogger opens its output file in append mode and never truncates.
gat_ctde was retrained in place three times across this project's
collapse-fix history (original -> LayerNorm-only -> per-slice-heads), and
because nothing ever cleared seed<N>/eval/omega_log.jsonl between those
runs, every seed's eval log silently accumulated all three runs stacked
together (150 episode records instead of 50). Every metric reported in
the paper so far (Table I, Fig 2/3, the Abstract's headline paired-diff
claim, and M3's Federation Cost section, which reuses these same logs as
its centralized reference) was computed from that contaminated file --
a hidden 3-way average across two stale, architecturally-different runs
and the true final one. m2_run_experiment.py now clears seed<N>/ before
any FUTURE fresh run (_clear_seed_dir); this script fixes the 30 seeds
that already exist.

The checkpoints themselves are NOT corrupted (torch.save overwrites, does
not append) -- verified directly: seed900's checkpoint, loaded fresh and
evaluated in isolation, reproduces the LAST block of its own contaminated
log exactly (0.006 compliance, 3506 total blocks). So this script only
needs to re-run eval, not training -- cheap (50 forward-pass-only
episodes/seed, no gradient steps) versus a full 300-episode retrain.

Usage:
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m2_reeval_gat_ctde.py \
        --campaign-dir ../experiments/results/m2_campaign
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.env import RANEnv  # noqa: E402
from qoe_oran_framework.omega_logger import OmegaLogger  # noqa: E402
from qoe_oran_framework.marl.ctde_policy import GatCtdeMarlPolicy  # noqa: E402
from qoe_oran_framework.marl.marl_env import node_feature_dim, request_context_dim  # noqa: E402
from qoe_oran_framework.marl.marl_training import run_episodes_marl  # noqa: E402
from qoe_oran_framework.marl.topology import build_adjacency  # noqa: E402
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/framework/qoe_oran_framework/configs/saclb_offline_dqn.yaml"
BACKLOG_CAPACITY = 2000.0
ACTION_DIM = 2
EVAL_SEED_OFFSET = 5000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m2_campaign")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="Override: only re-eval these seeds (smoke-test subset). Default: all 30.")
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}

    with open(Path(args.campaign_dir) / "campaign_results.json") as fh:
        campaign = json.load(fh)
    seeds = args.seeds if args.seeds is not None else [s for g in campaign["seed_groups"] for s in g]

    n_agents = len(cfg.gnb_ids)
    node_dim = node_feature_dim(cfg)
    ctx_dim = request_context_dim(cfg)
    adj = build_adjacency(n_agents)

    def kpm_factory(seed):
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )

    fresh_results = {}
    for seed in seeds:
        ckpt_path = f"{args.campaign_dir}/gat_ctde/seed{seed}/train/checkpoint.pt"
        eval_path = Path(f"{args.campaign_dir}/gat_ctde/seed{seed}/eval/omega_log.jsonl")

        policy = GatCtdeMarlPolicy(n_agents, node_dim, ctx_dim, ACTION_DIM, adj)
        policy.load_checkpoint(ckpt_path)  # strict=True internally -- raises loudly on any architecture mismatch

        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.unlink(missing_ok=True)  # the actual fix: clear before writing, don't append onto stale data

        eval_seed = EVAL_SEED_OFFSET + seed
        eval_env = RANEnv(cfg, kpm_factory(eval_seed), seed=eval_seed, reward_mode="sla")
        with OmegaLogger(str(eval_path)) as omega:
            summary = run_episodes_marl(eval_env, policy, "gat_ctde", omega, args.eval_episodes, eval_seed,
                                         f"gat_ctde_seed{seed}_eval_reeval", "offline_eval", False, cfg)
        eval_env.close()
        fresh_results[seed] = summary
        print(f"[reeval] seed={seed}: sla_compliance_all_slices={summary['sla_compliance_all_slices']:.4f}")

    for seed in seeds:
        campaign["results"]["gat_ctde"][str(seed)] = fresh_results[seed]
    out_path = Path(args.campaign_dir) / "campaign_results.json"
    with open(out_path, "w") as fh:
        json.dump(campaign, fh, indent=2)
    print(f"[reeval] wrote corrected gat_ctde entries into {out_path}")


if __name__ == "__main__":
    main()
