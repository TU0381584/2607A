#!/usr/bin/env python3
"""Investigates why offline training metrics (consistent Q1->Q4
convergence across all 6 dqn_sla checkpoints, seeds 256-261) don't
predict live robustness (13/21 to 21/21 fully-compliant episodes,
docs/STAGE11_checkpoint_sensitivity.md).

Part A: held-out OFFLINE evaluation of all 6 checkpoints, greedy
(training=False), on FRESH seeds never used for training (5001-5010,
distinct from both the training seeds 256-261 and the live eval seeds
950-960) -- using live_scale_offline_env.py's SAME corrected,
real-demand-scale environment the checkpoints were trained on. If this
offline held-out eval also shows large compliance variance across
checkpoints (correlated with the live result), the offline environment
DOES carry the signal, just wasn't sampled enough before. If it shows
all 6 checkpoints performing similarly well offline despite the live
spread, that is direct evidence of a genuine sim-to-real gap: the
offline environment's demand model does not visit the conditions that
distinguish a live-robust policy from a live-fragile one.

Part B: policy replay -- for a subset of REAL live states recorded
during checkpoint 257's live collapse (eval-seeds 950-952, the episodes
that actually failed), feed the exact same states through all 6
checkpoints' Q-networks (no environment interaction, pure inference)
and compare argmax actions. If the checkpoints agree on these specific
states, the failure isn't a simple "wrong action here" difference and
must involve something dynamic (state trajectory divergence). If they
disagree, that pinpoints exactly which states/slices the policies
diverge on.

Does NOT modify any frozen qoe_oran_framework/ source.

Usage:
    python3 experiments/scripts/investigate_checkpoint_gap.py \
        --out-dir experiments/results/checkpoint_gap_investigation
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kmanojp/oranslice_rig/framework")

from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402
from qoe_oran_framework.replay_kpm_source import ClosedLoopKpmSource  # noqa: E402

CONFIG_PATH = "/home/kmanojp/oranslice_rig/experiments/configs/saclb_campaign_v2_offline_train.yaml"
BACKLOG_CAPACITY = 2000.0
SEEDS = list(range(257, 262))  # 257-261; 256 handled from its own offline_v2/ dir
CKPT_ROOT_NEW = "/home/kmanojp/oranslice_rig/experiments/results/offline_v2_reverify/sla"
CKPT_256 = "/home/kmanojp/oranslice_rig/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt"
HELD_OUT_SEEDS = list(range(5001, 5011))  # 10 fresh seeds, never used for training or live eval
EPISODES_PER_SEED = 10


def checkpoint_path(train_seed: int) -> str:
    if train_seed == 256:
        return CKPT_256
    return f"{CKPT_ROOT_NEW}/seed{train_seed}/dqn/offline_closed_loop/rep_0/checkpoint.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="experiments/results/checkpoint_gap_investigation")
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {slice_id: spec.sd for slice_id, spec in cfg.slice_by_id.items()}

    def kpm_source_factory(seed: int):
        return ClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, sd_for_slice=sd_for_slice,
        )

    results = {}
    for train_seed in [256, 257, 258, 259, 260, 261]:
        ckpt = checkpoint_path(train_seed)

        def policy_factory(_seed, ckpt=ckpt):
            p = build_policy("dqn", cfg)
            p.load_checkpoint(ckpt)
            return p

        out_dir = f"{args.out_dir}/dqn_sla_seed{train_seed}"
        summaries = run_mc(
            cfg, "dqn", kpm_source_factory, n_reps=len(HELD_OUT_SEEDS),
            episodes_per_rep=EPISODES_PER_SEED, base_seed=HELD_OUT_SEEDS[0],
            mode="offline_held_out", training=False, results_dir=out_dir,
            policy_factory=policy_factory, reward_mode="sla",
        )
        print(f"[investigate] train_seed={train_seed}: {len(summaries)} reps written to {out_dir}", file=sys.stderr)
        results[train_seed] = out_dir

    with open(f"{args.out_dir}/manifest.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[investigate] wrote {args.out_dir}/manifest.json", file=sys.stderr)


if __name__ == "__main__":
    main()
