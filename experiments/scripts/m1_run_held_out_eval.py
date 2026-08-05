#!/usr/bin/env python3
"""M1, Step 3: held-out offline evaluation of the six frozen single-gNB
dqn_sla checkpoints (training seeds 256-261, the exact checkpoints Stage 11
live-evaluated), under a chosen ClosedLoopKpmSource configuration -- either
the original frozen defaults (backlog_capacity=2000, drift_coef=0.1,
offered_volatility=0.04, ar1_coef=0 [i.e. i.i.d., matching the parent
exactly]) or the recalibrated config chosen by m1_fit_recalibration.py.

Same protocol as docs/STAGE12_offline_online_gap.md /
docs/STAGE13_recalibration_attempt.md: 10 fresh seeds (5001-5010) never used
for training or live eval, 10 episodes/seed = 100 held-out episodes/checkpoint,
greedy (training=False). Does NOT modify qoe_oran_framework/ (frozen); does
NOT retrain any checkpoint -- only the evaluation environment changes.

Compliance definition matches paper #4's Table I / main.tex exactly: an
episode is fully compliant iff every per-step per-slice
`per_slice_compliant` is true for all three slices for the whole episode.

Usage (from repo root, cwd=framework/ required -- see script docstring in
m1_fit_recalibration.py for why):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m1_run_held_out_eval.py \
        --backlog-capacity 2000 --drift-coef 0.1 --offered-volatility 0.04 --ar1-coef 0.0 \
        --out-dir ../experiments/results/m1_recalibration/held_out_baseline \
        --tag baseline
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from m1_recalibrated_kpm_source import RecalibratedClosedLoopKpmSource  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/experiments/configs/saclb_campaign_v2_offline_train.yaml"
CKPT_256 = f"{REPO_ROOT}/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt"
CKPT_ROOT_NEW = f"{REPO_ROOT}/experiments/results/offline_v2_reverify/sla"
HELD_OUT_SEEDS = list(range(5001, 5011))  # matches investigate_checkpoint_gap.py exactly
EPISODES_PER_SEED = 10


def checkpoint_path(train_seed: int) -> str:
    if train_seed == 256:
        return CKPT_256
    return f"{CKPT_ROOT_NEW}/seed{train_seed}/dqn/offline_closed_loop/rep_0/checkpoint.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backlog-capacity", type=float, required=True)
    ap.add_argument("--drift-coef", type=float, required=True)
    ap.add_argument("--offered-volatility", type=float, required=True)
    ap.add_argument("--ar1-coef", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}

    def kpm_factory(seed):
        return RecalibratedClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=args.backlog_capacity, drift_coef=args.drift_coef,
            offered_volatility=args.offered_volatility, ar1_coef=args.ar1_coef,
            sd_for_slice=sd_for_slice,
        )

    compliance = {}
    for train_seed in [256, 257, 258, 259, 260, 261]:
        ckpt = checkpoint_path(train_seed)

        def policy_factory(_seed, ckpt=ckpt):
            p = build_policy("dqn", cfg)
            p.load_checkpoint(ckpt)
            return p

        out_dir = f"{args.out_dir}/dqn_sla_seed{train_seed}"
        run_mc(cfg, "dqn", kpm_factory, n_reps=len(HELD_OUT_SEEDS), episodes_per_rep=EPISODES_PER_SEED,
               base_seed=HELD_OUT_SEEDS[0], mode=f"offline_held_out_{args.tag}", training=False,
               results_dir=out_dir, policy_factory=policy_factory, reward_mode="sla")

        n_compliant = 0
        n_total = 0
        for p in Path(out_dir).glob(f"dqn/offline_held_out_{args.tag}/rep_*/omega_log.jsonl"):
            with open(p) as fh:
                for line in fh:
                    rec = json.loads(line)
                    ev = rec.get("evidence", {})
                    if "episode_sla_compliance_all_slices" in ev:
                        n_total += 1
                        # episode_sla_compliance_all_slices is a FRACTION
                        # (compliant_steps/episode_steps), not a boolean --
                        # paper #4's "fully compliant" definition (main.tex,
                        # Sec. IV-A) requires every step compliant, i.e. this
                        # ratio equal to 1.0, not merely truthy/nonzero.
                        if ev["episode_sla_compliance_all_slices"] >= 1.0 - 1e-9:
                            n_compliant += 1
        pct = 100.0 * n_compliant / n_total if n_total else float("nan")
        compliance[train_seed] = {"n_compliant": n_compliant, "n_total": n_total, "pct": pct}
        print(f"[m1-eval:{args.tag}] seed={train_seed}: {n_compliant}/{n_total} = {pct:.1f}%")

    out_path = Path(args.out_dir) / f"compliance_{args.tag}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "tag": args.tag,
            "params": {"backlog_capacity": args.backlog_capacity, "drift_coef": args.drift_coef,
                       "offered_volatility": args.offered_volatility, "ar1_coef": args.ar1_coef},
            "compliance": compliance,
        }, fh, indent=2)
    print(f"[m1-eval:{args.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
