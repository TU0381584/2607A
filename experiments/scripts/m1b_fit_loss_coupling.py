#!/usr/bin/env python3
"""M1 Block B: small, time-boxed grid fit of loss_noise_std/loss_noise_ar1
(LossBacklogCoupledKpmSource) against the live pooled per-slice SLA-margin
distribution -- the same target/methodology as m1_fit_recalibration.py,
scoped down to only the new loss-channel parameters since this is a single
bounded experiment testing one hypothesis, not a re-opening of the
demand-side search. Demand-side params fixed at M1's own best-fit
(backlog_capacity=3200, drift_coef=0.1, offered_volatility=0.04, ar1=0.0,
from experiments/results/m1_recalibration/fit_search.json).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
from live_scale_offline_env import MEAN_OFFERED_RATIO  # noqa: E402
from m1b_loss_backlog_coupled_source import LossBacklogCoupledKpmSource  # noqa: E402
from qoe_oran_framework.config import load_saclb_config  # noqa: E402
from qoe_oran_framework.mc_runner import build_policy, run_mc  # noqa: E402

REPO_ROOT = "/home/kmanojp/oranslice_rig"
CONFIG_PATH = f"{REPO_ROOT}/experiments/configs/saclb_campaign_v2_offline_train.yaml"
CKPT_256 = f"{REPO_ROOT}/experiments/results/offline_v2/sla/seed256/dqn/offline_closed_loop/rep_0/checkpoint.pt"
CKPT_ROOT_NEW = f"{REPO_ROOT}/experiments/results/offline_v2_reverify/sla"
FIT_CHECKPOINTS = [256, 257, 258]
FIT_SEEDS_PER_CKPT = 2
FIT_EPISODES_PER_SEED = 3
SLICE_IDS = ["embb", "urllc", "mmtc"]

# M1's own best-fit demand-side params, held fixed here.
BACKLOG_CAPACITY = 3200.0
DRIFT_COEF = 0.1
OFFERED_VOLATILITY = 0.04
AR1_COEF = 0.0

GRID_LOSS_NOISE_STD = [0.0, 0.05, 0.1, 0.2]
GRID_LOSS_NOISE_AR1 = [0.0, 0.5, 0.85]


def checkpoint_path(train_seed: int) -> str:
    if train_seed == 256:
        return CKPT_256
    return f"{CKPT_ROOT_NEW}/seed{train_seed}/dqn/offline_closed_loop/rep_0/checkpoint.pt"


def collect_margins(cfg, sd_for_slice, loss_noise_std, loss_noise_ar1, out_root):
    pooled = {s: [] for s in SLICE_IDS}

    def kpm_factory(seed):
        return LossBacklogCoupledKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=BACKLOG_CAPACITY, drift_coef=DRIFT_COEF,
            offered_volatility=OFFERED_VOLATILITY, ar1_coef=AR1_COEF,
            loss_noise_std=loss_noise_std, loss_noise_ar1=loss_noise_ar1,
            sd_for_slice=sd_for_slice,
        )

    for train_seed in FIT_CHECKPOINTS:
        ckpt = checkpoint_path(train_seed)

        def policy_factory(_seed, ckpt=ckpt):
            p = build_policy("dqn", cfg)
            p.load_checkpoint(ckpt)
            return p

        out_dir = f"{out_root}/seed{train_seed}"
        run_mc(cfg, "dqn", kpm_factory, n_reps=FIT_SEEDS_PER_CKPT, episodes_per_rep=FIT_EPISODES_PER_SEED,
               base_seed=8001, mode="fit_search_lossb", training=False, results_dir=out_dir,
               policy_factory=policy_factory, reward_mode="sla")
        for p in Path(out_dir).glob("dqn/fit_search_lossb/rep_*/omega_log.jsonl"):
            with open(p) as fh:
                for line in fh:
                    rec = json.loads(line)
                    m = rec.get("evidence", {}).get("per_slice_sla_margin")
                    if m is None:
                        continue
                    for s in SLICE_IDS:
                        if s in m:
                            pooled[s].append(m[s])
    return pooled


def loss(offline_stats, live_stats):
    total = 0.0
    for s in SLICE_IDS:
        om, ostd = offline_stats[s]
        lm, lstd = live_stats[s]
        denom = max(lstd, 1e-3)
        total += ((om - lm) / denom) ** 2 + ((ostd - lstd) / denom) ** 2
    return total


def main() -> None:
    import numpy as np
    with open(f"{REPO_ROOT}/experiments/results/m1_recalibration/live_traces.json") as fh:
        live = json.load(fh)
    live_stats = {}
    for s in SLICE_IDS:
        v = np.array(live["pooled_margin_by_slice"][s])
        live_stats[s] = (float(v.mean()), float(v.std()))
    print(f"[m1b-fit] live target: {live_stats}")

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}

    grid = [(std, ar1) for std in GRID_LOSS_NOISE_STD for ar1 in GRID_LOSS_NOISE_AR1]
    print(f"[m1b-fit] grid size: {len(grid)}")

    results = []
    t0 = time.time()
    for i, (std, ar1) in enumerate(grid):
        pooled = collect_margins(cfg, sd_for_slice, std, ar1, f"/tmp/m1b_fit_scratch/pt{i}")
        offline_stats = {s: (float(np.array(v).mean()), float(np.array(v).std())) for s, v in pooled.items() if v}
        if len(offline_stats) < 3:
            continue
        l = loss(offline_stats, live_stats)
        results.append({"loss_noise_std": std, "loss_noise_ar1": ar1, "offline_stats": offline_stats, "loss": l})
        print(f"[m1b-fit] {i+1}/{len(grid)} std={std} ar1={ar1} loss={l:.3f} elapsed={time.time()-t0:.0f}s")

    results.sort(key=lambda r: r["loss"])
    out_path = Path(f"{REPO_ROOT}/experiments/results/m1_recalibration/fit_search_lossb.json")
    with open(out_path, "w") as fh:
        json.dump({"live_stats": live_stats, "results": results}, fh, indent=2)
    print(f"[m1b-fit] wrote {out_path}")
    print("[m1b-fit] best config:", results[0])


if __name__ == "__main__":
    main()
