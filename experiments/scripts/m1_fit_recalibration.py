#!/usr/bin/env python3
"""M1, Step 2: grid-search fit of RecalibratedClosedLoopKpmSource's
backlog_capacity / drift_coef / offered_volatility / ar1_coef against the
live per-slice SLA-margin distribution extracted by
m1_extract_live_traces.py, using cheap short rollouts (3 representative
checkpoints x a few episodes each per grid point, not the full 6 x 100
protocol -- that full protocol is reserved for the final before/after
comparison in m1_run_held_out_eval.py).

Does NOT modify qoe_oran_framework/ (frozen). Must be run with cwd=framework/
(the RANEnv qoe-mapper checkpoint load path is relative -- see
docs/STAGE10_fullpower_reeval.md section 6 for the same bug class).

Loss per candidate config: standardized squared error between offline and
live (mean, std) of per_slice_sla_margin, summed over the 3 slices, using
the live std as the normalizer (so slices with naturally noisier live
margins don't dominate the loss). This targets the DISTRIBUTION, not just
the mean, per the M1 brief.

Usage (from repo root):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m1_fit_recalibration.py \
        --live-traces ../experiments/results/m1_recalibration/live_traces.json \
        --out ../experiments/results/m1_recalibration/fit_search.json
"""
import argparse
import json
import sys
import time
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
FIT_CHECKPOINTS = [256, 257, 258]  # perfect-live-adjacent, worst-live, perfect-live -- 3 representative seeds
FIT_SEEDS_PER_CKPT = 2
FIT_EPISODES_PER_SEED = 3
SLICE_IDS = ["embb", "urllc", "mmtc"]

GRID_BACKLOG_CAPACITY = [200.0, 500.0, 800.0, 1200.0, 2000.0, 3200.0]
GRID_DRIFT_COEF = [0.05, 0.1, 0.2]
GRID_OFFERED_VOLATILITY = [0.02, 0.04, 0.08]
GRID_AR1_COEF = [0.0, 0.5, 0.85]


def checkpoint_path(train_seed: int) -> str:
    if train_seed == 256:
        return CKPT_256
    return f"{CKPT_ROOT_NEW}/seed{train_seed}/dqn/offline_closed_loop/rep_0/checkpoint.pt"


def collect_margins(cfg, sd_for_slice, backlog_capacity, drift_coef, offered_volatility, ar1_coef, out_root):
    """Runs FIT_CHECKPOINTS under this candidate config, returns pooled
    per-slice margin lists from the resulting omega logs."""
    import os
    from qoe_oran_framework.omega_logger import OmegaLogger
    pooled = {s: [] for s in SLICE_IDS}

    def kpm_factory(seed):
        return RecalibratedClosedLoopKpmSource(
            seed=seed, gnb_ids=cfg.gnb_ids, slice_ids=list(cfg.slice_by_id),
            B=cfg.B, mean_offered_ratio=MEAN_OFFERED_RATIO,
            backlog_capacity=backlog_capacity, drift_coef=drift_coef,
            offered_volatility=offered_volatility, ar1_coef=ar1_coef,
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
               base_seed=8001, mode="fit_search", training=False, results_dir=out_dir,
               policy_factory=policy_factory, reward_mode="sla")
        for p in Path(out_dir).glob("dqn/fit_search/rep_*/omega_log.jsonl"):
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
    import math
    total = 0.0
    for s in SLICE_IDS:
        om, ostd = offline_stats[s]
        lm, lstd = live_stats[s]
        denom = max(lstd, 1e-3)
        total += ((om - lm) / denom) ** 2 + ((ostd - lstd) / denom) ** 2
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-traces", default=f"{REPO_ROOT}/experiments/results/m1_recalibration/live_traces.json")
    ap.add_argument("--out", default=f"{REPO_ROOT}/experiments/results/m1_recalibration/fit_search.json")
    ap.add_argument("--scratch-root", default="/tmp/m1_fit_scratch")
    args = ap.parse_args()

    with open(args.live_traces) as fh:
        live = json.load(fh)
    import numpy as np
    live_stats = {}
    for s in SLICE_IDS:
        v = np.array(live["pooled_margin_by_slice"][s])
        live_stats[s] = (float(v.mean()), float(v.std()))
    print(f"[m1-fit] live target (mean, std) per slice: {live_stats}")

    cfg = load_saclb_config(CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}

    grid = []
    for bc in GRID_BACKLOG_CAPACITY:
        for dc in GRID_DRIFT_COEF:
            for ov in GRID_OFFERED_VOLATILITY:
                for ar1 in GRID_AR1_COEF:
                    grid.append((bc, dc, ov, ar1))
    print(f"[m1-fit] grid size: {len(grid)}")

    results = []
    t_start = time.time()
    for i, (bc, dc, ov, ar1) in enumerate(grid):
        run_root = f"{args.scratch_root}/pt{i}"
        pooled = collect_margins(cfg, sd_for_slice, bc, dc, ov, ar1, run_root)
        offline_stats = {}
        ok = True
        for s in SLICE_IDS:
            if not pooled[s]:
                ok = False
                break
            v = np.array(pooled[s])
            offline_stats[s] = (float(v.mean()), float(v.std()))
        if not ok:
            continue
        l = loss(offline_stats, live_stats)
        results.append({
            "backlog_capacity": bc, "drift_coef": dc, "offered_volatility": ov, "ar1_coef": ar1,
            "offline_stats": offline_stats, "loss": l,
        })
        if (i + 1) % 20 == 0 or i == len(grid) - 1:
            elapsed = time.time() - t_start
            print(f"[m1-fit] {i+1}/{len(grid)} done, elapsed={elapsed:.0f}s, "
                  f"best_loss_so_far={min(r['loss'] for r in results):.3f}")

    results.sort(key=lambda r: r["loss"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"live_stats": live_stats, "results": results}, fh, indent=2)
    print(f"[m1-fit] wrote {out_path}")
    print("[m1-fit] top 5 configs by loss:")
    for r in results[:5]:
        print(f"  bc={r['backlog_capacity']} drift={r['drift_coef']} vol={r['offered_volatility']} "
              f"ar1={r['ar1_coef']} loss={r['loss']:.3f} stats={r['offline_stats']}")


if __name__ == "__main__":
    main()
