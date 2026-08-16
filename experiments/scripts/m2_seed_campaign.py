#!/usr/bin/env python3
"""M2 hardening (Block E, task 2): the real seed campaign, all three arms,
matched hyperparameters (see docs/PAPER5_M2_gat_ctde.md section 8 for the
gamma/epsilon-schedule parity fix this campaign depends on).

Fixed seed list, declared here (not generated at runtime) so the campaign
is exactly reproducible: 10 "seed groups" of 3 runs each = 30 independent
(fresh network init, independently-seeded env) runs per arm. A "run" is
not a re-evaluation of the same trained policy -- it is a full,
independent train+eval cycle, since training itself is not perfectly
reproducible even at a fixed seed (torch's own floating-point reduction
order under CPU multithreading -- documented in this project's own
saclb_offline_dqn.yaml comments). Grouping by seed group (not just
pooling all 30) is what lets the paired gat_ctde-vs-single_agent_dqn
comparison below actually pair like with like.

SEEDS: 30 total, groups of 3: [900,901,902], [903,904,905], ...,
[927,928,929].

Offline stress-regime environment only (see docs/PAPER5_M1_recalibration.md's
conclusion) -- no live claim anywhere in this campaign.

Merge-safe re-run support: if campaign_results.json already exists at
--out-dir, its "results" dict is loaded first and only the arms passed via
--arms are (re-)run and overwritten -- arms not listed are carried over
unchanged. Added to re-run gat_ctde alone after the GATEncoder
normalization fix (docs/PAPER5_M2_gat_ctde.md's collapse root-cause
section) without disturbing independent_dqn/single_agent_dqn's already-
valid results, which don't use GATEncoder and don't need re-running:
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m2_seed_campaign.py \
        --out-dir ../experiments/results/m2_campaign --arms gat_ctde

Usage (from repo root, cwd=framework/ required):
    cd framework && ../venv/bin/python3 \
        ../experiments/scripts/m2_seed_campaign.py \
        --out-dir ../experiments/results/m2_campaign
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
import m2_run_experiment as m2  # noqa: E402

DEFAULT_SEED_BASE = 900  # 10 groups x 3 = 30 seeds, base..base+29


def seed_groups_from(base: int):
    return [list(range(base + 3 * i, base + 3 + 3 * i)) for i in range(10)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m2_campaign")
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--arms", nargs="+", default=["gat_ctde", "independent_dqn", "single_agent_dqn"])
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                     help="First of 30 consecutive seeds (10 groups of 3), grouped identically to the "
                          "original 900-929 campaign -- override with a disjoint range (e.g. 1000) for an "
                          "independent-seed replication rather than a same-seed reproduction check.")
    ap.add_argument("--resume-seeds", action="store_true",
                     help="Skip seeds that already have a checkpoint+eval-log pair under --out-dir "
                          "(reload their compliance from the existing eval log instead of retraining). "
                          "Only safe when resuming an interrupted run of the SAME script/architecture -- "
                          "not a general artifact-reuse flag, see m2_run_experiment.py's "
                          "_reload_eval_compliance docstring.")
    args = ap.parse_args()

    seed_groups = seed_groups_from(args.seed_base)
    all_seeds = [s for group in seed_groups for s in group]

    cfg = m2.load_saclb_config(m2.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[campaign] {len(all_seeds)} seeds ({len(seed_groups)} groups of 3), base={args.seed_base}: {all_seeds}")

    out_path = Path(args.out_dir) / "campaign_results.json"
    all_results = {}
    if out_path.exists():
        with open(out_path) as fh:
            existing = json.load(fh)
        assert existing["seed_groups"] == seed_groups, "resumed campaign seed_groups mismatch"
        all_results = existing["results"]
        print(f"[campaign] merging into existing {out_path}: "
              f"carrying over {[a for a in all_results if a not in args.arms]}, "
              f"(re-)running {args.arms}")

    t0 = time.time()
    for arm in args.arms:
        print(f"[campaign] === arm: {arm} ===")
        if arm == "gat_ctde":
            res = m2.run_gat_ctde_arm(cfg, sd_for_slice, all_seeds, args.train_episodes,
                                       args.eval_episodes, args.out_dir, arm,
                                       resume_seeds=args.resume_seeds)
        elif arm == "independent_dqn":
            res = m2.run_independent_dqn_arm(cfg, sd_for_slice, all_seeds, args.train_episodes,
                                              args.eval_episodes, args.out_dir, arm)
        elif arm == "single_agent_dqn":
            res = m2.run_single_agent_dqn_arm(cfg, sd_for_slice, all_seeds, args.train_episodes,
                                               args.eval_episodes, args.out_dir, arm)
        else:
            raise ValueError(arm)
        all_results[arm] = res
        elapsed = time.time() - t0
        print(f"[campaign] arm {arm} done, cumulative elapsed {elapsed:.0f}s")
        # Write incrementally after each arm so a crash mid-campaign doesn't lose completed arms.
        out_path = Path(args.out_dir) / "campaign_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({"seed_groups": seed_groups, "results": all_results}, fh, indent=2)

    print(f"[campaign] wrote {args.out_dir}/campaign_results.json, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
