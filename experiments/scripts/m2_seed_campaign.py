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

SEED_GROUPS = [list(range(900 + 3 * i, 903 + 3 * i)) for i in range(10)]  # 10 groups x 3 = 30 seeds
ALL_SEEDS = [s for group in SEED_GROUPS for s in group]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m2_campaign")
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--arms", nargs="+", default=["gat_ctde", "independent_dqn", "single_agent_dqn"])
    args = ap.parse_args()

    cfg = m2.load_saclb_config(m2.CONFIG_PATH)
    sd_for_slice = {sid: spec.sd for sid, spec in cfg.slice_by_id.items()}
    print(f"[campaign] {len(ALL_SEEDS)} seeds ({len(SEED_GROUPS)} groups of 3): {ALL_SEEDS}")

    t0 = time.time()
    all_results = {}
    for arm in args.arms:
        print(f"[campaign] === arm: {arm} ===")
        if arm == "gat_ctde":
            res = m2.run_gat_ctde_arm(cfg, sd_for_slice, ALL_SEEDS, args.train_episodes,
                                       args.eval_episodes, args.out_dir, arm)
        elif arm == "independent_dqn":
            res = m2.run_independent_dqn_arm(cfg, sd_for_slice, ALL_SEEDS, args.train_episodes,
                                              args.eval_episodes, args.out_dir, arm)
        elif arm == "single_agent_dqn":
            res = m2.run_single_agent_dqn_arm(cfg, sd_for_slice, ALL_SEEDS, args.train_episodes,
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
            json.dump({"seed_groups": SEED_GROUPS, "results": all_results}, fh, indent=2)

    print(f"[campaign] wrote {args.out_dir}/campaign_results.json, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
