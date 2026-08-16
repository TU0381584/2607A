#!/usr/bin/env python3
"""M4: full disruption-resilience campaign, orchestrating m4_run_experiment.py
across (arm, disruption kind, severity, seed). Merge-safe/resumable, same
pattern as m2_seed_campaign.py: writes campaign_results.json incrementally,
skips (arm, kind, severity, seed) cells already present unless --force.

Seeds default to the first 10 of M2's 30-seed list (900-909), matching
M3's own choice -- every arm (including the federated no-DP checkpoint,
which M3 only trained for these 10) has a valid checkpoint for this range.

Usage:
    python3 experiments/scripts/m4_seed_campaign.py \
        --seeds 900 901 902 --arms gat_ctde --kinds dropout \
        --severities 1 --out-dir experiments/results/m4_campaign
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/kmanojp/oranslice_rig/experiments/scripts")
import m4_run_experiment as m4  # noqa: E402

DEFAULT_SEEDS = list(range(900, 910))  # matches M3's own 10-seed subset
DEFAULT_ARMS = ["gat_ctde", "independent_dqn", "single_agent_dqn", "fl_gat_ctde_sigma0.0"]
DEFAULT_KINDS = ["dropout", "spike", "churn"]
DEFAULT_SEVERITIES = [1, 2, 3]


def applicable_kinds(arm: str, kinds) -> list:
    if arm == "single_agent_dqn":
        return [k for k in kinds if k != "churn"]
    return list(kinds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=DEFAULT_ARMS)
    ap.add_argument("--kinds", nargs="+", default=DEFAULT_KINDS, choices=DEFAULT_KINDS)
    ap.add_argument("--severities", type=int, nargs="+", default=DEFAULT_SEVERITIES, choices=[1, 2, 3])
    ap.add_argument("--out-dir", default="/home/kmanojp/oranslice_rig/experiments/results/m4_campaign")
    ap.add_argument("--m2-campaign-dir", default=m4.DEFAULT_M2_CAMPAIGN_DIR,
                     help="Where to load M2 checkpoints from -- override to point at a fresh reproduction.")
    ap.add_argument("--m3-campaign-dir", default=m4.DEFAULT_M3_CAMPAIGN_DIR,
                     help="Where to load the M3 federated checkpoint from -- override for a fresh reproduction.")
    ap.add_argument("--force", action="store_true", help="Re-run cells even if already present in campaign_results.json")
    args = ap.parse_args()

    out_path = Path(args.out_dir) / "campaign_results.json"
    all_results = {}
    if out_path.exists():
        with open(out_path) as fh:
            all_results = json.load(fh)["results"]

    cells = []
    for arm in args.arms:
        for kind in applicable_kinds(arm, args.kinds):
            for severity in args.severities:
                for seed in args.seeds:
                    cells.append((arm, kind, severity, seed))

    print(f"[m4-campaign] {len(cells)} (arm, kind, severity, seed) cells queued "
          f"({len(args.seeds)} seeds x {len(args.arms)} arms x up to {len(args.kinds)} kinds x "
          f"{len(args.severities)} severities, churn skipped for single_agent_dqn)")

    t0 = time.time()
    n_run = n_skipped = 0
    for arm, kind, severity, seed in cells:
        severity_label = f"{kind}_sev{severity}"
        key = f"{arm}/{severity_label}/{seed}"
        if not args.force and key in all_results:
            n_skipped += 1
            continue

        summary = m4.run_condition(arm, seed, kind, severity, args.out_dir,
                                    args.m2_campaign_dir, args.m3_campaign_dir)
        all_results[key] = {
            "arm": arm, "kind": kind, "severity": severity, "seed": seed,
            "sla_compliance_all_slices": summary["sla_compliance_all_slices"],
            "n_episodes": summary["n_episodes"],
        }
        n_run += 1
        elapsed = time.time() - t0
        print(f"[m4-campaign] ({n_run+n_skipped}/{len(cells)}) {key}: "
              f"compliance={summary['sla_compliance_all_slices']:.4f} (cumulative {elapsed:.0f}s)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({"seeds": args.seeds, "results": all_results}, fh, indent=2)

    print(f"[m4-campaign] done: {n_run} run, {n_skipped} skipped (already present), "
          f"wrote {out_path}, total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
