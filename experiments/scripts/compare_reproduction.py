#!/usr/bin/env python3
"""Compares a fresh reproduce_paper5_full.sh run against the already-
committed M1-M4 results, to verify the pipeline is authentically
reproducible from a clean state -- not just "ran without crashing."

For M2/M3/M4 (fixed-seed RL training/eval), exact numeric equality is
the expected outcome absent any nondeterministic op in the training
path -- reported as such, not silently treated as "close enough" if it
diverges. M1 has no training in its path at all (grid search + held-out
eval against already-frozen checkpoints), so its outputs should also
match exactly.

Usage:
    python3 experiments/scripts/compare_reproduction.py \
        --repro-root experiments/results/reproduction_check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_correctness_metrics import per_seed_metrics  # noqa: E402

REPO_ROOT = Path("/home/kmanojp/oranslice_rig")


def compare_m1(repro_root: Path) -> None:
    print("=== M1: recalibration grid search + held-out eval ===")
    committed_dir = REPO_ROOT / "experiments/results/m1_recalibration"
    fresh_dir = repro_root / "m1"

    for tag in ["baseline", "recalibrated"]:
        c_path = committed_dir / f"held_out_{tag}" / f"compliance_{tag}.json"
        f_path = fresh_dir / f"held_out_{tag}" / f"compliance_{tag}.json"
        if not c_path.exists() or not f_path.exists():
            print(f"  [{tag}] SKIP -- missing ({'committed' if not c_path.exists() else 'fresh'} file absent)")
            continue
        with open(c_path) as fh:
            c = json.load(fh)["compliance"]
        with open(f_path) as fh:
            f = json.load(fh)["compliance"]
        mismatches = []
        for ckpt in c:
            if ckpt not in f:
                mismatches.append(f"seed {ckpt}: missing from fresh run")
                continue
            if c[ckpt]["pct"] != f[ckpt]["pct"]:
                mismatches.append(f"seed {ckpt}: committed={c[ckpt]['pct']}% fresh={f[ckpt]['pct']}%")
        if mismatches:
            print(f"  [{tag}] MISMATCH:")
            for m in mismatches:
                print(f"    {m}")
        else:
            print(f"  [{tag}] exact match, {len(c)} checkpoints: "
                  f"{ {k: v['pct'] for k, v in c.items()} }")


def compare_m2(repro_root: Path) -> None:
    print("=== M2: 30-seed x 3-arm campaign ===")
    committed_dir = REPO_ROOT / "experiments/results/m2_campaign"
    fresh_dir = repro_root / "m2_campaign"
    committed_results_path = committed_dir / "campaign_results.json"
    fresh_results_path = fresh_dir / "campaign_results.json"
    if not fresh_results_path.exists():
        print("  SKIP -- fresh m2_campaign/campaign_results.json not found (stage not complete yet)")
        return

    with open(committed_results_path) as fh:
        committed = json.load(fh)
    with open(fresh_results_path) as fh:
        fresh = json.load(fh)

    for arm in ["gat_ctde", "independent_dqn", "single_agent_dqn"]:
        c_by_seed = committed["results"].get(arm, {})
        f_by_seed = fresh["results"].get(arm, {})
        common_seeds = sorted(set(c_by_seed) & set(f_by_seed), key=int)
        if not common_seeds:
            print(f"  [{arm}] SKIP -- no common seeds yet")
            continue
        exact = 0
        mismatches = []
        for seed in common_seeds:
            c_val = c_by_seed[seed]["sla_compliance_all_slices"]
            f_val = f_by_seed[seed]["sla_compliance_all_slices"]
            if abs(c_val - f_val) < 1e-9:
                exact += 1
            else:
                mismatches.append(f"seed {seed}: committed={c_val:.4f} fresh={f_val:.4f}")
        print(f"  [{arm}] {exact}/{len(common_seeds)} seeds exact match"
              f"{' (all)' if exact == len(common_seeds) else ''}")
        for m in mismatches[:10]:
            print(f"    MISMATCH {m}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")


def compare_m3(repro_root: Path) -> None:
    print("=== M3: 10-seed x 5-sigma privacy sweep ===")
    committed_path = REPO_ROOT / "experiments/results/m3_campaign/campaign_results.json"
    fresh_path = repro_root / "m3_campaign" / "campaign_results.json"
    if not fresh_path.exists():
        print("  SKIP -- fresh m3_campaign/campaign_results.json not found (stage not complete yet)")
        return

    with open(committed_path) as fh:
        committed = json.load(fh)
    with open(fresh_path) as fh:
        fresh = json.load(fh)

    for sigma in sorted(committed["results"].keys(), key=float):
        if sigma not in fresh["results"]:
            print(f"  [sigma={sigma}] SKIP -- not in fresh run yet")
            continue
        c_by_seed = committed["results"][sigma]
        f_by_seed = fresh["results"][sigma]
        common_seeds = sorted(set(c_by_seed) & set(f_by_seed), key=int)
        exact = sum(1 for s in common_seeds
                    if abs(c_by_seed[s]["sla_compliance_all_slices"] - f_by_seed[s]["sla_compliance_all_slices"]) < 1e-9)
        print(f"  [sigma={sigma}] {exact}/{len(common_seeds)} seeds exact match")
        for s in common_seeds:
            c_val = c_by_seed[s]["sla_compliance_all_slices"]
            f_val = f_by_seed[s]["sla_compliance_all_slices"]
            if abs(c_val - f_val) >= 1e-9:
                print(f"    MISMATCH seed {s}: committed={c_val:.4f} fresh={f_val:.4f}")


def compare_m4(repro_root: Path) -> None:
    print("=== M4: disruption-resilience campaign ===")
    committed_path = REPO_ROOT / "experiments/results/m4_campaign/campaign_results.json"
    fresh_path = repro_root / "m4_campaign" / "campaign_results.json"
    if not fresh_path.exists():
        print("  SKIP -- fresh m4_campaign/campaign_results.json not found (stage not complete yet)")
        return

    with open(committed_path) as fh:
        committed = json.load(fh)["results"]
    with open(fresh_path) as fh:
        fresh = json.load(fh)["results"]

    common_keys = sorted(set(committed) & set(fresh))
    exact = 0
    mismatches = []
    for key in common_keys:
        c_val = committed[key]["sla_compliance_all_slices"]
        f_val = fresh[key]["sla_compliance_all_slices"]
        if abs(c_val - f_val) < 1e-9:
            exact += 1
        else:
            mismatches.append(f"{key}: committed={c_val:.4f} fresh={f_val:.4f}")
    print(f"  {exact}/{len(common_keys)} (arm/condition/seed) cells exact match"
          f"{' (all)' if exact == len(common_keys) and common_keys else ''}")
    for m in mismatches[:10]:
        print(f"    MISMATCH {m}")
    if len(mismatches) > 10:
        print(f"    ... and {len(mismatches) - 10} more")
    if len(common_keys) < len(fresh):
        print(f"  note: {len(fresh) - len(common_keys)} fresh cells have no committed counterpart to compare "
              f"(e.g. if the fresh run used different seeds/conditions)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repro-root", default=str(REPO_ROOT / "experiments/results/reproduction_check"))
    args = ap.parse_args()
    repro_root = Path(args.repro_root)

    compare_m1(repro_root)
    print()
    compare_m2(repro_root)
    print()
    compare_m3(repro_root)
    print()
    compare_m4(repro_root)


if __name__ == "__main__":
    main()
