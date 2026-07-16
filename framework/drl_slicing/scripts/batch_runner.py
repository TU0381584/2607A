#!/usr/bin/env python3
"""
Multi-seed experiment runner for reproducible SLA studies.
Orchestrates multiple runs with different random seeds and generates aggregate statistics.
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from oranslice_drl.config import load_experiment_config
from oranslice_drl.runner import run_experiment


def run_seeds(config_path: str, seeds: List[int], base_output_dir: str) -> None:
    config_base = load_experiment_config(config_path)
    base_output_dir_path = Path(base_output_dir)
    base_output_dir_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for seed in seeds:
        run_dir = base_output_dir_path / f"seed_{seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = load_experiment_config(config_path)
        config.random_seed = seed
        config.output.policy_json_path = str(run_dir / "rrmPolicy.json")
        config.output.run_log_csv = str(run_dir / "run_log.csv")

        print(f"[Seed {seed:03d}] Running {config_base.name}...")
        run_experiment(config)

        run_log_path = run_dir / "run_log.csv"
        if run_log_path.exists():
            with run_log_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if rows:
                    avg_reward = sum(float(r["reward"]) for r in rows) / len(rows)
                    total_violations = sum(int(r["sla_violations"]) for r in rows)
                    summary_rows.append({
                        "seed": seed,
                        "avg_reward": avg_reward,
                        "total_sla_violations": total_violations,
                        "num_steps": len(rows),
                    })
        print(f"[Seed {seed:03d}] Done.")

    if summary_rows:
        summary_path = base_output_dir_path / "summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved to: {summary_path}")
        print(f"\nResults across {len(seeds)} seeds:")
        for row in summary_rows:
            print(f"  Seed {row['seed']:03d}: avg_reward={row['avg_reward']:.3f}, violations={row['total_sla_violations']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed experiments for statistical rigor")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999],
                        help="Random seeds (default: 42 123 456 789 999)")
    parser.add_argument("--output-dir", default="batch_results", help="Output base directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_seeds(args.config, args.seeds, args.output_dir)


if __name__ == "__main__":
    main()
