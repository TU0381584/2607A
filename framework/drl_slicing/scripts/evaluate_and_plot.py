#!/usr/bin/env python3
"""
Evaluate DQN vs A2C and generate comparison plots.
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation and plot comparison")
    parser.add_argument("--config", required=True, help="Training config")
    parser.add_argument("--output-dir", default="results/phaseB_eval", help="Evaluation output folder")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 999, 1001, 1002, 1003, 1004, 1005],
        help="Seed list (use 10+ seeds for stable reporting)",
    )
    parser.add_argument("--ma-window", type=int, default=10, help="Moving-average window for reward reporting")
    parser.add_argument("--skip-run", action="store_true", help="Skip evaluation and only plot existing results")
    return parser.parse_args()


def run_eval(config: str, output_dir: str, seeds):
    cmd = [
        "python3",
        "scripts/train_drl.py",
        "--config", config,
        "--algorithms", "dqn", "a2c",
        "--seeds", *[str(seed) for seed in seeds],
        "--output-dir", output_dir,
        "--train",
        "--compare",
    ]
    subprocess.run(cmd, check=True)


def _moving_average(values, window: int):
    if not values:
        return []
    if window <= 1:
        return values
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _collect_reward_and_violation_summary(output_dir: str, ma_window: int):
    base = Path(output_dir)
    rows = []

    for algo in ["dqn", "a2c"]:
        algo_dir = base / f"{algo}_train"
        seed_dirs = sorted(path for path in algo_dir.glob("seed_*") if path.is_dir())
        if not seed_dirs:
            continue

        final_ma_rewards = []
        total_violations = []
        per_step_violations = []

        for seed_dir in seed_dirs:
            run_log = seed_dir / "run_log.csv"
            if not run_log.exists():
                continue

            import csv
            with run_log.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows_csv = list(reader)

            if not rows_csv:
                continue

            rewards = [float(row["reward"]) for row in rows_csv]
            violations = [float(row["sla_violations"]) for row in rows_csv]

            ma_rewards = _moving_average(rewards, ma_window)
            final_ma_rewards.append(ma_rewards[-1])
            total_violations.append(sum(violations))
            per_step_violations.append(sum(violations) / len(violations))

        if final_ma_rewards:
            import numpy as np
            rows.append(
                {
                    "algorithm": f"{algo}_train",
                    "num_seeds": len(final_ma_rewards),
                    "ma_reward_final_mean": float(np.mean(final_ma_rewards)),
                    "ma_reward_final_std": float(np.std(final_ma_rewards)),
                    "total_violations_mean": float(np.mean(total_violations)),
                    "total_violations_std": float(np.std(total_violations)),
                    "violations_per_step_mean": float(np.mean(per_step_violations)),
                    "violations_per_step_std": float(np.std(per_step_violations)),
                }
            )

    if rows:
        import csv
        out_dir = Path(output_dir) / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ma_reward_sla_report.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"✓ Moving-average reward + SLA report: {csv_path}")


def plot_comparison(output_dir: str, ma_window: int):
    comparison_path = Path(output_dir) / "comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {comparison_path}")

    with comparison_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    labels = []
    reward_means = []
    reward_stds = []
    violation_means = []
    violation_stds = []

    for key in ["dqn_train", "a2c_train"]:
        if key not in data:
            continue
        labels.append(key)
        reward_means.append(data[key]["reward_mean"])
        reward_stds.append(data[key]["reward_std"])
        violation_means.append(data[key]["violations_mean"])
        violation_stds.append(data[key]["violations_std"])

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, reward_means, yerr=reward_stds, capsize=6)
    plt.title("Reward Comparison (DQN vs A2C)")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, violation_means, yerr=violation_stds, capsize=6)
    plt.title("SLA Violation Comparison (DQN vs A2C)")
    plt.ylabel("Average SLA Violations")
    plt.tight_layout()
    plt.savefig(plots_dir / "violation_comparison.png", dpi=160)
    plt.close()

    _collect_reward_and_violation_summary(output_dir, ma_window)
    print(f"✓ Plots saved in: {plots_dir}")


def main() -> int:
    args = parse_args()
    if len(args.seeds) < 10:
        print("⚠ Warning: fewer than 10 seeds provided; results may be statistically weak.")
    if not args.skip_run:
        run_eval(args.config, args.output_dir, args.seeds)
    plot_comparison(args.output_dir, args.ma_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
