#!/usr/bin/env python3
"""
Plot convergence-over-time for DRL runs.
Generates reward and SLA-violation trends with mean ± std across seeds.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot convergence curves from run logs")
    parser.add_argument("--output-dir", required=True, help="Directory containing dqn_train/a2c_train")
    parser.add_argument("--mode", default="train", choices=["train", "eval"], help="Run mode folder suffix")
    parser.add_argument("--ma-window", type=int, default=10, help="Moving-average window for reward curves")
    return parser.parse_args()


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series
    out = np.zeros_like(series, dtype=np.float64)
    csum = np.cumsum(np.insert(series.astype(np.float64), 0, 0.0))
    for idx in range(len(series)):
        start = max(0, idx - window + 1)
        count = idx - start + 1
        out[idx] = (csum[idx + 1] - csum[start]) / count
    return out


def _read_algo_curves(base_dir: Path, algo: str, mode: str, ma_window: int) -> Dict[str, np.ndarray]:
    algo_dir = base_dir / f"{algo}_{mode}"
    seed_dirs = sorted([path for path in algo_dir.glob("seed_*") if path.is_dir()])

    reward_series: List[np.ndarray] = []
    violation_series: List[np.ndarray] = []

    for seed_dir in seed_dirs:
        run_log = seed_dir / "run_log.csv"
        if not run_log.exists():
            continue

        rewards = []
        violations = []
        with run_log.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rewards.append(float(row["reward"]))
                violations.append(float(row["sla_violations"]))

        if rewards:
            reward_arr = np.array(rewards, dtype=np.float64)
            reward_series.append(_moving_average(reward_arr, ma_window))
            violation_series.append(np.array(violations, dtype=np.float64))

    if not reward_series:
        return {}

    min_len = min(len(series) for series in reward_series)
    reward_mat = np.stack([series[:min_len] for series in reward_series], axis=0)
    viol_mat = np.stack([series[:min_len] for series in violation_series], axis=0)

    return {
        "steps": np.arange(min_len),
        "reward_mean": reward_mat.mean(axis=0),
        "reward_std": reward_mat.std(axis=0),
        "viol_mean": viol_mat.mean(axis=0),
        "viol_std": viol_mat.std(axis=0),
        "num_seeds": np.array([reward_mat.shape[0]]),
    }


def _plot_curve(
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
):
    plt.plot(steps, mean, label=label, color=color, linewidth=2)
    plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)


def main() -> int:
    args = parse_args()
    base_dir = Path(args.output_dir)

    dqn = _read_algo_curves(base_dir, "dqn", args.mode, args.ma_window)
    a2c = _read_algo_curves(base_dir, "a2c", args.mode, args.ma_window)

    if not dqn and not a2c:
        raise RuntimeError(f"No run logs found in {base_dir}")

    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5.5))
    if dqn:
        _plot_curve(dqn["steps"], dqn["reward_mean"], dqn["reward_std"], f"DQN ({int(dqn['num_seeds'][0])} seeds)", "tab:blue")
    if a2c:
        _plot_curve(a2c["steps"], a2c["reward_mean"], a2c["reward_std"], f"A2C ({int(a2c['num_seeds'][0])} seeds)", "tab:orange")
    plt.title(f"Convergence Over Time: Reward (MA window={args.ma_window})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "convergence_reward_ma.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    if dqn:
        _plot_curve(dqn["steps"], dqn["viol_mean"], dqn["viol_std"], f"DQN ({int(dqn['num_seeds'][0])} seeds)", "tab:blue")
    if a2c:
        _plot_curve(a2c["steps"], a2c["viol_mean"], a2c["viol_std"], f"A2C ({int(a2c['num_seeds'][0])} seeds)", "tab:orange")
    plt.title("Convergence Over Time: SLA Violations")
    plt.xlabel("Step")
    plt.ylabel("Violations per step")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "convergence_violations.png", dpi=170)
    plt.close()

    print(f"✓ Convergence plots saved in: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
