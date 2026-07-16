#!/usr/bin/env python3
"""
DRL experiment orchestrator for DRL and baseline controllers.

Supports:
- DQN/A2C/PPO training and inference modes
- Baseline controllers (rule_based, static, random, threshold_heuristic)
- Multi-seed sweeps with per-controller summaries
- Aggregate comparison reports (JSON + CSV)

Backward compatibility:
- --algorithms dqn a2c ppo --train still works
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from oranslice_drl.config import load_experiment_config
from oranslice_drl.runner import run_experiment


SUPPORTED_CONTROLLER_MODES = [
    "dqn_train",
    "dqn",
    "a2c_train",
    "a2c",
    "ppo_train",
    "ppo",
    "rule_based",
    "rule_based_drl_hybrid",
    "static",
    "random",
    "threshold_heuristic",
]

DEFAULT_SWEEP_CONTROLLERS = [
    "dqn_train",
    "a2c_train",
    "ppo_train",
    "rule_based",
    "static",
    "random",
    "threshold_heuristic",
]

INFERENCE_CONTROLLER_MODES = {"dqn", "a2c", "ppo"}


def _parse_controller_config_overrides(raw_items: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not raw_items:
        return overrides

    for item in raw_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --controller-config value '{item}'. "
                "Expected format: <controller_mode>=<config_path>"
            )
        mode, path = item.split("=", 1)
        mode = mode.strip()
        path = path.strip()
        if mode not in SUPPORTED_CONTROLLER_MODES:
            raise ValueError(f"Unknown controller mode in --controller-config: {mode}")
        if not path:
            raise ValueError(f"Missing config path for controller mode: {mode}")
        overrides[mode] = path

    return overrides


def _parse_controller_checkpoint_overrides(raw_items: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not raw_items:
        return overrides

    for item in raw_items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --controller-checkpoint value '{item}'. "
                "Expected format: <controller_mode>=<checkpoint_path>"
            )
        mode, checkpoint_path = item.split("=", 1)
        mode = mode.strip()
        checkpoint_path = checkpoint_path.strip()
        if mode not in SUPPORTED_CONTROLLER_MODES:
            raise ValueError(f"Unknown controller mode in --controller-checkpoint: {mode}")
        if not checkpoint_path:
            raise ValueError(f"Missing checkpoint path for controller mode: {mode}")
        overrides[mode] = checkpoint_path

    return overrides


def _discover_controllers_from_output_dir(output_dir: Path) -> List[str]:
    controllers: List[str] = []
    if not output_dir.exists():
        return controllers

    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "summary.csv").exists():
            controllers.append(child.name)

    return controllers


def _compute_seed_summary(run_log: Path, seed: int) -> Optional[Dict]:
    if not run_log.exists():
        return None

    with run_log.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return None

    rewards = [float(row["reward"]) for row in rows]
    violations = [float(row["sla_violations"]) for row in rows]

    return {
        "seed": seed,
        "avg_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "total_sla_violations": float(np.sum(violations)),
        "avg_sla_violations": float(np.mean(violations)),
        "num_steps": len(rows),
    }


def run_controller(
    controller_mode: str,
    config_path: str,
    output_dir: str,
    seeds: List[int],
    horizon_steps: Optional[int] = None,
    step_seconds: Optional[int] = None,
    continue_on_error: bool = True,
    offline_warm_start: bool = False,
    warm_start_source: Optional[str] = None,
    warm_start_epochs: Optional[int] = None,
    warm_start_batch_size: Optional[int] = None,
    warm_start_max_samples: Optional[int] = None,
    checkpoint_overrides: Optional[Dict[str, str]] = None,
    require_inference_checkpoints: bool = False,
    hybrid_dqn_checkpoint: Optional[str] = None,
    hybrid_a2c_checkpoint: Optional[str] = None,
    hybrid_base_weight: Optional[float] = None,
    hybrid_dqn_weight: Optional[float] = None,
    hybrid_a2c_weight: Optional[float] = None,
    hybrid_require_checkpoints: bool = False,
) -> Dict:
    """
    Run one controller mode across multiple seeds.

    Args:
        controller_mode: one of SUPPORTED_CONTROLLER_MODES
        config_path: path to base YAML config
        output_dir: directory for results
        seeds: list of random seeds
        horizon_steps: optional override of config horizon
        step_seconds: optional override of collector step_seconds
        continue_on_error: keep sweep running if one seed fails
        offline_warm_start: enable behavior-cloning warm-start before online training
        warm_start_source: optional dataset/log source path override
        warm_start_epochs: optional warm-start epochs override
        warm_start_batch_size: optional warm-start batch-size override
        warm_start_max_samples: optional sample cap for warm-start dataset
        checkpoint_overrides: optional mapping controller_mode->checkpoint_path
        require_inference_checkpoints: require checkpoints for dqn/a2c/ppo inference modes
        hybrid_dqn_checkpoint: optional DQN checkpoint used by rule_based_drl_hybrid
        hybrid_a2c_checkpoint: optional A2C checkpoint used by rule_based_drl_hybrid
        hybrid_base_weight: optional baseline weight for rule_based_drl_hybrid
        hybrid_dqn_weight: optional DQN weight for rule_based_drl_hybrid
        hybrid_a2c_weight: optional A2C weight for rule_based_drl_hybrid
        hybrid_require_checkpoints: require both hybrid checkpoints when hybrid mode is used

    Returns:
        summary stats for the controller
    """
    if controller_mode not in SUPPORTED_CONTROLLER_MODES:
        raise ValueError(f"Unknown controller mode: {controller_mode}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    controller_dir = output_path / controller_mode
    controller_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    failures = []

    print(f"\n{'=' * 60}")
    print(f"Running controller mode: {controller_mode}")
    print(f"{'=' * 60}")

    for seed in seeds:
        run_dir = controller_dir / f"seed_{seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = load_experiment_config(config_path)
        config.random_seed = seed
        config.controller.mode = controller_mode

        if horizon_steps is not None:
            config.horizon_steps = horizon_steps
        if step_seconds is not None:
            config.collector.step_seconds = step_seconds

        if offline_warm_start:
            config.controller.offline_warm_start = True
        if warm_start_source is not None:
            config.controller.warm_start_dataset_path = warm_start_source
        if warm_start_epochs is not None:
            config.controller.warm_start_epochs = warm_start_epochs
        if warm_start_batch_size is not None:
            config.controller.warm_start_batch_size = warm_start_batch_size
        if warm_start_max_samples is not None:
            config.controller.warm_start_max_samples = warm_start_max_samples

        if hybrid_dqn_checkpoint is not None:
            config.controller.hybrid_dqn_checkpoint_path = hybrid_dqn_checkpoint
        if hybrid_a2c_checkpoint is not None:
            config.controller.hybrid_a2c_checkpoint_path = hybrid_a2c_checkpoint
        if hybrid_base_weight is not None:
            config.controller.hybrid_base_weight = float(hybrid_base_weight)
        if hybrid_dqn_weight is not None:
            config.controller.hybrid_dqn_weight = float(hybrid_dqn_weight)
        if hybrid_a2c_weight is not None:
            config.controller.hybrid_a2c_weight = float(hybrid_a2c_weight)
        if hybrid_require_checkpoints:
            config.controller.hybrid_require_checkpoints = True

        if checkpoint_overrides and controller_mode in checkpoint_overrides:
            config.controller.checkpoint_path = checkpoint_overrides[controller_mode]

        if require_inference_checkpoints and controller_mode in INFERENCE_CONTROLLER_MODES:
            config.controller.require_checkpoint = True
            if not config.controller.checkpoint_path.strip():
                raise ValueError(
                    "Inference checkpoint required but none provided for controller mode "
                    f"'{controller_mode}'. Use --controller-checkpoint {controller_mode}=<path>."
                )

        config.name = f"{config.name}_{controller_mode}"
        config.output.policy_json_path = str(run_dir / "rrmPolicy.json")
        config.output.run_log_csv = str(run_dir / "run_log.csv")

        print(f"[Seed {seed:03d}] Running {config.name}...")
        try:
            run_experiment(config)
            print(f"[Seed {seed:03d}] Complete")
        except Exception as error:
            print(f"[Seed {seed:03d}] Error: {error}")
            failures.append({"seed": seed, "error": str(error)})
            if not continue_on_error:
                raise
            continue

        run_log = run_dir / "run_log.csv"
        stats = _compute_seed_summary(run_log, seed)
        if stats:
            summaries.append(stats)

    if not summaries:
        return {
            "controller": controller_mode,
            "num_seeds": 0,
            "num_failures": len(failures),
        }

    summary_path = controller_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nSummary saved: {summary_path}")

    rewards = [float(entry["avg_reward"]) for entry in summaries]
    total_rewards = [float(entry["total_reward"]) for entry in summaries]
    violations = [float(entry["total_sla_violations"]) for entry in summaries]
    violations_per_step = [float(entry["avg_sla_violations"]) for entry in summaries]

    stats = {
        "controller": controller_mode,
        "num_seeds": len(summaries),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_total_mean": float(np.mean(total_rewards)),
        "reward_total_std": float(np.std(total_rewards)),
        "violations_mean": float(np.mean(violations)),
        "violations_std": float(np.std(violations)),
        "violations_per_step_mean": float(np.mean(violations_per_step)),
        "violations_per_step_std": float(np.std(violations_per_step)),
    }
    if failures:
        stats["num_failures"] = len(failures)

    return stats


def run_algorithm(
    algorithm: str,
    config_path: str,
    output_dir: str,
    seeds: List[int],
    train: bool = True,
    horizon_steps: Optional[int] = None,
    step_seconds: Optional[int] = None,
    continue_on_error: bool = True,
    offline_warm_start: bool = False,
    warm_start_source: Optional[str] = None,
    warm_start_epochs: Optional[int] = None,
    warm_start_batch_size: Optional[int] = None,
    warm_start_max_samples: Optional[int] = None,
    checkpoint_overrides: Optional[Dict[str, str]] = None,
    require_inference_checkpoints: bool = False,
) -> Dict:
    """Backward-compatible wrapper for legacy algorithm-based invocation."""
    if algorithm == "dqn":
        controller_mode = "dqn_train" if train else "dqn"
    elif algorithm == "a2c":
        controller_mode = "a2c_train" if train else "a2c"
    elif algorithm == "ppo":
        controller_mode = "ppo_train" if train else "ppo"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return run_controller(
        controller_mode=controller_mode,
        config_path=config_path,
        output_dir=output_dir,
        seeds=seeds,
        horizon_steps=horizon_steps,
        step_seconds=step_seconds,
        continue_on_error=continue_on_error,
        offline_warm_start=offline_warm_start,
        warm_start_source=warm_start_source,
        warm_start_epochs=warm_start_epochs,
        warm_start_batch_size=warm_start_batch_size,
        warm_start_max_samples=warm_start_max_samples,
        checkpoint_overrides=checkpoint_overrides,
        require_inference_checkpoints=require_inference_checkpoints,
    )


def _read_controller_summary(summary_file: Path) -> Optional[Dict]:
    with summary_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return None

    rewards = [float(row.get("avg_reward", 0.0)) for row in rows]
    steps = [float(row.get("num_steps", 0.0)) for row in rows]

    if "total_reward" in rows[0]:
        total_rewards = [float(row.get("total_reward", 0.0)) for row in rows]
    else:
        total_rewards = [rewards[idx] * steps[idx] for idx in range(len(rows))]

    if "total_sla_violations" in rows[0]:
        violations = [float(row.get("total_sla_violations", 0.0)) for row in rows]
    else:
        violations = [float(row.get("violations", 0.0)) for row in rows]

    if "avg_sla_violations" in rows[0]:
        violations_per_step = [float(row.get("avg_sla_violations", 0.0)) for row in rows]
    else:
        violations_per_step = [
            (violations[idx] / steps[idx]) if steps[idx] > 0 else 0.0 for idx in range(len(rows))
        ]

    return {
        "num_seeds": len(rows),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_total_mean": float(np.mean(total_rewards)),
        "reward_total_std": float(np.std(total_rewards)),
        "violations_mean": float(np.mean(violations)),
        "violations_std": float(np.std(violations)),
        "violations_per_step_mean": float(np.mean(violations_per_step)),
        "violations_per_step_std": float(np.std(violations_per_step)),
    }


def compare_controllers(base_output_dir: str, controllers: Optional[Iterable[str]] = None) -> Dict[str, Dict]:
    """Create aggregate comparison outputs for all requested controllers."""
    output_path = Path(base_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_json = output_path / "comparison.json"
    comparison_csv = output_path / "comparison_summary.csv"

    if controllers is None:
        controllers_to_scan = _discover_controllers_from_output_dir(output_path)
    else:
        controllers_to_scan = list(dict.fromkeys(controllers))

    results: Dict[str, Dict] = {}
    csv_rows = []

    for controller in controllers_to_scan:
        summary_file = output_path / controller / "summary.csv"
        if not summary_file.exists():
            continue

        stats = _read_controller_summary(summary_file)
        if not stats:
            continue

        results[controller] = stats
        csv_rows.append({"controller": controller, **stats})

    if not results:
        print("No results found to compare.")
        return {}

    with comparison_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    csv_headers = [
        "controller",
        "num_seeds",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "reward_total_mean",
        "reward_total_std",
        "violations_mean",
        "violations_std",
        "violations_per_step_mean",
        "violations_per_step_std",
    ]
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nComparison JSON: {comparison_json}")
    print(f"Comparison CSV : {comparison_csv}")
    print("\nSummary:")
    for controller, stats in results.items():
        print(f"\n{controller}:")
        print(f"  Reward: {stats['reward_mean']:.3f} +/- {stats['reward_std']:.3f}")
        print(f"  Violations: {stats['violations_mean']:.1f} +/- {stats['violations_std']:.1f}")

    return results


def compare_algorithms(base_output_dir: str) -> None:
    """Backward-compatible wrapper: compare legacy DQN/A2C modes if present."""
    compare_controllers(base_output_dir)


def _resolve_controller_modes(args: argparse.Namespace) -> List[str]:
    if args.controllers:
        return args.controllers

    if args.algorithms:
        resolved: List[str] = []
        for algorithm in args.algorithms:
            if algorithm == "dqn":
                resolved.append("dqn_train" if args.train else "dqn")
            elif algorithm == "a2c":
                resolved.append("a2c_train" if args.train else "a2c")
            elif algorithm == "ppo":
                resolved.append("ppo_train" if args.train else "ppo")
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        return resolved

    if args.config:
        return DEFAULT_SWEEP_CONTROLLERS

    return []


def _validate_controller_modes(modes: Iterable[str]) -> None:
    for mode in modes:
        if mode not in SUPPORTED_CONTROLLER_MODES:
            raise ValueError(f"Unknown controller mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare DRL/baseline controllers")
    parser.add_argument("--config", required=False, help="Base YAML config")

    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=SUPPORTED_CONTROLLER_MODES,
        help="Controller modes to run (preferred)",
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["dqn", "a2c", "ppo"],
        help="Legacy selector for DRL algorithms (mapped to controller modes)",
    )

    parser.add_argument(
        "--controller-config",
        action="append",
        default=[],
        help="Optional per-controller config override: <mode>=<config_path>",
    )

    parser.add_argument(
        "--controller-checkpoint",
        action="append",
        default=[],
        help="Optional per-controller checkpoint override: <mode>=<checkpoint_path>",
    )

    parser.add_argument(
        "--require-inference-checkpoints",
        action="store_true",
        help="Require checkpoints for inference controllers (dqn/a2c/ppo)",
    )

    parser.add_argument(
        "--hybrid-dqn-checkpoint",
        default=None,
        help="Checkpoint path for DQN policy inside rule_based_drl_hybrid",
    )

    parser.add_argument(
        "--hybrid-a2c-checkpoint",
        default=None,
        help="Checkpoint path for A2C policy inside rule_based_drl_hybrid",
    )

    parser.add_argument(
        "--hybrid-base-weight",
        type=float,
        default=None,
        help="Weight for baseline action in rule_based_drl_hybrid",
    )

    parser.add_argument(
        "--hybrid-dqn-weight",
        type=float,
        default=None,
        help="Weight for DQN action in rule_based_drl_hybrid",
    )

    parser.add_argument(
        "--hybrid-a2c-weight",
        type=float,
        default=None,
        help="Weight for A2C action in rule_based_drl_hybrid",
    )

    parser.add_argument(
        "--hybrid-require-checkpoints",
        action="store_true",
        help="Require DQN/A2C checkpoints when using rule_based_drl_hybrid",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds",
    )

    parser.add_argument(
        "--output-dir",
        default="results/phaseB_eval",
        help="Base output directory",
    )

    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help="Optional horizon_steps override",
    )

    parser.add_argument(
        "--step-seconds",
        type=int,
        default=None,
        help="Optional collector step_seconds override for live Prometheus polling",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Legacy flag used with --algorithms (dqn/a2c/ppo)",
    )

    parser.add_argument(
        "--offline-warm-start",
        action="store_true",
        help="Enable offline behavior-cloning warm-start before online training",
    )

    parser.add_argument(
        "--warm-start-source",
        default=None,
        help="CSV file or directory containing state_vector/action_index data",
    )

    parser.add_argument(
        "--warm-start-epochs",
        type=int,
        default=None,
        help="Optional warm-start epochs override",
    )

    parser.add_argument(
        "--warm-start-batch-size",
        type=int,
        default=None,
        help="Optional warm-start batch size override",
    )

    parser.add_argument(
        "--warm-start-max-samples",
        type=int,
        default=None,
        help="Optional cap on warm-start samples (0 or unset means all)",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate aggregate comparison for output-dir",
    )

    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help="Continue remaining runs if one seed/controller fails (default: enabled)",
    )
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Stop immediately when one seed/controller fails",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.config and not args.compare:
        print("Nothing to do. Provide --config to run and/or --compare to aggregate.")
        return 2

    controller_overrides = _parse_controller_config_overrides(args.controller_config)
    checkpoint_overrides = _parse_controller_checkpoint_overrides(args.controller_checkpoint)

    did_run = False
    resolved_controllers: List[str] = []

    if args.config:
        resolved_controllers = _resolve_controller_modes(args)
        _validate_controller_modes(resolved_controllers)

        if not (args.controllers or args.algorithms):
            print(
                "No --controllers/--algorithms provided. "
                "Running default sweep: "
                + ", ".join(resolved_controllers)
            )

        for controller_mode in resolved_controllers:
            controller_cfg = controller_overrides.get(controller_mode, args.config)
            run_controller(
                controller_mode=controller_mode,
                config_path=controller_cfg,
                output_dir=args.output_dir,
                seeds=args.seeds,
                horizon_steps=args.horizon_steps,
                step_seconds=args.step_seconds,
                continue_on_error=args.continue_on_error,
                offline_warm_start=args.offline_warm_start,
                warm_start_source=args.warm_start_source,
                warm_start_epochs=args.warm_start_epochs,
                warm_start_batch_size=args.warm_start_batch_size,
                warm_start_max_samples=args.warm_start_max_samples,
                checkpoint_overrides=checkpoint_overrides,
                require_inference_checkpoints=args.require_inference_checkpoints,
                hybrid_dqn_checkpoint=args.hybrid_dqn_checkpoint,
                hybrid_a2c_checkpoint=args.hybrid_a2c_checkpoint,
                hybrid_base_weight=args.hybrid_base_weight,
                hybrid_dqn_weight=args.hybrid_dqn_weight,
                hybrid_a2c_weight=args.hybrid_a2c_weight,
                hybrid_require_checkpoints=args.hybrid_require_checkpoints,
            )
        did_run = True

    if args.compare or did_run:
        controllers_for_compare: Optional[List[str]] = None
        if resolved_controllers:
            controllers_for_compare = resolved_controllers
        elif args.controllers:
            controllers_for_compare = args.controllers
        compare_controllers(args.output_dir, controllers_for_compare)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
