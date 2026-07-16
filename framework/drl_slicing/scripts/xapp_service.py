#!/usr/bin/env python3
"""
Run a DRL policy as an xApp-like service process.

This wrapper runs the existing experiment loop with a selected algorithm/mode
and xApp-specific outputs, so DQN and A2C can be launched as separate services.
"""
import argparse
import time
from pathlib import Path

from oranslice_drl.config import load_experiment_config
from oranslice_drl.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DRL policy as xApp service")
    parser.add_argument("--config", required=True, help="Base YAML config path")
    parser.add_argument("--algorithm", required=True, choices=["dqn", "a2c"], help="Policy algorithm")
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    parser.add_argument("--repeat", type=int, default=1, help="How many runs to execute (0 = infinite)")
    parser.add_argument("--name", default="", help="Service name override")
    parser.add_argument(
        "--policy-json-path",
        default="",
        help="Optional explicit path for policy output (use for live gNB control)",
    )
    parser.add_argument(
        "--run-log-csv",
        default="",
        help="Optional explicit path for run log CSV",
    )
    return parser.parse_args()


def _apply_overrides(
    config,
    algorithm: str,
    train: bool,
    service_name: str,
    policy_json_path: str,
    run_log_csv: str,
) -> None:
    config.name = service_name
    if algorithm == "dqn":
        config.controller.mode = "dqn_train" if train else "dqn"
    elif algorithm == "a2c":
        config.controller.mode = "a2c_train" if train else "a2c"

    if not policy_json_path and not run_log_csv:
        out_dir = Path("results") / "xapps" / service_name
        out_dir.mkdir(parents=True, exist_ok=True)
        config.output.policy_json_path = str(out_dir / "rrmPolicy.json")
        config.output.run_log_csv = str(out_dir / "run_log.csv")
    else:
        if policy_json_path:
            config.output.policy_json_path = policy_json_path
        if run_log_csv:
            config.output.run_log_csv = run_log_csv

        Path(config.output.policy_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.output.run_log_csv).parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    service_name = args.name or f"xapp-{args.algorithm}-{'train' if args.train else 'infer'}"

    run_count = 0
    while True:
        run_count += 1
        cfg = load_experiment_config(args.config)
        _apply_overrides(
            cfg,
            args.algorithm,
            args.train,
            service_name,
            args.policy_json_path,
            args.run_log_csv,
        )
        print(f"[{service_name}] run {run_count} | mode={cfg.controller.mode} | horizon={cfg.horizon_steps}")
        run_experiment(cfg)

        if args.repeat > 0 and run_count >= args.repeat:
            break

        time.sleep(max(cfg.collector.step_seconds, 1))

    print(f"[{service_name}] completed {run_count} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
