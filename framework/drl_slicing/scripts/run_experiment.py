#!/usr/bin/env python3
import argparse

from oranslice_drl.config import load_experiment_config
from oranslice_drl.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ORANSlice DRL/baseline experiment loop")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
