#!/usr/bin/env python3
"""Build a compact offline warm-start dataset from run logs."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from oranslice_drl.offline_warmstart import load_state_action_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline warm-start state-action dataset")
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file or directory containing run logs with state_vector/action_index",
    )
    parser.add_argument(
        "--output",
        default="data/offline_warmstart_dataset.csv",
        help="Output dataset CSV path",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=0,
        help="Expected state dimension (0 = infer automatically)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to export (0 = all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    states, actions = load_state_action_dataset(
        source_path=args.input,
        expected_state_dim=max(0, int(args.state_dim)),
        max_samples=max(0, int(args.max_samples)),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action_index", "state_vector"])
        writer.writeheader()
        for state_vec, action_idx in zip(states, actions):
            writer.writerow(
                {
                    "action_index": int(action_idx),
                    "state_vector": json.dumps(state_vec.tolist()),
                }
            )

    print(f"Wrote {len(actions)} samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
