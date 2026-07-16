"""Utilities for offline imitation warm-start datasets."""

import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _iter_csv_paths(source_path: Path):
    if source_path.is_file() and source_path.suffix.lower() == ".csv":
        yield source_path
        return

    if source_path.is_dir():
        for csv_path in sorted(source_path.rglob("*.csv")):
            yield csv_path


def load_state_action_dataset(
    source_path: str,
    expected_state_dim: int = 0,
    max_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a state-action dataset for behavior-cloning warm-start.

    The source can be:
    - A single CSV file containing columns `state_vector` and `action_index`
    - A directory tree with run logs containing those columns

    Returns:
        states: [N, expected_state_dim] float32
        actions: [N] int64
    """
    root = Path(source_path)
    if not root.exists():
        raise FileNotFoundError(f"Warm-start source path not found: {source_path}")

    inferred_state_dim = int(expected_state_dim)
    states: List[np.ndarray] = []
    actions: List[int] = []

    for csv_path in _iter_csv_paths(root):
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if "state_vector" not in headers or "action_index" not in headers:
                continue

            for row in reader:
                state_raw = (row.get("state_vector") or "").strip()
                action_raw = (row.get("action_index") or "").strip()
                if not state_raw or not action_raw:
                    continue

                try:
                    state_vec = np.array(json.loads(state_raw), dtype=np.float32)
                    action_idx = int(float(action_raw))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue

                if state_vec.ndim != 1:
                    continue

                if inferred_state_dim <= 0:
                    inferred_state_dim = int(state_vec.shape[0])

                if state_vec.shape[0] != inferred_state_dim:
                    continue

                states.append(state_vec)
                actions.append(action_idx)

                if max_samples > 0 and len(states) >= max_samples:
                    break

        if max_samples > 0 and len(states) >= max_samples:
            break

    if not states:
        raise ValueError(
            "No valid state-action samples found. "
            "Expected CSV rows containing state_vector and action_index columns."
        )

    return np.stack(states).astype(np.float32), np.array(actions, dtype=np.int64)
