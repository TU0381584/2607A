import json
import os
import tempfile
from typing import Dict, List

from .config import SliceTarget
from .types import SliceAction


def _slice_json_entry(target: SliceTarget, action: SliceAction) -> dict:
    entry = {
        "sst": target.sst,
        "dedicated_ratio": int(action.dedicated_ratio),
        "min_ratio": int(action.min_ratio),
        "max_ratio": int(action.max_ratio),
    }
    if target.sd >= 0:
        entry["sd"] = int(target.sd)
    return entry


def write_rrm_policy_json(policy_path: str, actions: List[SliceAction], targets: Dict[str, SliceTarget]) -> None:
    action_by_slice = {item.slice_id: item for item in actions}
    payload = {
        "rrmPolicyRatio": [
            _slice_json_entry(targets[slice_id], action_by_slice[slice_id])
            for slice_id in targets.keys()
            if slice_id in action_by_slice
        ]
    }
    os.makedirs(os.path.dirname(policy_path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        tmp_name = handle.name
    os.replace(tmp_name, policy_path)
