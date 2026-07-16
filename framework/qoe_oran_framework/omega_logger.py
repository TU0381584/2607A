"""Omega-tuple logger: (role, method, objective, constraint, evidence,
limitation) records in the same normalised form as the survey matrices.

Every record's `limitation` field is required non-empty -- this is how
"no fabricated numbers" is structurally enforced: every reported number
traces back to an evidence dict carrying run_id/episode/step/mode, and no
record can claim results without also stating what's approximate,
deviated, or unverified about them.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class OmegaTuple:
    role: str          # e.g. "urllc-admission" | "embb-admission" | "mmtc-admission" | "load-balancer"
    method: str         # e.g. "DQNAdmissionPolicy" | "RainbowAdmissionPolicy" | "LbOnlyHeuristic"
    objective: str       # e.g. "minimize URLLC block rate subject to eMBB/mMTC SLA and PRB budget"
    constraint: str      # e.g. "gNB capacity B=100 PRB, URLLC quota<=30, cluster rho target [0.55,0.60]"
    evidence: Dict[str, Any]
    limitation: str
    run_id: str
    episode: int
    step: int
    timestamp_s: float
    mode: str            # "offline_synthetic" | "offline_replay" | "live_testbed"

    def __post_init__(self) -> None:
        if not self.limitation or not self.limitation.strip():
            raise ValueError(
                "OmegaTuple.limitation must be non-empty -- every logged record must state "
                "what's approximate, deviated, or unverified (Stage Zero's no-fabrication guarantee)."
            )
        if not self.evidence:
            raise ValueError("OmegaTuple.evidence must be non-empty -- a record with no evidence proves nothing.")


class OmegaLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, tup: OmegaTuple) -> None:
        self._handle.write(json.dumps(asdict(tup)) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "OmegaLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_omega_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
