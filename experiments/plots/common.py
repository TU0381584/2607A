"""Shared plotting infrastructure for the campaign's Phase 4 figures.

Every figure script in experiments/plots/ imports from here so that color,
marker, and style stay IDENTICAL across all figures for the same arm (a
hard requirement -- see the campaign handover's Phase 4 spec). Reads only
from omega logs (and batch_manifest.jsonl for live-eval runs split across
health-checked restarts, see CAMPAIGN_LOG.md) -- no hand-authored figures.

Palette: the dataviz skill's validated default categorical palette,
slots 1-5 (blue/green/magenta/yellow/aqua), used in FIXED order -- this is
the order documented as passing every CVD/contrast gate on the adjacent-
pair list (bar/line/box charts, which is everything in this campaign's
required figure set -- none are scatter/choropleth, so the stricter
all-pairs 4-series cap doesn't apply here).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import matplotlib as mpl

# ---- IEEE single-column figure style --------------------------------------
IEEE_COLUMN_WIDTH_IN = 3.5
mpl.rcParams.update({
    "figure.figsize": (IEEE_COLUMN_WIDTH_IN, IEEE_COLUMN_WIDTH_IN * 0.75),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed as real text, not curves, for LaTeX-friendly PDFs
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

# ---- fixed per-arm style: color, marker, linestyle (never color-alone) ----
ARMS: List[str] = ["baseline", "dqn_sla", "a2c_sla", "dqn_qoe", "a2c_qoe"]

ARM_STYLE: Dict[str, dict] = {
    "baseline": {"color": "#2a78d6", "marker": "o", "linestyle": "-",  "label": "baseline (static)"},
    "dqn_sla":  {"color": "#008300", "marker": "s", "linestyle": "--", "label": "DQN (SLA reward)"},
    "a2c_sla":  {"color": "#e87ba4", "marker": "^", "linestyle": ":",  "label": "A2C (SLA reward)"},
    "dqn_qoe":  {"color": "#eda100", "marker": "D", "linestyle": "-.", "label": "DQN (QoE reward)"},
    "a2c_qoe":  {"color": "#1baf7a", "marker": "v", "linestyle": (0, (3, 1, 1, 1)), "label": "A2C (QoE reward)"},
}

SLICE_ORDER: List[str] = ["embb", "urllc", "mmtc"]
SLICE_STYLE: Dict[str, dict] = {
    "embb":  {"color": "#2a78d6", "hatch": None, "label": "eMBB"},
    "urllc": {"color": "#e34948", "hatch": "//", "label": "URLLC"},
    "mmtc":  {"color": "#eda100", "hatch": "..", "label": "mMTC"},
}


@dataclass
class OmegaRow:
    role: str
    method: str
    objective: str
    constraint: str
    evidence: dict
    limitation: str
    run_id: str
    episode: int
    step: int
    timestamp_s: float
    mode: str
    raw: dict


def read_omega_log(path: Path) -> Iterator[OmegaRow]:
    """Yields every record (step-level AND episode-rollup) from one
    omega_log.jsonl, in file order. Callers filter on row.step == -1 for
    rollups vs. >=1 for step records, per omega_logger.py's own convention."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield OmegaRow(
                role=row.get("role", ""), method=row.get("method", ""),
                objective=row.get("objective", ""), constraint=row.get("constraint", ""),
                evidence=row.get("evidence", {}), limitation=row.get("limitation", ""),
                run_id=row.get("run_id", ""), episode=row.get("episode", -1),
                step=row.get("step", -1), timestamp_s=row.get("timestamp_s", 0.0),
                mode=row.get("mode", ""), raw=row,
            )


def read_batch_manifest(path: Path) -> List[dict]:
    """Reads a live-eval rep's batch_manifest.jsonl (see
    experiments/scripts/run_live_eval_arm.py) -- needed to reconstruct
    cross-batch episode ordering, since each batch reseeds independently
    (see that script's docstring / CAMPAIGN_LOG.md)."""
    if not Path(path).exists():
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["batch_index"])


def arm_run_dir(results_root: Path, arm: str, reward_mode: str, seed: Optional[int] = None) -> Path:
    """Mirrors the directory convention both mc_runner.run_mc (offline) and
    run_live_eval_arm.py (live) use: results_root/arm/reward_mode/rep_*."""
    base = Path(results_root) / arm / reward_mode
    if seed is not None:
        return base / f"rep_seed{seed}"
    return base
