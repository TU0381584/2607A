"""Shared plotting infrastructure for paper #5's M2/M3 figures --
separate from common.py (paper #4's already-finalized figure set) so
nothing here can perturb CACS26's existing, submitted plots.
Reuses common.py's IEEE column-width/font rcParams unchanged (same
journal-figure sizing convention) and follows the dataviz skill's
validated default categorical palette: gat_ctde/independent_dqn/
single_agent_dqn take slots 1-3 (blue/orange/aqua), the specific triple
the palette's own documentation certifies passes the strict all-pairs
CVD/contrast gate (not just the looser adjacent-pairs one bar/line charts
normally need) -- chosen deliberately conservative since these 3 arms
are directly compared throughout the results section.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import numpy as np

from common import IEEE_COLUMN_WIDTH_IN  # noqa: F401 -- re-exported for figure scripts

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from m2_correctness_metrics import bootstrap_ci  # noqa: E402,F401 -- re-exported for figure scripts, not reimplemented

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
    "lines.markersize": 5,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

M2_ARM_ORDER: List[str] = ["gat_ctde", "independent_dqn", "single_agent_dqn"]
M2_ARM_STYLE: Dict[str, dict] = {
    "gat_ctde":         {"color": "#2a78d6", "marker": "o", "label": "GAT-CTDE (proposed)"},
    "independent_dqn":  {"color": "#eb6834", "marker": "^", "label": "Independent DQN"},
    "single_agent_dqn": {"color": "#1baf7a", "marker": "s", "label": "Single-agent DQN"},
}

M3_STYLE: Dict[str, dict] = {
    "centralized": {"color": "#2a78d6", "marker": "o", "label": "Centralised (GAT-CTDE)"},
    "federated":   {"color": "#eb6834", "marker": "D", "label": "Federated, no DP ($\\sigma$=0)"},
    "curve":       {"color": "#2a78d6", "marker": "o", "label": "Federated GAT-CTDE"},
}

# M4 shows all four arms together (unlike M3's Fig. 4, which only ever
# pairs gat_ctde against federated, so reusing independent_dqn's orange
# for federated there is harmless) -- federated needs its OWN slot here to
# avoid colliding with independent_dqn's orange. Slot 4 (yellow) of the
# dataviz skill's validated categorical order, next after gat_ctde/
# independent_dqn/single_agent_dqn's slots 1-3; gat_ctde/independent_dqn/
# single_agent_dqn keep their established colors unchanged for identity
# consistency across every figure in the paper.
M4_ARM_ORDER: List[str] = ["gat_ctde", "independent_dqn", "single_agent_dqn", "fl_gat_ctde_sigma0.0"]
M4_ARM_STYLE: Dict[str, dict] = {
    "gat_ctde":            M2_ARM_STYLE["gat_ctde"],
    "independent_dqn":     M2_ARM_STYLE["independent_dqn"],
    "single_agent_dqn":    M2_ARM_STYLE["single_agent_dqn"],
    "fl_gat_ctde_sigma0.0": {"color": "#eda100", "marker": "D", "label": "Federated"},
}

# Fixed (never themed) status roles, dataviz skill palette.md -- used for
# paired win/tie/loss outcome coloring, never for series identity, so a
# status color never impersonates an arm's own categorical hue.
STATUS_COLORS: Dict[str, str] = {
    "good": "#0ca30c",
    "critical": "#d03b3b",
    "neutral": "#898781",
}


def load_m2_campaign(results_path: str) -> Tuple[List[int], Dict[str, Dict[str, dict]]]:
    with open(results_path) as fh:
        data = json.load(fh)
    seed_groups = data["seed_groups"]
    all_seeds = [s for g in seed_groups for s in g]
    return all_seeds, data["results"]


def load_m3_campaign(results_path: str) -> Tuple[List[int], Dict[str, Dict[str, dict]]]:
    with open(results_path) as fh:
        data = json.load(fh)
    return data["seeds"], data["results"]


def eval_omega_path(campaign_dir: str, arm_or_tag: str, seed: int) -> Path:
    """Handles the one real path-shape difference between arms:
    single_agent_dqn nests under dqn/offline_eval/rep_0/ (mc_runner.run_mc's
    own convention) while gat_ctde/independent_dqn/the M3 FL arm write a
    flat eval/omega_log.jsonl directly (marl_training.run_episodes_marl's
    convention)."""
    base = Path(campaign_dir) / arm_or_tag / f"seed{seed}" / "eval"
    flat = base / "omega_log.jsonl"
    if flat.exists():
        return flat
    return base / "dqn" / "offline_eval" / "rep_0" / "omega_log.jsonl"
