"""Shared plotting infrastructure for paper #5's M2/M3 figures --
separate from common.py (paper #4's already-finalized figure set) so
nothing here can perturb paper_conf's existing, submitted plots.
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
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import numpy as np

from common import IEEE_COLUMN_WIDTH_IN  # noqa: F401 -- re-exported for figure scripts

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
    "centralized": {"color": "#2a78d6", "marker": "o", "label": "Centralized (GAT-CTDE)"},
    "federated":   {"color": "#eb6834", "marker": "D", "label": "Federated, no DP ($\\sigma$=0)"},
    "curve":       {"color": "#2a78d6", "marker": "o", "label": "Federated GAT-CTDE"},
}


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


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
