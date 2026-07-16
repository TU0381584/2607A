#!/usr/bin/env python3
"""
Plot slice-specific and slice-wide congestion KPIs from controller run logs.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot slice-specific and slice-wide congestion results")
    parser.add_argument("--output-dir", required=True, help="Directory containing controller run outputs")
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["dqn", "a2c", "rule_based"],
        help="Controller folders to load",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed to plot")
    parser.add_argument("--ma-window", type=int, default=10, help="Moving-average window")
    parser.add_argument("--shock-start", type=int, default=200, help="First congestion shock step")
    parser.add_argument("--shock-interval", type=int, default=200, help="Congestion shock period")
    parser.add_argument("--shock-duration", type=int, default=20, help="Congestion shock duration")
    parser.add_argument("--plots-dir", default="", help="Output plot directory (defaults to <output-dir>/plots)")
    parser.add_argument(
        "--allow-non-live",
        action="store_true",
        help="Allow plotting run logs that are not marked as live Prometheus traces",
    )
    return parser.parse_args()


def _display_name(controller: str) -> str:
    names = {
        "dqn": "DQN",
        "dqn_train": "DQN",
        "a2c": "A2C",
        "a2c_train": "A2C",
        "rule_based": "Rule-Based Baseline",
        "rule_based_drl_hybrid": "Rule-Based + DRL Hybrid",
        "threshold_heuristic": "Threshold Baseline",
    }
    return names.get(controller, controller)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    out = np.zeros_like(values, dtype=np.float64)
    csum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        count = idx - start + 1
        out[idx] = (csum[idx + 1] - csum[start]) / count
    return out


def _load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _assert_live_rows(rows: List[dict], csv_path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Run log has no rows: {csv_path}")

    for required in ("collector_mode", "live_trace_only"):
        if required not in rows[0]:
            raise RuntimeError(
                f"Run log missing required provenance column '{required}': {csv_path}. "
                "Re-run with updated live pipeline before plotting."
            )

    for idx, row in enumerate(rows, start=1):
        collector_mode = row.get("collector_mode", "").strip().lower()
        live_trace_only = row.get("live_trace_only", "")
        if collector_mode != "prometheus" or not _is_truthy(live_trace_only):
            raise RuntimeError(
                f"Non-live row detected in {csv_path} at data row {idx}: "
                f"collector_mode={collector_mode!r}, live_trace_only={live_trace_only!r}"
            )


def _extract_slice_keys(header: List[str]) -> List[str]:
    keys = []
    for column in header:
        if column.startswith("slice_") and column.endswith("_sla_met"):
            keys.append(column[: -len("_sla_met")])
    return sorted(keys)


def _slice_id_from_key(slice_key: str) -> str:
    # slice_1_0 -> 1-0
    return slice_key.replace("slice_", "", 1).replace("_", "-")


def _read_controller_series(
    output_dir: Path,
    controller: str,
    seed: int,
    allow_non_live: bool,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    run_log = output_dir / controller / f"seed_{seed:03d}" / "run_log.csv"
    if not run_log.exists():
        raise FileNotFoundError(f"Missing run log: {run_log}")

    rows = _load_rows(run_log)
    if not rows:
        raise RuntimeError(f"Run log has no rows: {run_log}")
    if not allow_non_live:
        _assert_live_rows(rows, run_log)

    header = list(rows[0].keys())
    slice_keys = _extract_slice_keys(header)

    steps = np.array([int(row["step"]) for row in rows], dtype=np.int64)
    series: Dict[str, np.ndarray] = {
        "step": steps,
        "slice_wide_sla_satisfaction": np.array(
            [float(row.get("slice_wide_sla_satisfaction", 0.0)) for row in rows], dtype=np.float64
        ),
        "slice_wide_efficiency": np.array(
            [float(row.get("slice_wide_efficiency", 0.0)) for row in rows], dtype=np.float64
        ),
    }

    for slice_key in slice_keys:
        for suffix in [
            "sla_met",
            "throughput_kbps",
            "offered_load_kbps",
            "latency_ms",
            "loss_pct",
        ]:
            col = f"{slice_key}_{suffix}"
            series[col] = np.array([float(row.get(col, 0.0)) for row in rows], dtype=np.float64)

        throughput = series[f"{slice_key}_throughput_kbps"]
        offered = series[f"{slice_key}_offered_load_kbps"]
        series[f"{slice_key}_served_to_demand_ratio"] = np.clip(throughput / np.maximum(offered, 1e-6), 0.0, 1.5)

    return slice_keys, series


def _add_shock_windows(
    axis: plt.Axes,
    max_step: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    current = shock_start
    while current <= max_step:
        axis.axvspan(current, current + shock_duration, color="red", alpha=0.12, linewidth=0)
        current += shock_interval


def _plot_slice_specific_sla(
    plots_dir: Path,
    slice_keys: List[str],
    controller_data: Dict[str, Dict[str, np.ndarray]],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axes = plt.subplots(len(slice_keys), 1, figsize=(11, 3.4 * len(slice_keys)), sharex=True)
    if len(slice_keys) == 1:
        axes = [axes]

    max_step = 0
    for idx, slice_key in enumerate(slice_keys):
        axis = axes[idx]
        for controller, data in controller_data.items():
            steps = data["step"]
            sla = data[f"{slice_key}_sla_met"]
            axis.plot(steps, _moving_average(sla, ma_window), linewidth=2, label=_display_name(controller))
            max_step = max(max_step, int(steps[-1]))

        axis.set_title(f"Slice {_slice_id_from_key(slice_key)} SLA Satisfaction")
        axis.set_ylabel("SLA met (MA)")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)

    axes[-1].set_xlabel("Step")
    axes[0].legend()
    fig.suptitle(f"Slice-Specific SLA Satisfaction (MA window={ma_window})", y=0.995)
    fig.tight_layout()
    fig.savefig(plots_dir / "slice_specific_sla_satisfaction.png", dpi=180)
    plt.close(fig)


def _plot_slice_specific_load_vs_served(
    plots_dir: Path,
    slice_keys: List[str],
    controller_data: Dict[str, Dict[str, np.ndarray]],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axes = plt.subplots(len(slice_keys), 1, figsize=(12, 3.8 * len(slice_keys)), sharex=True)
    if len(slice_keys) == 1:
        axes = [axes]

    max_step = 0
    for idx, slice_key in enumerate(slice_keys):
        axis = axes[idx]
        for controller, data in controller_data.items():
            steps = data["step"]
            offered = data[f"{slice_key}_offered_load_kbps"]
            served = data[f"{slice_key}_throughput_kbps"]
            label = _display_name(controller)
            axis.plot(steps, _moving_average(served, ma_window), linewidth=2, label=f"{label} served")
            axis.plot(steps, _moving_average(offered, ma_window), linewidth=1.5, linestyle="--", label=f"{label} demand")
            max_step = max(max_step, int(steps[-1]))

        axis.set_title(f"Slice {_slice_id_from_key(slice_key)} Demand vs Served Throughput")
        axis.set_ylabel("kbps (MA)")
        axis.grid(alpha=0.25)
        _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)

    axes[-1].set_xlabel("Step")
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle(f"Slice-Specific Demand and Served Throughput (MA window={ma_window})", y=0.995)
    fig.tight_layout()
    fig.savefig(plots_dir / "slice_specific_demand_vs_served.png", dpi=180)
    plt.close(fig)


def _plot_slice_specific_demand_served_ratio(
    plots_dir: Path,
    slice_keys: List[str],
    controller_data: Dict[str, Dict[str, np.ndarray]],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axes = plt.subplots(len(slice_keys), 1, figsize=(11, 3.5 * len(slice_keys)), sharex=True)
    if len(slice_keys) == 1:
        axes = [axes]

    max_step = 0
    for idx, slice_key in enumerate(slice_keys):
        axis = axes[idx]
        for controller, data in controller_data.items():
            steps = data["step"]
            ratio = data[f"{slice_key}_served_to_demand_ratio"]
            axis.plot(steps, _moving_average(ratio, ma_window), linewidth=2, label=_display_name(controller))
            max_step = max(max_step, int(steps[-1]))

        axis.set_title(f"Slice {_slice_id_from_key(slice_key)} Demand-Service Ratio")
        axis.set_ylabel("served / demand (MA)")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)

    axes[-1].set_xlabel("Step")
    axes[0].legend()
    fig.suptitle(f"Slice-Specific Demand vs Served Ratio (MA window={ma_window})", y=0.995)
    fig.tight_layout()
    fig.savefig(plots_dir / "slice_specific_demand_vs_served_ratio.png", dpi=180)
    plt.close(fig)


def _plot_slice_wide_sla(
    plots_dir: Path,
    controller_data: Dict[str, Dict[str, np.ndarray]],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.5))

    max_step = 0
    for controller, data in controller_data.items():
        steps = data["step"]
        satisfaction = data["slice_wide_sla_satisfaction"]
        axis.plot(steps, _moving_average(satisfaction, ma_window), linewidth=2.2, label=_display_name(controller))
        max_step = max(max_step, int(steps[-1]))

    _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)
    axis.set_title(f"Slice-Wide SLA Satisfaction (MA window={ma_window})")
    axis.set_xlabel("Step")
    axis.set_ylabel("Satisfied slices / total slices")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "slice_wide_sla_satisfaction.png", dpi=180)
    plt.close(fig)


def _plot_slice_wide_efficiency(
    plots_dir: Path,
    controller_data: Dict[str, Dict[str, np.ndarray]],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.5))

    max_step = 0
    for controller, data in controller_data.items():
        steps = data["step"]
        efficiency = data["slice_wide_efficiency"]
        axis.plot(steps, _moving_average(efficiency, ma_window), linewidth=2.2, label=_display_name(controller))
        max_step = max(max_step, int(steps[-1]))

    _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)
    axis.set_title(f"Slice-Wide Throughput Efficiency (MA window={ma_window})")
    axis.set_xlabel("Step")
    axis.set_ylabel("Throughput / Offered load")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "slice_wide_efficiency.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    controller_data: Dict[str, Dict[str, np.ndarray]] = {}
    union_slice_keys: List[str] = []

    for controller in args.controllers:
        slice_keys, data = _read_controller_series(
            output_dir,
            controller,
            args.seed,
            allow_non_live=args.allow_non_live,
        )
        controller_data[controller] = data
        for key in slice_keys:
            if key not in union_slice_keys:
                union_slice_keys.append(key)

    if not union_slice_keys:
        raise RuntimeError("No per-slice columns found in run logs.")

    _plot_slice_specific_sla(
        plots_dir,
        union_slice_keys,
        controller_data,
        args.ma_window,
        args.shock_start,
        args.shock_interval,
        args.shock_duration,
    )
    _plot_slice_specific_load_vs_served(
        plots_dir,
        union_slice_keys,
        controller_data,
        args.ma_window,
        args.shock_start,
        args.shock_interval,
        args.shock_duration,
    )
    _plot_slice_specific_demand_served_ratio(
        plots_dir,
        union_slice_keys,
        controller_data,
        args.ma_window,
        args.shock_start,
        args.shock_interval,
        args.shock_duration,
    )
    _plot_slice_wide_sla(
        plots_dir,
        controller_data,
        args.ma_window,
        args.shock_start,
        args.shock_interval,
        args.shock_duration,
    )
    _plot_slice_wide_efficiency(
        plots_dir,
        controller_data,
        args.ma_window,
        args.shock_start,
        args.shock_interval,
        args.shock_duration,
    )

    print(f"Saved congestion plots to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
