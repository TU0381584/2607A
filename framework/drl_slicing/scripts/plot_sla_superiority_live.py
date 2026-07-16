#!/usr/bin/env python3
"""
Generate SLA-superiority plots for live 3-UE ORAN congestion runs.

These plots are designed to surface policy differences under profile-priority
SLA objectives (URLLC > eMBB > mMTC) using live run_log.csv outputs.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROFILE_PRIORITY_WEIGHTS = {
    "urllc": 4.0,
    "embb": 2.0,
    "mmtc": 1.0,
}
PROFILE_ORDER = ["embb", "urllc", "mmtc"]


def _default_profiles_csv() -> Path:
    # drl_slicing/scripts -> ORANSlice/docker_open5gs/generated
    return Path(__file__).resolve().parents[2] / "docker_open5gs" / "generated" / "ue_fleet_profiles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SLA-priority superiority metrics from live controller logs")
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
        "--profiles-csv",
        default=str(_default_profiles_csv()),
        help="UE profile CSV with profile/slice_id mapping",
    )
    parser.add_argument(
        "--allow-non-live",
        action="store_true",
        help="Allow plotting run logs not marked as live Prometheus traces",
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

    steps = np.array([int(row.get("step", 0)) for row in rows], dtype=np.int64)
    series: Dict[str, np.ndarray] = {
        "step": steps,
        "reward": np.array([float(row.get("reward", 0.0)) for row in rows], dtype=np.float64),
        "sla_violations": np.array([float(row.get("sla_violations", 0.0)) for row in rows], dtype=np.float64),
    }

    for slice_key in slice_keys:
        for suffix in [
            "sla_met",
            "throughput_kbps",
            "offered_load_kbps",
            "latency_ms",
            "loss_pct",
            "latency_budget_ms",
            "loss_budget_pct",
        ]:
            column = f"{slice_key}_{suffix}"
            series[column] = np.array([float(row.get(column, 0.0)) for row in rows], dtype=np.float64)

    return slice_keys, series


def _load_slice_profile_map(profiles_csv: Path) -> Dict[str, str]:
    if not profiles_csv.is_file():
        return {}

    with profiles_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, str] = {}
        for row in reader:
            slice_id = (row.get("slice_id") or "").strip()
            profile = (row.get("profile") or "").strip().lower()
            if not slice_id:
                continue
            if profile not in PROFILE_PRIORITY_WEIGHTS:
                continue
            mapping[slice_id] = profile
        return mapping


def _infer_profile_from_budgets(latency_budget_ms: float, loss_budget_pct: float) -> str:
    if latency_budget_ms <= 20.0 or loss_budget_pct <= 1.0:
        return "urllc"
    if latency_budget_ms <= 50.0 or loss_budget_pct <= 2.5:
        return "embb"
    return "mmtc"


def _resolve_slice_profiles(
    slice_keys: List[str],
    series: Dict[str, np.ndarray],
    profile_map: Dict[str, str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for slice_key in slice_keys:
        slice_id = _slice_id_from_key(slice_key)
        profile = profile_map.get(slice_id)

        if profile is None:
            lat_budget = float(series[f"{slice_key}_latency_budget_ms"][0])
            loss_budget = float(series[f"{slice_key}_loss_budget_pct"][0])
            profile = _infer_profile_from_budgets(lat_budget, loss_budget)

        resolved[slice_key] = profile

    return resolved


def _add_shock_windows(
    axis: plt.Axes,
    max_step: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    current = max(0, shock_start)
    period = max(1, shock_interval)
    duration = max(1, shock_duration)
    while current <= max_step:
        axis.axvspan(current, current + duration, color="red", alpha=0.12, linewidth=0)
        current += period


def _compute_weighted_debt_and_sla(
    slice_keys: List[str],
    series: Dict[str, np.ndarray],
    slice_profiles: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    weighted_debt = np.zeros_like(series["step"], dtype=np.float64)
    weighted_sla = np.zeros_like(series["step"], dtype=np.float64)

    total_weight = 0.0
    for slice_key in slice_keys:
        profile = slice_profiles.get(slice_key, "embb")
        weight = float(PROFILE_PRIORITY_WEIGHTS.get(profile, 1.0))
        total_weight += weight

        latency = series[f"{slice_key}_latency_ms"]
        loss = series[f"{slice_key}_loss_pct"]
        latency_budget = np.maximum(series[f"{slice_key}_latency_budget_ms"], 1e-6)
        loss_budget = np.maximum(series[f"{slice_key}_loss_budget_pct"], 1e-6)
        throughput = series[f"{slice_key}_throughput_kbps"]
        offered = np.maximum(series[f"{slice_key}_offered_load_kbps"], 1.0)
        sla_met = series[f"{slice_key}_sla_met"]

        latency_over = np.maximum(0.0, (latency - latency_budget) / latency_budget)
        loss_over = np.maximum(0.0, (loss - loss_budget) / loss_budget)
        demand_gap = np.maximum(0.0, 1.0 - (throughput / offered))

        # Same weighting intent as the online reward, plus demand-gap pressure.
        weighted_debt += weight * (2.0 * latency_over + 3.0 * loss_over + 1.2 * demand_gap)
        weighted_sla += weight * sla_met

    weighted_sla /= max(total_weight, 1e-6)
    return weighted_debt, weighted_sla


def _build_shock_windows(max_step: int, shock_start: int, shock_interval: int, shock_duration: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    current = max(0, shock_start)
    interval = max(1, shock_interval)
    duration = max(1, shock_duration)
    while current <= max_step:
        windows.append((current, current + duration))
        current += interval
    return windows


def _compute_recovery_steps(
    steps: np.ndarray,
    weighted_sla: np.ndarray,
    shock_windows: List[Tuple[int, int]],
    threshold: float = 0.999,
) -> List[float]:
    out: List[float] = []
    for _, recovery_start in shock_windows:
        start_idx = int(np.searchsorted(steps, recovery_start, side="left"))
        if start_idx >= len(steps):
            continue

        recovered = None
        for idx in range(start_idx, len(steps)):
            if weighted_sla[idx] >= threshold:
                recovered = float(steps[idx] - recovery_start)
                break

        if recovered is None:
            recovered = float(max(int(steps[-1]) - recovery_start + 1, 0))
        out.append(max(recovered, 0.0))

    return out


def _compute_burst_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for active in mask:
        if bool(active):
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    if not lengths:
        lengths.append(0)
    return lengths


def _profile_groups(slice_profiles: Dict[str, str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {profile: [] for profile in PROFILE_ORDER}
    for slice_key, profile in slice_profiles.items():
        groups.setdefault(profile, []).append(slice_key)
    return groups


def _plot_priority_weighted_debt(
    plots_dir: Path,
    controller_data: Dict[str, Dict[str, np.ndarray]],
    weighted_debt: Dict[str, np.ndarray],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.5))

    max_step = 0
    for controller, data in controller_data.items():
        steps = data["step"]
        axis.plot(steps, _moving_average(weighted_debt[controller], ma_window), linewidth=2.2, label=_display_name(controller))
        max_step = max(max_step, int(steps[-1]))

    _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)
    axis.set_title(f"Priority-Weighted SLA Debt (lower is better, MA={ma_window})")
    axis.set_xlabel("Step")
    axis.set_ylabel("Weighted debt")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "priority_weighted_sla_debt.png", dpi=180)
    plt.close(fig)


def _plot_weighted_sla(
    plots_dir: Path,
    controller_data: Dict[str, Dict[str, np.ndarray]],
    weighted_sla: Dict[str, np.ndarray],
    ma_window: int,
    shock_start: int,
    shock_interval: int,
    shock_duration: int,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.5))

    max_step = 0
    for controller, data in controller_data.items():
        steps = data["step"]
        axis.plot(steps, _moving_average(weighted_sla[controller], ma_window), linewidth=2.2, label=_display_name(controller))
        max_step = max(max_step, int(steps[-1]))

    _add_shock_windows(axis, max_step, shock_start, shock_interval, shock_duration)
    axis.set_title(f"Priority-Weighted SLA Satisfaction (higher is better, MA={ma_window})")
    axis.set_xlabel("Step")
    axis.set_ylabel("Weighted SLA hit ratio")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "priority_weighted_sla_satisfaction.png", dpi=180)
    plt.close(fig)


def _plot_shock_recovery_steps(
    plots_dir: Path,
    recovery_steps: Dict[str, List[float]],
) -> None:
    controllers = list(recovery_steps.keys())
    means = [float(np.mean(recovery_steps[c])) if recovery_steps[c] else 0.0 for c in controllers]

    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar([_display_name(c) for c in controllers], means, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(controllers)])
    axis.set_title("Average Recovery Time After Congestion Shocks")
    axis.set_ylabel("Steps to recover full weighted SLA")
    axis.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2.0, value + 0.05, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(plots_dir / "shock_recovery_steps.png", dpi=180)
    plt.close(fig)


def _plot_violation_burst_cdf(
    plots_dir: Path,
    burst_lengths: Dict[str, List[int]],
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.2))

    for controller, lengths in burst_lengths.items():
        arr = np.sort(np.array(lengths, dtype=np.int64))
        y = np.arange(1, len(arr) + 1, dtype=np.float64) / max(len(arr), 1)
        axis.step(arr, y, where="post", linewidth=2.0, label=_display_name(controller))

    axis.set_title("CDF of Consecutive SLA-Violation Burst Lengths")
    axis.set_xlabel("Consecutive violating steps")
    axis.set_ylabel("CDF")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "violation_burst_cdf.png", dpi=180)
    plt.close(fig)


def _plot_profile_tail_risk(
    plots_dir: Path,
    controllers: List[str],
    controller_data: Dict[str, Dict[str, np.ndarray]],
    controller_profiles: Dict[str, Dict[str, str]],
) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.3))
    x = np.arange(len(PROFILE_ORDER), dtype=np.float64)
    width = 0.8 / max(len(controllers), 1)

    for idx, controller in enumerate(controllers):
        data = controller_data[controller]
        groups = _profile_groups(controller_profiles[controller])
        values: List[float] = []

        for profile in PROFILE_ORDER:
            keys = groups.get(profile, [])
            if not keys:
                values.append(0.0)
                continue

            profile_scores = []
            for slice_key in keys:
                latency = data[f"{slice_key}_latency_ms"]
                loss = data[f"{slice_key}_loss_pct"]
                latency_budget = np.maximum(data[f"{slice_key}_latency_budget_ms"], 1e-6)
                loss_budget = np.maximum(data[f"{slice_key}_loss_budget_pct"], 1e-6)
                throughput = data[f"{slice_key}_throughput_kbps"]
                offered = np.maximum(data[f"{slice_key}_offered_load_kbps"], 1.0)

                latency_over = np.maximum(0.0, (latency - latency_budget) / latency_budget)
                loss_over = np.maximum(0.0, (loss - loss_budget) / loss_budget)
                demand_gap = np.maximum(0.0, 1.0 - throughput / offered)
                score = 2.0 * latency_over + 3.0 * loss_over + 1.2 * demand_gap
                profile_scores.append(float(np.percentile(score, 95)))

            values.append(float(np.mean(profile_scores)))

        offsets = x + (idx - (len(controllers) - 1) / 2.0) * width
        axis.bar(offsets, values, width=width, label=_display_name(controller))

    axis.set_xticks(x)
    axis.set_xticklabels([p.upper() for p in PROFILE_ORDER])
    axis.set_title("Profile Tail-Risk Score (95th percentile, lower is better)")
    axis.set_ylabel("Tail-risk score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "profile_tail_risk.png", dpi=180)
    plt.close(fig)


def _plot_profile_demand_randomness(
    plots_dir: Path,
    controllers: List[str],
    controller_data: Dict[str, Dict[str, np.ndarray]],
    controller_profiles: Dict[str, Dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.0), sharex=True)
    x = np.arange(len(PROFILE_ORDER), dtype=np.float64)
    width = 0.8 / max(len(controllers), 1)

    for idx, controller in enumerate(controllers):
        data = controller_data[controller]
        groups = _profile_groups(controller_profiles[controller])

        cv_values: List[float] = []
        spread_values: List[float] = []

        for profile in PROFILE_ORDER:
            keys = groups.get(profile, [])
            if not keys:
                cv_values.append(0.0)
                spread_values.append(0.0)
                continue

            cvs = []
            spreads = []
            for slice_key in keys:
                offered = np.maximum(data[f"{slice_key}_offered_load_kbps"], 1e-6)
                mean = float(np.mean(offered))
                std = float(np.std(offered))
                p10 = max(float(np.percentile(offered, 10)), 1e-6)
                p90 = float(np.percentile(offered, 90))
                cvs.append(std / max(mean, 1e-6))
                spreads.append(p90 / p10)

            cv_values.append(float(np.mean(cvs)))
            spread_values.append(float(np.mean(spreads)))

        offsets = x + (idx - (len(controllers) - 1) / 2.0) * width
        axes[0].bar(offsets, cv_values, width=width, label=_display_name(controller))
        axes[1].bar(offsets, spread_values, width=width, label=_display_name(controller))

    axes[0].set_title("Offered-Load Randomness by Profile (Coefficient of Variation)")
    axes[0].set_ylabel("CV")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Offered-Load Randomness by Profile (P90/P10)")
    axes[1].set_ylabel("P90 / P10")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.upper() for p in PROFILE_ORDER])
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(plots_dir / "profile_demand_randomness.png", dpi=180)
    plt.close(fig)


def _plot_live_profile_demand_trace(
    plots_dir: Path,
    reference_controller: str,
    series: Dict[str, np.ndarray],
    slice_profiles: Dict[str, str],
    ma_window: int,
) -> None:
    profile_to_slice: Dict[str, str] = {}
    for slice_key, profile in slice_profiles.items():
        if profile in PROFILE_ORDER and profile not in profile_to_slice:
            profile_to_slice[profile] = slice_key

    fig, axis = plt.subplots(figsize=(11, 5.3))

    for profile in PROFILE_ORDER:
        slice_key = profile_to_slice.get(profile)
        if not slice_key:
            continue

        offered = series[f"{slice_key}_offered_load_kbps"]
        steps = series["step"]
        axis.plot(steps, offered, alpha=0.20, linewidth=1.0)
        axis.plot(steps, _moving_average(offered, ma_window), linewidth=2.2, label=profile.upper())

    axis.set_title(f"Live Offered-Load Trace by Profile ({_display_name(reference_controller)})")
    axis.set_xlabel("Step")
    axis.set_ylabel("Offered load (kbps)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "live_profile_offered_load_trace.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    profile_map = _load_slice_profile_map(Path(args.profiles_csv))

    controller_data: Dict[str, Dict[str, np.ndarray]] = {}
    controller_slice_keys: Dict[str, List[str]] = {}
    controller_profiles: Dict[str, Dict[str, str]] = {}
    weighted_debt: Dict[str, np.ndarray] = {}
    weighted_sla: Dict[str, np.ndarray] = {}

    for controller in args.controllers:
        slice_keys, series = _read_controller_series(
            output_dir=output_dir,
            controller=controller,
            seed=args.seed,
            allow_non_live=args.allow_non_live,
        )
        controller_data[controller] = series
        controller_slice_keys[controller] = slice_keys

        slice_profiles = _resolve_slice_profiles(slice_keys, series, profile_map)
        controller_profiles[controller] = slice_profiles

        debt, wsla = _compute_weighted_debt_and_sla(slice_keys, series, slice_profiles)
        weighted_debt[controller] = debt
        weighted_sla[controller] = wsla

    max_step = max(int(data["step"][-1]) for data in controller_data.values())
    shock_windows = _build_shock_windows(
        max_step=max_step,
        shock_start=args.shock_start,
        shock_interval=args.shock_interval,
        shock_duration=args.shock_duration,
    )

    recovery_steps: Dict[str, List[float]] = {}
    burst_lengths: Dict[str, List[int]] = {}
    for controller, data in controller_data.items():
        recovery_steps[controller] = _compute_recovery_steps(data["step"], weighted_sla[controller], shock_windows)
        violation_mask = data["sla_violations"] > 0.0
        burst_lengths[controller] = _compute_burst_lengths(violation_mask)

    _plot_priority_weighted_debt(
        plots_dir=plots_dir,
        controller_data=controller_data,
        weighted_debt=weighted_debt,
        ma_window=args.ma_window,
        shock_start=args.shock_start,
        shock_interval=args.shock_interval,
        shock_duration=args.shock_duration,
    )
    _plot_weighted_sla(
        plots_dir=plots_dir,
        controller_data=controller_data,
        weighted_sla=weighted_sla,
        ma_window=args.ma_window,
        shock_start=args.shock_start,
        shock_interval=args.shock_interval,
        shock_duration=args.shock_duration,
    )
    _plot_shock_recovery_steps(plots_dir=plots_dir, recovery_steps=recovery_steps)
    _plot_violation_burst_cdf(plots_dir=plots_dir, burst_lengths=burst_lengths)
    _plot_profile_tail_risk(
        plots_dir=plots_dir,
        controllers=args.controllers,
        controller_data=controller_data,
        controller_profiles=controller_profiles,
    )
    _plot_profile_demand_randomness(
        plots_dir=plots_dir,
        controllers=args.controllers,
        controller_data=controller_data,
        controller_profiles=controller_profiles,
    )

    ref_controller = args.controllers[0]
    _plot_live_profile_demand_trace(
        plots_dir=plots_dir,
        reference_controller=ref_controller,
        series=controller_data[ref_controller],
        slice_profiles=controller_profiles[ref_controller],
        ma_window=args.ma_window,
    )

    print(f"Saved SLA-superiority plots to: {plots_dir}")
    for controller in args.controllers:
        avg_debt = float(np.mean(weighted_debt[controller]))
        avg_wsla = float(np.mean(weighted_sla[controller]))
        avg_recovery = float(np.mean(recovery_steps[controller])) if recovery_steps[controller] else 0.0
        print(
            f"{controller}: avg_weighted_debt={avg_debt:.4f}, "
            f"avg_weighted_sla={avg_wsla:.4f}, "
            f"avg_recovery_steps={avg_recovery:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
