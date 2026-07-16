from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class SliceTarget:
    slice_id: str
    sst: int
    sd: int
    latency_budget_ms: float
    loss_budget_pct: float
    min_ratio_bounds: List[int]
    max_ratio_bounds: List[int]
    dedicated_ratio: int


@dataclass
class CollectorConfig:
    mode: str
    step_seconds: int
    csv_path: str
    prometheus_url: str
    metric_queries: Dict[str, str]
    enable_action_feedback: bool = False
    action_feedback_gain: float = 0.35
    strict_live_queries: bool = False
    allow_non_live_for_training: bool = False


@dataclass
class ControllerConfig:
    mode: str
    alpha: float
    min_step: int
    max_step: int
    checkpoint_path: str = ""
    require_checkpoint: bool = False
    offline_warm_start: bool = False
    warm_start_dataset_path: str = ""
    warm_start_epochs: int = 3
    warm_start_batch_size: int = 64
    warm_start_max_samples: int = 0
    hybrid_base_weight: float = 0.50
    hybrid_dqn_weight: float = 0.25
    hybrid_a2c_weight: float = 0.25
    hybrid_dqn_checkpoint_path: str = ""
    hybrid_a2c_checkpoint_path: str = ""
    hybrid_require_checkpoints: bool = False


@dataclass
class LiveCongestionConfig:
    enabled: bool = False
    shock_start_step: int = 200
    shock_interval_steps: int = 200
    shock_duration_steps: int = 20
    drop_ratio: float = 0.99
    container: str = "oaignb"
    interface: str = "eth0"
    baseline_rate_mbit: int = 1000
    burst_kbit: int = 64
    latency_ms: int = 50


@dataclass
class OutputConfig:
    policy_json_path: str
    run_log_csv: str


@dataclass
class ExperimentConfig:
    name: str
    horizon_steps: int
    random_seed: int
    slices: List[SliceTarget]
    collector: CollectorConfig
    controller: ControllerConfig
    live_congestion: LiveCongestionConfig
    output: OutputConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    raw = _load_yaml(Path(config_path))
    slices = [
        SliceTarget(
            slice_id=item["slice_id"],
            sst=int(item["sst"]),
            sd=int(item["sd"]),
            latency_budget_ms=float(item["latency_budget_ms"]),
            loss_budget_pct=float(item["loss_budget_pct"]),
            min_ratio_bounds=[int(item["min_ratio_bounds"][0]), int(item["min_ratio_bounds"][1])],
            max_ratio_bounds=[int(item["max_ratio_bounds"][0]), int(item["max_ratio_bounds"][1])],
            dedicated_ratio=int(item["dedicated_ratio"]),
        )
        for item in raw["slices"]
    ]
    collector_cfg = raw["collector"]
    controller_cfg = raw["controller"]
    live_congestion_cfg = raw.get("live_congestion", {})
    output_cfg = raw["output"]

    return ExperimentConfig(
        name=raw["name"],
        horizon_steps=int(raw["horizon_steps"]),
        random_seed=int(raw.get("random_seed", 42)),
        slices=slices,
        collector=CollectorConfig(
            mode=collector_cfg["mode"],
            step_seconds=int(collector_cfg.get("step_seconds", 1)),
            csv_path=collector_cfg.get("csv_path", ""),
            prometheus_url=collector_cfg.get("prometheus_url", ""),
            metric_queries=collector_cfg.get("metric_queries", {}),
            enable_action_feedback=bool(collector_cfg.get("enable_action_feedback", False)),
            action_feedback_gain=float(collector_cfg.get("action_feedback_gain", 0.35)),
            strict_live_queries=bool(collector_cfg.get("strict_live_queries", False)),
            allow_non_live_for_training=bool(collector_cfg.get("allow_non_live_for_training", False)),
        ),
        controller=ControllerConfig(
            mode=controller_cfg["mode"],
            alpha=float(controller_cfg.get("alpha", 0.2)),
            min_step=int(controller_cfg.get("min_step", 2)),
            max_step=int(controller_cfg.get("max_step", 5)),
            checkpoint_path=str(controller_cfg.get("checkpoint_path", "")),
            require_checkpoint=bool(controller_cfg.get("require_checkpoint", False)),
            offline_warm_start=bool(controller_cfg.get("offline_warm_start", False)),
            warm_start_dataset_path=str(controller_cfg.get("warm_start_dataset_path", "")),
            warm_start_epochs=int(controller_cfg.get("warm_start_epochs", 3)),
            warm_start_batch_size=int(controller_cfg.get("warm_start_batch_size", 64)),
            warm_start_max_samples=int(controller_cfg.get("warm_start_max_samples", 0)),
            hybrid_base_weight=float(controller_cfg.get("hybrid_base_weight", 0.50)),
            hybrid_dqn_weight=float(controller_cfg.get("hybrid_dqn_weight", 0.25)),
            hybrid_a2c_weight=float(controller_cfg.get("hybrid_a2c_weight", 0.25)),
            hybrid_dqn_checkpoint_path=str(controller_cfg.get("hybrid_dqn_checkpoint_path", "")),
            hybrid_a2c_checkpoint_path=str(controller_cfg.get("hybrid_a2c_checkpoint_path", "")),
            hybrid_require_checkpoints=bool(controller_cfg.get("hybrid_require_checkpoints", False)),
        ),
        live_congestion=LiveCongestionConfig(
            enabled=bool(live_congestion_cfg.get("enabled", False)),
            shock_start_step=int(live_congestion_cfg.get("shock_start_step", 200)),
            shock_interval_steps=int(live_congestion_cfg.get("shock_interval_steps", 200)),
            shock_duration_steps=int(live_congestion_cfg.get("shock_duration_steps", 20)),
            drop_ratio=float(live_congestion_cfg.get("drop_ratio", 0.99)),
            container=str(live_congestion_cfg.get("container", "oaignb")),
            interface=str(live_congestion_cfg.get("interface", "eth0")),
            baseline_rate_mbit=int(live_congestion_cfg.get("baseline_rate_mbit", 1000)),
            burst_kbit=int(live_congestion_cfg.get("burst_kbit", 64)),
            latency_ms=int(live_congestion_cfg.get("latency_ms", 50)),
        ),
        output=OutputConfig(
            policy_json_path=output_cfg["policy_json_path"],
            run_log_csv=output_cfg["run_log_csv"],
        ),
    )
