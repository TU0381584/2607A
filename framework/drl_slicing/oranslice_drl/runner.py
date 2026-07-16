import csv
import json
import os
import subprocess
import time
from typing import Dict, List

from .collectors import CsvReplayCollector, MetricsCollector, PrometheusCollector
from .config import ExperimentConfig, SliceTarget
from .controllers import (
    Controller,
    DrlControllerStub,
    RandomController,
    RuleBasedSLAController,
    RuleBasedDRLHybridController,
    ThresholdHeuristicController,
    StaticController,
    DQNController,
    A2CController,
    PPOController,
)
from .policy_io import write_rrm_policy_json
from .reward import compute_sla_reward
from .drl_training import StateEncoder, ActionDecoder


_INFERENCE_CONTROLLER_MODES = {"dqn", "a2c", "ppo"}
_TRAIN_CONTROLLER_MODES = {"dqn_train", "a2c_train", "ppo_train"}
_SYNTHETIC_QUERY_TOKENS = (
    "__step__",
    "vector(",
    "time(",
    "sin(",
    "cos(",
    "tan(",
    "rand(",
)


def _assert_live_oai_stack_running() -> None:
    """Fail fast unless the live OAI E2E stack is currently running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Docker CLI not found. Live OAI E2E mode requires Docker.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to query Docker containers for live OAI preflight.") from error

    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    required = ["amf", "smf", "upf", "oaignb"]
    missing = [name for name in required if name not in names]

    if missing:
        missing_txt = ", ".join(missing)
        raise RuntimeError(
            "Live OAI E2E stack is not up. Missing containers: "
            f"{missing_txt}. Start it first with: "
            "bash drl_slicing/scripts/run_closed_loop_sim.sh --ue-count 5 --active-algo dqn"
        )

    if not any(name.startswith("nr_ue_") for name in names):
        raise RuntimeError(
            "Live OAI E2E stack check failed: no OAI UE containers (nr_ue_*) are running. "
            "Start the stack with run_closed_loop_sim.sh before training."
        )


def _build_targets(config: ExperimentConfig) -> Dict[str, SliceTarget]:
    return {item.slice_id: item for item in config.slices}


def _validate_live_metric_queries(metric_queries: Dict[str, str]) -> None:
    for metric_name, query in metric_queries.items():
        lowered = query.lower()
        for token in _SYNTHETIC_QUERY_TOKENS:
            if token in lowered:
                raise ValueError(
                    "collector.strict_live_queries=true, but metric query "
                    f"'{metric_name}' contains forbidden synthetic token '{token}'."
                )


def _build_collector(config: ExperimentConfig, slice_ids: List[str]) -> MetricsCollector:
    mode = config.collector.mode

    if mode == "prometheus":
        _assert_live_oai_stack_running()
        if config.collector.strict_live_queries:
            _validate_live_metric_queries(config.collector.metric_queries)
        return PrometheusCollector(
            config.collector.prometheus_url,
            config.collector.metric_queries,
            slice_ids,
            config.collector.step_seconds,
            enable_action_feedback=config.collector.enable_action_feedback,
            action_feedback_gain=config.collector.action_feedback_gain,
        )

    if mode == "csv_replay":
        if not (
            config.collector.allow_non_live_for_training
            and config.controller.mode in _TRAIN_CONTROLLER_MODES
        ):
            raise ValueError(
                "collector.mode=csv_replay is only allowed for offline DRL training. "
                "Set collector.allow_non_live_for_training=true and use *_train controller modes."
            )
        return CsvReplayCollector(
            config.collector.csv_path,
            slice_ids,
            config.collector.step_seconds,
        )

    raise ValueError(f"Unknown collector mode: {mode}")


def _build_controller(config: ExperimentConfig, targets: Dict[str, SliceTarget]) -> Controller:
    mode = config.controller.mode
    if mode == "static":
        return StaticController(targets)
    if mode == "rule_based":
        return ThresholdHeuristicController(targets, config.controller)
    if mode == "rule_based_drl_hybrid":
        return RuleBasedDRLHybridController(targets, config.controller)
    if mode == "rule_based_sla":
        return RuleBasedSLAController(targets, config.controller)
    if mode == "random":
        return RandomController(targets, config.random_seed)
    if mode == "threshold_heuristic":
        return ThresholdHeuristicController(targets, config.controller)
    if mode == "drl_stub":
        return DrlControllerStub(targets, config.controller)
    if mode == "dqn":
        return DQNController(targets, config.controller, train=False)
    if mode == "dqn_train":
        return DQNController(targets, config.controller, train=True)
    if mode == "a2c":
        return A2CController(targets, config.controller, train=False)
    if mode == "a2c_train":
        return A2CController(targets, config.controller, train=True)
    if mode == "ppo":
        return PPOController(targets, config.controller, train=False)
    if mode == "ppo_train":
        return PPOController(targets, config.controller, train=True)
    raise ValueError(f"Unknown controller mode: {mode}")


def _load_controller_checkpoint(config: ExperimentConfig, controller: Controller) -> str:
    checkpoint_path = config.controller.checkpoint_path.strip()
    mode = config.controller.mode

    if not checkpoint_path:
        if config.controller.require_checkpoint and mode in _INFERENCE_CONTROLLER_MODES:
            raise ValueError(
                f"controller.require_checkpoint=true but no controller.checkpoint_path was provided for mode '{mode}'."
            )
        return ""

    if not hasattr(controller, "load_checkpoint"):
        raise ValueError(
            f"Controller mode '{mode}' does not support checkpoint loading."
        )

    if not os.path.isfile(checkpoint_path):
        # Training controllers should be allowed to start from scratch and save
        # a new checkpoint at the end of the run.
        if mode in _TRAIN_CONTROLLER_MODES:
            return ""
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    controller.load_checkpoint(checkpoint_path)
    return checkpoint_path


def _save_controller_checkpoint(config: ExperimentConfig, controller: Controller) -> None:
    checkpoint_path = config.controller.checkpoint_path.strip()
    if not checkpoint_path:
        return
    if config.controller.mode not in _TRAIN_CONTROLLER_MODES:
        return
    if not hasattr(controller, "save_checkpoint"):
        raise ValueError(
            f"Controller mode '{config.controller.mode}' does not support checkpoint saving."
        )

    parent = os.path.dirname(checkpoint_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    controller.save_checkpoint(checkpoint_path)


class _LiveCongestionShaper:
    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config.live_congestion
        self.enabled = bool(self.cfg.enabled)
        self.active = False

    def _run_tc(self, args: List[str], ignore_error: bool = False) -> None:
        cmd = ["docker", "exec", self.cfg.container, "tc", *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and not ignore_error:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown tc error"
            raise RuntimeError(f"Failed tc command ({' '.join(cmd)}): {detail}")

    def _apply_drop(self) -> None:
        keep_ratio = max(0.0, min(1.0, 1.0 - float(self.cfg.drop_ratio)))
        rate_mbit = max(1, int(round(float(self.cfg.baseline_rate_mbit) * keep_ratio)))
        self._run_tc(
            [
                "qdisc",
                "replace",
                "dev",
                self.cfg.interface,
                "root",
                "tbf",
                "rate",
                f"{rate_mbit}mbit",
                "burst",
                f"{self.cfg.burst_kbit}kbit",
                "latency",
                f"{self.cfg.latency_ms}ms",
            ]
        )
        self.active = True

    def _restore(self) -> None:
        self._run_tc(["qdisc", "del", "dev", self.cfg.interface, "root"], ignore_error=True)
        self.active = False

    def _is_shock_step(self, step: int) -> bool:
        start = max(0, int(self.cfg.shock_start_step))
        interval = max(1, int(self.cfg.shock_interval_steps))
        duration = max(1, int(self.cfg.shock_duration_steps))
        if step < start:
            return False
        return ((step - start) % interval) < duration

    def sync(self, step: int) -> None:
        if not self.enabled:
            return
        in_shock = self._is_shock_step(step)
        if in_shock and not self.active:
            self._apply_drop()
        elif not in_shock and self.active:
            self._restore()

    def status(self, step: int) -> str:
        if not self.enabled:
            return "disabled"
        return "shock" if self._is_shock_step(step) else "normal"

    def close(self) -> None:
        if self.active:
            self._restore()


def _append_run_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def run_experiment(config: ExperimentConfig) -> None:
    targets = _build_targets(config)
    slice_ids = list(targets.keys())
    collector = _build_collector(config, slice_ids)
    controller = _build_controller(config, targets)
    loaded_checkpoint = _load_controller_checkpoint(config, controller)
    congestion_shaper = _LiveCongestionShaper(config)
    state_encoder = StateEncoder(targets)
    action_decoder = ActionDecoder(targets)

    try:
        for step in range(config.horizon_steps):
            congestion_shaper.sync(step)

            state = collector.get_state(step)
            actions = controller.act(state)
            state_vec = state_encoder.encode(state)
            action_vec = action_decoder.encode_actions(actions)
            action_idx = action_decoder.encode_scalar(action_vec)
            write_rrm_policy_json(config.output.policy_json_path, actions, targets)

            collector.on_action(actions, targets, step)

            done = step >= (config.horizon_steps - 1)
            next_step = step if done else step + 1
            next_state = collector.get_state(next_step)
            reward, violations = compute_sla_reward(next_state, targets)
            train_metrics = controller.train_step(reward, next_state, done)

            slice_wide_sla_hits = 0
            slice_wide_throughput = 0.0
            slice_wide_offered = 0.0
            slice_wide_latency_viols = 0
            slice_wide_loss_viols = 0

            for slice_id in slice_ids:
                metrics = next_state.slices[slice_id]
                target = targets[slice_id]

                latency_violation = int(metrics.latency_ms > target.latency_budget_ms)
                loss_violation = int(metrics.loss_pct > target.loss_budget_pct)
                sla_met = int(latency_violation == 0 and loss_violation == 0)

                slice_wide_sla_hits += sla_met
                slice_wide_throughput += float(metrics.throughput_kbps)
                slice_wide_offered += float(metrics.offered_load_kbps)
                slice_wide_latency_viols += latency_violation
                slice_wide_loss_viols += loss_violation

            summary = ";".join(f"{a.slice_id}:min={a.min_ratio},max={a.max_ratio}" for a in actions)
            row = {
                "experiment": config.name,
                "controller_mode": config.controller.mode,
                "collector_mode": config.collector.mode,
                "trace_source": "prometheus_live" if config.collector.mode == "prometheus" else config.collector.mode,
                "live_trace_only": int(config.collector.mode == "prometheus"),
                "checkpoint_loaded": int(bool(loaded_checkpoint)),
                "checkpoint_path": loaded_checkpoint,
                "congestion_enabled": int(bool(config.live_congestion.enabled)),
                "congestion_state": congestion_shaper.status(step),
                "step": step,
                "timestamp_s": state.timestamp_s,
                "reward": reward,
                "sla_violations": violations,
                "action_summary": summary,
                "action_index": action_idx,
                "action_vector": json.dumps(action_vec.tolist()),
                "state_vector": json.dumps(state_vec.tolist()),
                "slice_wide_sla_satisfaction": float(slice_wide_sla_hits / max(len(slice_ids), 1)),
                "slice_wide_throughput_kbps": float(slice_wide_throughput),
                "slice_wide_offered_load_kbps": float(slice_wide_offered),
                "slice_wide_efficiency": float(slice_wide_throughput / max(slice_wide_offered, 1e-6)),
                "slice_wide_latency_violations": int(slice_wide_latency_viols),
                "slice_wide_loss_violations": int(slice_wide_loss_viols),
            }

            for slice_id in slice_ids:
                metrics = next_state.slices[slice_id]
                target = targets[slice_id]
                latency_violation = int(metrics.latency_ms > target.latency_budget_ms)
                loss_violation = int(metrics.loss_pct > target.loss_budget_pct)
                slice_key = f"slice_{slice_id.replace('-', '_')}"

                row[f"{slice_key}_throughput_kbps"] = float(metrics.throughput_kbps)
                row[f"{slice_key}_offered_load_kbps"] = float(metrics.offered_load_kbps)
                row[f"{slice_key}_latency_ms"] = float(metrics.latency_ms)
                row[f"{slice_key}_loss_pct"] = float(metrics.loss_pct)
                row[f"{slice_key}_latency_budget_ms"] = float(target.latency_budget_ms)
                row[f"{slice_key}_loss_budget_pct"] = float(target.loss_budget_pct)
                row[f"{slice_key}_latency_violation"] = latency_violation
                row[f"{slice_key}_loss_violation"] = loss_violation
                row[f"{slice_key}_sla_met"] = int(latency_violation == 0 and loss_violation == 0)

            if train_metrics:
                for key, value in train_metrics.items():
                    row[f"train_{key}"] = value
            _append_run_log(config.output.run_log_csv, row)
            time.sleep(config.collector.step_seconds)
    finally:
        congestion_shaper.close()

    _save_controller_checkpoint(config, controller)
