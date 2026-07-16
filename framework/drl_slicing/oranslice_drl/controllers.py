import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from .config import ControllerConfig, SliceTarget
from .types import EnvState, SliceAction
from .drl_policy import DQNPolicy, A2CPolicy, PPOPolicy, RLPolicy
from .drl_training import StateEncoder, ActionDecoder, DRLTrainer


class Controller(ABC):
    @abstractmethod
    def act(self, state: EnvState) -> List[SliceAction]:
        raise NotImplementedError

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        """Optional training callback; no-op for non-learning controllers."""
        return {}
    
    def get_training_info(self) -> Dict:
        """Return training metrics/info if applicable."""
        return {}


class StaticController(Controller):
    def __init__(self, targets: Dict[str, SliceTarget]) -> None:
        self.targets = targets

    def act(self, state: EnvState) -> List[SliceAction]:
        actions: List[SliceAction] = []
        for target in self.targets.values():
            actions.append(
                SliceAction(
                    slice_id=target.slice_id,
                    min_ratio=target.min_ratio_bounds[0],
                    max_ratio=target.max_ratio_bounds[1],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions


class RuleBasedSLAController(Controller):
    def __init__(self, targets: Dict[str, SliceTarget], cfg: ControllerConfig) -> None:
        self.targets = targets
        self.cfg = cfg
        self.current_min: Dict[str, int] = {k: v.min_ratio_bounds[0] for k, v in targets.items()}
        self.current_max: Dict[str, int] = {k: v.max_ratio_bounds[1] for k, v in targets.items()}

    def _compute_adjustment(self, violation_strength: float) -> int:
        if violation_strength <= 0:
            return -self.cfg.min_step
        return min(self.cfg.max_step, max(self.cfg.min_step, int(violation_strength * self.cfg.max_step)))

    def act(self, state: EnvState) -> List[SliceAction]:
        actions: List[SliceAction] = []
        for slice_id, metrics in state.slices.items():
            target = self.targets[slice_id]
            latency_over = max(0.0, (metrics.latency_ms - target.latency_budget_ms) / max(target.latency_budget_ms, 1e-6))
            loss_over = max(0.0, (metrics.loss_pct - target.loss_budget_pct) / max(target.loss_budget_pct, 1e-6))
            violation_strength = 0.6 * latency_over + 0.4 * loss_over
            delta = self._compute_adjustment(violation_strength)

            next_min = self.current_min[slice_id] + delta
            next_max = self.current_max[slice_id] + delta

            next_min = max(target.min_ratio_bounds[0], min(target.min_ratio_bounds[1], next_min))
            next_max = max(target.max_ratio_bounds[0], min(target.max_ratio_bounds[1], next_max))
            if next_min > next_max:
                next_min = next_max

            self.current_min[slice_id] = next_min
            self.current_max[slice_id] = next_max

            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=next_min,
                    max_ratio=next_max,
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions


class ThresholdHeuristicController(Controller):
    """
    Baseline threshold controller that is intentionally not SLA-aware.

    It only reacts to served-demand ratio (throughput/offered_load),
    then redistributes min/max ratios using persistent per-slice scores.
    """

    def __init__(self, targets: Dict[str, SliceTarget], cfg: ControllerConfig) -> None:
        self.targets = targets
        self.cfg = cfg
        self.low_ratio = 0.80
        self.high_ratio = 0.97
        self.increase_gain = 0.35
        self.decrease_gain = 0.20
        self.scores: Dict[str, float] = {slice_id: 1.0 for slice_id in targets.keys()}

    def _update_score(self, slice_id: str, served_ratio: float) -> None:
        score = self.scores[slice_id]

        if served_ratio < self.low_ratio:
            score += self.increase_gain * (self.low_ratio - served_ratio + 0.2)
        elif served_ratio > self.high_ratio:
            score -= self.decrease_gain * (served_ratio - self.high_ratio + 0.2)
        else:
            score *= 0.995

        self.scores[slice_id] = float(np.clip(score, 0.2, 5.0))

    def act(self, state: EnvState) -> List[SliceAction]:
        for slice_id, metrics in state.slices.items():
            offered = max(metrics.offered_load_kbps, 1.0)
            served_ratio = float(np.clip(metrics.throughput_kbps / offered, 0.0, 1.25))
            self._update_score(slice_id, served_ratio)

        total_score = sum(self.scores.values())
        if total_score <= 0.0:
            total_score = float(max(len(self.scores), 1))

        actions: List[SliceAction] = []
        for slice_id, target in self.targets.items():
            share = self.scores[slice_id] / total_score

            min_lo, min_hi = target.min_ratio_bounds
            max_lo, max_hi = target.max_ratio_bounds

            next_min = int(round(min_lo + share * (min_hi - min_lo)))
            next_max = int(round(max_lo + share * (max_hi - max_lo)))

            next_min = max(min_lo, min(min_hi, next_min))
            next_max = max(max_lo, min(max_hi, next_max))
            if next_min > next_max:
                next_max = next_min

            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=next_min,
                    max_ratio=next_max,
                    dedicated_ratio=target.dedicated_ratio,
                )
            )

        return actions


class RuleBasedDRLHybridController(Controller):
    """
    Rule-based baseline enhanced with frozen DQN and A2C policy proposals.

    Final action is a weighted blend of baseline, DQN, and A2C actions.
    """

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        device: str = "cpu",
    ) -> None:
        self.targets = targets
        self.cfg = cfg
        self.device = device

        self.base_controller = ThresholdHeuristicController(targets, cfg)
        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        self.weights = self._normalize_weights(
            base_weight=cfg.hybrid_base_weight,
            dqn_weight=cfg.hybrid_dqn_weight,
            a2c_weight=cfg.hybrid_a2c_weight,
        )

        self.dqn_policy = DQNPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            n_branches=self.action_decoder.num_slices,
            hidden_dim=128,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
            device=device,
        )
        self.a2c_policy = A2CPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            n_branches=self.action_decoder.num_slices,
            hidden_dim=128,
            learning_rate=1e-3,
            gamma=0.99,
            device=device,
        )

        self.dqn_checkpoint_loaded = ""
        self.a2c_checkpoint_loaded = ""
        self._load_hybrid_checkpoints()

    @staticmethod
    def _normalize_weights(base_weight: float, dqn_weight: float, a2c_weight: float) -> Dict[str, float]:
        base = max(0.0, float(base_weight))
        dqn = max(0.0, float(dqn_weight))
        a2c = max(0.0, float(a2c_weight))
        total = base + dqn + a2c
        if total <= 1e-9:
            return {"base": 1.0, "dqn": 0.0, "a2c": 0.0}
        return {
            "base": base / total,
            "dqn": dqn / total,
            "a2c": a2c / total,
        }

    @staticmethod
    def _resolve_checkpoint_path(path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        drl_root = Path(__file__).resolve().parents[1]
        return (drl_root / candidate).resolve()

    def _load_policy_checkpoint(self, policy: RLPolicy, path: str, label: str, required: bool) -> str:
        clean = str(path).strip()
        if not clean:
            if required:
                raise ValueError(f"Hybrid controller requires {label} checkpoint path, but it is empty.")
            return ""

        resolved = self._resolve_checkpoint_path(clean)
        if not resolved.is_file():
            if required:
                raise FileNotFoundError(f"Hybrid controller {label} checkpoint not found: {resolved}")
            return ""

        policy.load_checkpoint(str(resolved))
        return str(resolved)

    def _load_hybrid_checkpoints(self) -> None:
        required = bool(self.cfg.hybrid_require_checkpoints)
        self.dqn_checkpoint_loaded = self._load_policy_checkpoint(
            policy=self.dqn_policy,
            path=self.cfg.hybrid_dqn_checkpoint_path,
            label="DQN",
            required=required,
        )
        self.a2c_checkpoint_loaded = self._load_policy_checkpoint(
            policy=self.a2c_policy,
            path=self.cfg.hybrid_a2c_checkpoint_path,
            label="A2C",
            required=required,
        )

    @staticmethod
    def _actions_to_allocation(actions: List[SliceAction]) -> Dict[str, Dict[str, int]]:
        return {
            action.slice_id: {
                "min_ratio": int(action.min_ratio),
                "max_ratio": int(action.max_ratio),
            }
            for action in actions
        }

    def act(self, state: EnvState) -> List[SliceAction]:
        baseline_actions = self.base_controller.act(state)
        baseline_alloc = self._actions_to_allocation(baseline_actions)

        state_vec = self.state_encoder.encode(state)

        dqn_action_idx, _ = self.dqn_policy.select_action(state_vec, training=False)
        dqn_alloc = self.action_decoder.decode(self.action_decoder.decode_to_vector(dqn_action_idx))

        a2c_action_idx, _ = self.a2c_policy.select_action(state_vec, training=False)
        a2c_alloc = self.action_decoder.decode(self.action_decoder.decode_to_vector(a2c_action_idx))

        out_actions: List[SliceAction] = []
        for slice_id, target in self.targets.items():
            base_min = baseline_alloc[slice_id]["min_ratio"]
            base_max = baseline_alloc[slice_id]["max_ratio"]
            dqn_min = dqn_alloc[slice_id]["min_ratio"]
            dqn_max = dqn_alloc[slice_id]["max_ratio"]
            a2c_min = a2c_alloc[slice_id]["min_ratio"]
            a2c_max = a2c_alloc[slice_id]["max_ratio"]

            raw_min = (
                self.weights["base"] * base_min
                + self.weights["dqn"] * dqn_min
                + self.weights["a2c"] * a2c_min
            )
            raw_max = (
                self.weights["base"] * base_max
                + self.weights["dqn"] * dqn_max
                + self.weights["a2c"] * a2c_max
            )

            min_lo, min_hi = target.min_ratio_bounds
            max_lo, max_hi = target.max_ratio_bounds
            min_ratio = int(round(raw_min))
            max_ratio = int(round(raw_max))

            min_ratio = max(min_lo, min(min_hi, min_ratio))
            max_ratio = max(max_lo, min(max_hi, max_ratio))
            if min_ratio > max_ratio:
                min_ratio = max_ratio

            out_actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                    dedicated_ratio=target.dedicated_ratio,
                )
            )

        return out_actions

    def get_training_info(self) -> Dict:
        return {
            "hybrid_base_weight": self.weights["base"],
            "hybrid_dqn_weight": self.weights["dqn"],
            "hybrid_a2c_weight": self.weights["a2c"],
            "hybrid_dqn_checkpoint_loaded": self.dqn_checkpoint_loaded,
            "hybrid_a2c_checkpoint_loaded": self.a2c_checkpoint_loaded,
        }


class RandomController(Controller):
    def __init__(self, targets: Dict[str, SliceTarget], seed: int) -> None:
        self.targets = targets
        self.rng = random.Random(seed)

    def act(self, state: EnvState) -> List[SliceAction]:
        actions: List[SliceAction] = []
        for target in self.targets.values():
            min_ratio = self.rng.randint(target.min_ratio_bounds[0], target.min_ratio_bounds[1])
            max_ratio = self.rng.randint(max(min_ratio, target.max_ratio_bounds[0]), target.max_ratio_bounds[1])
            actions.append(
                SliceAction(
                    slice_id=target.slice_id,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions


class DrlControllerStub(RuleBasedSLAController):
    def __init__(self, targets: Dict[str, SliceTarget], cfg: ControllerConfig) -> None:
        super().__init__(targets, cfg)


def _apply_offline_warm_start(
    policy: RLPolicy,
    cfg: ControllerConfig,
    state_dim: int,
    action_decoder: ActionDecoder,
) -> Dict:
    """Apply offline behavior-cloning warm-start if enabled in config."""
    if not cfg.offline_warm_start:
        return {}

    source_path = cfg.warm_start_dataset_path.strip()
    if not source_path:
        raise ValueError(
            "offline_warm_start is enabled, but controller.warm_start_dataset_path is empty."
        )

    from .offline_warmstart import load_state_action_dataset

    states, actions = load_state_action_dataset(
        source_path=source_path,
        expected_state_dim=state_dim,
        max_samples=max(0, int(cfg.warm_start_max_samples)),
    )

    # Backward compatibility: legacy datasets may store scalar action indices.
    if actions.ndim == 1:
        actions = np.stack([action_decoder.decode_to_vector(int(a)) for a in actions], axis=0)

    metrics = policy.behavior_clone_train(
        states=states,
        actions=actions,
        epochs=max(1, int(cfg.warm_start_epochs)),
        batch_size=max(1, int(cfg.warm_start_batch_size)),
    )
    metrics.update(
        {
            "source": source_path,
            "samples_loaded": int(states.shape[0]),
            "epochs": int(max(1, int(cfg.warm_start_epochs))),
            "batch_size": int(max(1, int(cfg.warm_start_batch_size))),
        }
    )
    return metrics


class DQNController(Controller):
    """
    DRL controller using Deep Q-Network policy.
    Optionally supports training when replay buffer is updated.
    """

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        learning_rate: float = 1e-3,
        train: bool = False,
        device: str = "cpu",
    ):
        self.targets = targets
        self.cfg = cfg
        self.train_mode = train

        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        # Initialize DQN policy
        self.policy = DQNPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            n_branches=self.action_decoder.num_slices,
            hidden_dim=128,
            learning_rate=learning_rate,
            gamma=0.99,
            epsilon_start=1.0 if train else 0.05,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            device=device,
        )

        # Trainer for online learning
        if train:
            self.trainer = DRLTrainer(
                self.policy,
                self.state_encoder,
                self.action_decoder,
                batch_size=32,
                update_freq=4,
                warmup_steps=100,
            )
        else:
            self.trainer = None

        self._warm_start_metrics: Dict = {}
        self._warm_start_emitted = False
        if train and cfg.offline_warm_start:
            self._warm_start_metrics = _apply_offline_warm_start(
                policy=self.policy,
                cfg=cfg,
                state_dim=self.state_encoder.state_dim,
                action_decoder=self.action_decoder,
            )

        self._last_state = None
        self._last_action_idx = None

    def act(self, state: EnvState) -> List[SliceAction]:
        """
        Select actions using DQN policy.
        """
        state_vec = self.state_encoder.encode(state)
        action_idx, q_value = self.policy.select_action(state_vec, training=self.train_mode)
        action_vec = self.action_decoder.decode_to_vector(action_idx)
        self._last_state = state
        self._last_action_idx = action_vec
        allocation = self.action_decoder.decode(action_vec)

        actions: List[SliceAction] = []
        for slice_id, alloc in allocation.items():
            target = self.targets[slice_id]
            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=alloc["min_ratio"],
                    max_ratio=alloc["max_ratio"],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        """
        Update policy with experience (requires previous state/action).
        Call this after each environment step.
        """
        if not self.train_mode or self.trainer is None:
            return {}
        if self._last_state is None or self._last_action_idx is None:
            return {}

        self.trainer.step(
            self._last_state,
            self._last_action_idx,
            reward,
            next_state,
            done,
        )
        metrics = self.trainer.get_metrics()
        if self._warm_start_metrics and not self._warm_start_emitted:
            metrics.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
            self._warm_start_emitted = True
        return metrics

    def record_transition(self, state: EnvState, action_list: List[SliceAction]) -> None:
        """Record current state/action for next train_step call."""
        self._last_state = state
        # Convert SliceAction list to factorized action vector for replay buffer.
        self._last_action_idx = self.action_decoder.encode_actions(action_list)

    def get_training_info(self) -> Dict:
        """Return current training metrics."""
        if self.trainer is None:
            return {}
        info = {
            "epsilon": self.policy.epsilon,
            "step_count": self.policy.train_step_count,
            **self.trainer.get_metrics(),
        }
        if self._warm_start_metrics:
            info.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
        return info

    def save_checkpoint(self, path: str) -> None:
        self.policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_checkpoint(path)


class A2CController(Controller):
    """
    DRL controller using Advantage Actor-Critic policy.
    Implementation to be completed in Phase 2b.
    """

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        learning_rate: float = 1e-3,
        train: bool = False,
        device: str = "cpu",
    ):
        self.targets = targets
        self.cfg = cfg
        self.train_mode = train

        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        # Initialize A2C policy
        self.policy = A2CPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            n_branches=self.action_decoder.num_slices,
            hidden_dim=128,
            learning_rate=learning_rate,
            gamma=0.99,
            device=device,
        )

        self.trainer = None
        if train:
            from .drl_training import DRLTrainer
            self.trainer = DRLTrainer(
                self.policy,
                self.state_encoder,
                self.action_decoder,
                batch_size=32,
                update_freq=4,
                warmup_steps=100,
            )

        self._warm_start_metrics: Dict = {}
        self._warm_start_emitted = False
        if train and cfg.offline_warm_start:
            self._warm_start_metrics = _apply_offline_warm_start(
                policy=self.policy,
                cfg=cfg,
                state_dim=self.state_encoder.state_dim,
                action_decoder=self.action_decoder,
            )

        self._last_state = None
        self._last_action_idx = None

    def act(self, state: EnvState) -> List[SliceAction]:
        """
        Select actions using A2C policy.
        """
        state_vec = self.state_encoder.encode(state)
        action_idx, value = self.policy.select_action(state_vec, training=self.train_mode)
        action_vec = self.action_decoder.decode_to_vector(action_idx)
        self._last_state = state
        self._last_action_idx = action_vec
        allocation = self.action_decoder.decode(action_vec)

        actions: List[SliceAction] = []
        for slice_id, alloc in allocation.items():
            target = self.targets[slice_id]
            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=alloc["min_ratio"],
                    max_ratio=alloc["max_ratio"],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        """Update A2C policy with latest transition if training mode is enabled."""
        if not self.train_mode or self.trainer is None:
            return {}
        if self._last_state is None or self._last_action_idx is None:
            return {}

        self.trainer.step(
            self._last_state,
            self._last_action_idx,
            reward,
            next_state,
            done,
        )
        metrics = self.trainer.get_metrics()
        if self._warm_start_metrics and not self._warm_start_emitted:
            metrics.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
            self._warm_start_emitted = True
        return metrics

    def get_training_info(self) -> Dict:
        """Return current training metrics."""
        if self.trainer is None:
            return {}
        info = {
            "step_count": self.policy.train_step_count,
            **self.trainer.get_metrics(),
        }
        if self._warm_start_metrics:
            info.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
        return info

    def save_checkpoint(self, path: str) -> None:
        self.policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_checkpoint(path)


class PPOController(Controller):
    """DRL controller using PPO policy."""

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        cfg: ControllerConfig,
        learning_rate: float = 3e-4,
        train: bool = False,
        device: str = "cpu",
    ):
        self.targets = targets
        self.cfg = cfg
        self.train_mode = train

        self.state_encoder = StateEncoder(targets)
        self.action_decoder = ActionDecoder(targets)

        self.policy = PPOPolicy(
            state_dim=self.state_encoder.state_dim,
            action_dim=self.action_decoder.action_dim,
            n_branches=self.action_decoder.num_slices,
            hidden_dim=128,
            learning_rate=learning_rate,
            gamma=0.99,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            device=device,
        )

        self.trainer = None
        if train:
            self.trainer = DRLTrainer(
                self.policy,
                self.state_encoder,
                self.action_decoder,
                batch_size=32,
                update_freq=4,
                warmup_steps=100,
            )

        self._warm_start_metrics: Dict = {}
        self._warm_start_emitted = False
        if train and cfg.offline_warm_start:
            self._warm_start_metrics = _apply_offline_warm_start(
                policy=self.policy,
                cfg=cfg,
                state_dim=self.state_encoder.state_dim,
                action_decoder=self.action_decoder,
            )

        self._last_state = None
        self._last_action_idx = None
        self._last_log_prob = None

    def act(self, state: EnvState) -> List[SliceAction]:
        state_vec = self.state_encoder.encode(state)
        action_idx, log_prob = self.policy.select_action(state_vec, training=self.train_mode)
        action_vec = self.action_decoder.decode_to_vector(action_idx)
        self._last_state = state
        self._last_action_idx = action_vec
        self._last_log_prob = float(log_prob)

        allocation = self.action_decoder.decode(action_vec)
        actions: List[SliceAction] = []
        for slice_id, alloc in allocation.items():
            target = self.targets[slice_id]
            actions.append(
                SliceAction(
                    slice_id=slice_id,
                    min_ratio=alloc["min_ratio"],
                    max_ratio=alloc["max_ratio"],
                    dedicated_ratio=target.dedicated_ratio,
                )
            )
        return actions

    def train_step(self, reward: float, next_state: EnvState, done: bool = False) -> Dict:
        if not self.train_mode or self.trainer is None:
            return {}
        if self._last_state is None or self._last_action_idx is None:
            return {}

        self.trainer.step(
            self._last_state,
            self._last_action_idx,
            reward,
            next_state,
            done,
            action_info=self._last_log_prob,
        )
        metrics = self.trainer.get_metrics()
        if self._warm_start_metrics and not self._warm_start_emitted:
            metrics.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
            self._warm_start_emitted = True
        return metrics

    def get_training_info(self) -> Dict:
        if self.trainer is None:
            return {}
        info = {
            "step_count": self.policy.train_step_count,
            **self.trainer.get_metrics(),
        }
        if self._warm_start_metrics:
            info.update({f"warm_start_{k}": v for k, v in self._warm_start_metrics.items()})
        return info

    def save_checkpoint(self, path: str) -> None:
        self.policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_checkpoint(path)
