"""
DRL training infrastructure: replay buffer, state encoding, and training loop.
"""
from typing import Dict, List, Tuple, Union
from collections import deque
import numpy as np
from .types import EnvState, SliceMetrics, SliceAction
from .config import SliceTarget


class StateEncoder:
    """
    Encode per-slice metrics into a normalized state vector.
    
    State dimensions (per slice):
        0: efficiency = achieved_throughput / offered_load
        1: latency_ratio = latency_ms / sla_budget_ms
        2: loss_ratio = loss_pct / sla_budget_pct
        3-5: queue depth (future), queue trend (future), slice_type_id
    
    Total for 3 slices: 18-dim state (or 6-dim if minimal)
    """

    def __init__(self, targets: Dict[str, SliceTarget]):
        self.targets = targets
        self.slice_ids = sorted(list(targets.keys()))
        self.n_slices = len(self.slice_ids)
        # Minimal state: 3 metrics per slice (efficiency, latency_sla, loss_sla)
        self.state_dim = self.n_slices * 3

    def encode(self, state: EnvState) -> np.ndarray:
        """
        Encode EnvState to normalized float32 vector.
        
        Args:
            state: EnvState with per-slice metrics
            
        Returns:
            [state_dim] float32 array, values in [0, 1] or slightly beyond for SLA
        """
        features = []
        for slice_id in self.slice_ids:
            if slice_id not in state.slices:
                # Missing data: zero out
                features.extend([0.0, 0.0, 0.0])
                continue

            metrics = state.slices[slice_id]
            target = self.targets[slice_id]

            # Efficiency: achieved / offered
            offered = max(metrics.offered_load_kbps, 1.0)
            efficiency = min(metrics.throughput_kbps / offered, 1.0)

            # Latency SLA ratio (can exceed 1.0 if violated)
            latency_sla = metrics.latency_ms / max(target.latency_budget_ms, 1e-6)
            latency_sla = np.clip(latency_sla, 0.0, 2.0)  # Clip to [0, 2]

            # Loss SLA ratio (can exceed 1.0 if violated)
            loss_sla = metrics.loss_pct / max(target.loss_budget_pct, 1e-6)
            loss_sla = np.clip(loss_sla, 0.0, 2.0)

            features.extend([efficiency, latency_sla, loss_sla])

        return np.array(features, dtype=np.float32)


class ActionDecoder:
    """
    Decode discrete action index to slice resource allocation.
    
    Action space per slice: Cartesian product of min_ratio and max_ratio.
    This decoder uses a factorized multi-discrete representation where each
    slice owns one branch with `action_per_slice = ratio_steps * ratio_steps`
    options. The full joint space remains representable as
    `flat_action_dim = action_per_slice ** n_slices` for logging/warm-start.
    """

    def __init__(
        self,
        targets: Dict[str, SliceTarget],
        ratio_steps: int = 12,  # 12 steps for each min/max bound
    ):
        self.targets = targets
        self.slice_ids = sorted(list(targets.keys()))
        self.num_slices = len(self.slice_ids)
        self.ratio_steps = ratio_steps

        # Build mapping from ratio value to index
        self.ratio_bins = self._build_ratio_bins()
        self.action_per_slice = ratio_steps * ratio_steps
        self.action_dim = self.action_per_slice
        self.flat_action_dim = self.action_per_slice ** max(self.num_slices, 1)

    def _build_ratio_bins(self) -> Dict[str, List[int]]:
        """Create discrete ratio bins across all slices."""
        bins = {}
        for slice_id, target in self.targets.items():
            min_lo, min_hi = target.min_ratio_bounds
            max_lo, max_hi = target.max_ratio_bounds

            min_bins = np.linspace(min_lo, min_hi, self.ratio_steps).astype(int).tolist()
            max_bins = np.linspace(max_lo, max_hi, self.ratio_steps).astype(int).tolist()

            bins[slice_id] = {"min": min_bins, "max": max_bins}
        return bins

    def _decode_scalar_to_vector(self, action: int) -> np.ndarray:
        """Backward-compatible scalar decode into per-slice action branches."""
        value = int(action)
        branches: List[int] = []
        for _ in self.slice_ids:
            branches.append(value % self.action_per_slice)
            value = value // self.action_per_slice
        return np.array(branches, dtype=np.int64)

    def decode(self, action: Union[int, List[int], np.ndarray]) -> Dict[str, Dict[str, int]]:
        """
        Convert action branch indices to per-slice allocation.
        
        Returns:
            { slice_id: { 'min_ratio': int, 'max_ratio': int }, ... }
        """
        if isinstance(action, np.ndarray):
            action_vector = action.astype(np.int64).reshape(-1)
        elif isinstance(action, (list, tuple)):
            action_vector = np.array(action, dtype=np.int64).reshape(-1)
        else:
            action_vector = self._decode_scalar_to_vector(int(action))

        if action_vector.shape[0] < self.num_slices:
            # Pad missing branches with zeros for robustness.
            padding = np.zeros(self.num_slices - action_vector.shape[0], dtype=np.int64)
            action_vector = np.concatenate([action_vector, padding], axis=0)

        allocation = {}

        for i, slice_id in enumerate(self.slice_ids):
            slice_action = int(action_vector[i]) % self.action_per_slice

            min_idx = slice_action % self.ratio_steps
            max_idx = slice_action // self.ratio_steps

            bins = self.ratio_bins[slice_id]
            min_ratio = bins["min"][min_idx]
            max_ratio = bins["max"][max_idx]

            # Ensure min <= max
            if min_ratio > max_ratio:
                min_ratio = max_ratio

            allocation[slice_id] = {
                "min_ratio": int(min_ratio),
                "max_ratio": int(max_ratio),
            }

        return allocation

    def _nearest_bin_index(self, bins: List[int], value: int) -> int:
        """Map a ratio value to the closest discrete bin index."""
        distances = [abs(bin_value - value) for bin_value in bins]
        return int(np.argmin(distances))

    def encode_allocation(self, allocation: Dict[str, Dict[str, int]]) -> np.ndarray:
        """
        Convert per-slice allocation dictionary to factorized branch indices.

        Args:
            allocation: {slice_id: {"min_ratio": int, "max_ratio": int}}

        Returns:
            [n_slices] int64 array compatible with `decode`.
        """
        action_indices: List[int] = []

        for slice_id in self.slice_ids:
            slice_alloc = allocation.get(slice_id)
            if slice_alloc is None:
                min_idx = 0
                max_idx = 0
            else:
                min_ratio = int(slice_alloc.get("min_ratio", self.ratio_bins[slice_id]["min"][0]))
                max_ratio = int(slice_alloc.get("max_ratio", self.ratio_bins[slice_id]["max"][0]))
                if min_ratio > max_ratio:
                    min_ratio = max_ratio
                min_idx = self._nearest_bin_index(self.ratio_bins[slice_id]["min"], min_ratio)
                max_idx = self._nearest_bin_index(self.ratio_bins[slice_id]["max"], max_ratio)

            slice_action = min_idx + (max_idx * self.ratio_steps)
            action_indices.append(int(slice_action))

        return np.array(action_indices, dtype=np.int64)

    def flatten_action_vector(self, action_vector: Union[List[int], np.ndarray]) -> int:
        """Encode factorized branch indices as one scalar (for logging only)."""
        vec = np.array(action_vector, dtype=np.int64).reshape(-1)
        if vec.shape[0] < self.num_slices:
            vec = np.pad(vec, (0, self.num_slices - vec.shape[0]), mode="constant")

        value = 0
        multiplier = 1
        for idx in range(self.num_slices):
            value += int(vec[idx] % self.action_per_slice) * multiplier
            multiplier *= self.action_per_slice
        return int(value)

    def decode_to_vector(self, action: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """Normalize scalar/list/array action formats to [n_slices] int64 vector."""
        if isinstance(action, np.ndarray):
            vec = action.astype(np.int64).reshape(-1)
        elif isinstance(action, (list, tuple)):
            vec = np.array(action, dtype=np.int64).reshape(-1)
        else:
            vec = self._decode_scalar_to_vector(int(action))

        if vec.shape[0] < self.num_slices:
            vec = np.pad(vec, (0, self.num_slices - vec.shape[0]), mode="constant")
        elif vec.shape[0] > self.num_slices:
            vec = vec[: self.num_slices]

        return vec

    def encode_scalar(self, action: Union[int, List[int], np.ndarray]) -> int:
        """Encode any action format into legacy scalar index (for compatibility only)."""
        return self.flatten_action_vector(self.decode_to_vector(action))

    def encode_actions(self, actions: List[SliceAction]) -> np.ndarray:
        """Convert list of SliceAction objects to factorized action vector."""
        allocation = {
            action.slice_id: {"min_ratio": int(action.min_ratio), "max_ratio": int(action.max_ratio)}
            for action in actions
        }
        return self.encode_allocation(allocation)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray, List[int]],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_info: float = None,
    ) -> None:
        """Add transition to buffer."""
        action_array = np.atleast_1d(np.asarray(action, dtype=np.int64))
        self.buffer.append((state, action_array, reward, next_state, done, action_info))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch from buffer."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has {len(self.buffer)} transitions, need {batch_size}")

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, action_infos = zip(
            *[self.buffer[i] for i in indices]
        )

        action_info_array = np.array(
            [np.nan if value is None else float(value) for value in action_infos],
            dtype=np.float32,
        )

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "action_info": action_info_array,
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def reset(self) -> None:
        """Clear buffer."""
        self.buffer.clear()


class DRLTrainer:
    """
    Trainer for DRL policies: manages training loop, replay buffer, and logging.
    """

    def __init__(
        self,
        policy,
        state_encoder: StateEncoder,
        action_decoder: ActionDecoder,
        batch_size: int = 32,
        update_freq: int = 4,
        warmup_steps: int = 1000,
    ):
        self.policy = policy
        self.state_encoder = state_encoder
        self.action_decoder = action_decoder
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.step_count = 0
        self.episode_count = 0
        self.training_metrics = []

    def step(
        self,
        state: EnvState,
        action: Union[int, np.ndarray, List[int]],
        reward: float,
        next_state: EnvState,
        done: bool,
        action_info: float = None,
    ) -> None:
        """Record transition and update policy if ready."""
        state_vec = self.state_encoder.encode(state)
        next_state_vec = self.state_encoder.encode(next_state)
        self.replay_buffer.add(state_vec, action, reward, next_state_vec, done, action_info)

        self.step_count += 1

        # Train after warmup period
        if self.step_count >= self.warmup_steps and self.step_count % self.update_freq == 0:
            loss_dict = self._train_batch()
            self.training_metrics.append(loss_dict)

    def _train_batch(self) -> Dict[str, float]:
        """Sample batch and update policy."""
        batch = self.replay_buffer.sample(self.batch_size)
        return self.policy.train_step(batch)

    def get_metrics(self) -> Dict[str, float]:
        """Aggregate training metrics."""
        if not self.training_metrics:
            return {}

        keys = self.training_metrics[0].keys()
        return {
            key: np.mean([m[key] for m in self.training_metrics[-100:]])  # Last 100 steps
            for key in keys
        }

    def reset_metrics(self) -> None:
        """Clear accumulated metrics."""
        self.training_metrics.clear()
