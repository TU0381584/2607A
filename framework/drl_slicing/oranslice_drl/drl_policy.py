"""
DRL Policy implementations: DQN, A2C, and PPO.
Abstract base class for algorithm-agnostic integration.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _load_checkpoint_file(path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint files with secure defaults when supported by torch."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility with older torch versions that lack weights_only.
        return torch.load(path, map_location=device)


class RLPolicy(ABC):
    """Abstract base class for RL policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[Union[int, np.ndarray], float]:
        """
        Select action(s) given state.
        
        Returns:
            action: discrete action index or action array
            action_info: auxiliary info (entropy, value, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Dict) -> Dict:
        """
        Perform one training step on a batch of transitions.
        
        Returns:
            loss_dict: training metrics (loss, grad_norm, etc.)
        """
        raise NotImplementedError

    def behavior_clone_train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        epochs: int = 3,
        batch_size: int = 64,
    ) -> Dict:
        """Optional supervised warm-start hook (override where supported)."""
        raise NotImplementedError("Behavior-cloning warm-start is not implemented for this policy.")

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        raise NotImplementedError


class QNetwork(nn.Module):
    """Deep Q-Network for slice resource allocation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, n_branches: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_branches = max(1, int(n_branches))

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_branches * action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return logits.view(-1, self.n_branches, self.action_dim)


class DQNPolicy(RLPolicy):
    """
    Deep Q-Network (DQN) for discrete action space.
    
    State: continuous, normalized per-slice metrics
    Action: discrete indices for (min_ratio, max_ratio) pairs
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_branches: int = 1,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_branches = max(1, int(n_branches))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, self.n_branches).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim, self.n_branches).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[Union[int, np.ndarray], float]:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: [state_dim] numpy array
            training: if True, use epsilon-greedy; else greedy
            
        Returns:
            action: discrete index
            q_value: Q-value of selected action
        """
        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_dim, size=self.n_branches, dtype=np.int64)
            q_value = 0.0
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            action_t = q_values.argmax(dim=2).squeeze(0)
            action = action_t.cpu().numpy().astype(np.int64)
            q_value = float(q_values.max(dim=2)[0].sum().item())

        if self.n_branches == 1:
            return int(action[0]), q_value
        return action, q_value

    def train_step(self, batch: Dict) -> Dict:
        """
        DQN training step.
        
        Batch must contain:
            states: [batch_size, state_dim]
            actions: [batch_size]
            rewards: [batch_size]
            next_states: [batch_size, state_dim]
            dones: [batch_size]
        """
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(1)

        if actions.shape[1] != self.n_branches:
            raise ValueError(
                f"Expected actions with {self.n_branches} branches, got shape {tuple(actions.shape)}"
            )

        # Compute Q(s,a)
        q_values = self.q_network(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q(s',a') using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=2)[0]
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        if self.train_step_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "epsilon": self.epsilon,
            "avg_target_q": float(target_q_values.mean().item()),
        }

    def behavior_clone_train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        epochs: int = 3,
        batch_size: int = 64,
    ) -> Dict:
        """Supervised warm-start on discrete actions via cross-entropy."""
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 1:
            if self.n_branches != 1:
                raise ValueError(
                    "DQN warm-start data must provide multi-branch action vectors for this policy."
                )
            actions = actions.reshape(-1, 1)

        if actions.shape[1] != self.n_branches:
            raise ValueError(
                f"DQN warm-start action dimension mismatch: expected {self.n_branches}, got {actions.shape[1]}"
            )

        valid = np.all((actions >= 0) & (actions < self.action_dim), axis=1)
        if not np.any(valid):
            raise ValueError("No valid actions available for DQN warm-start.")

        states = states[valid]
        actions = actions[valid]
        if len(states) == 0:
            raise ValueError("Empty dataset after warm-start action filtering.")

        loss_fn = nn.CrossEntropyLoss()
        losses: List[float] = []
        accuracies: List[float] = []

        self.q_network.train()
        for _ in range(max(1, int(epochs))):
            perm = np.random.permutation(len(states))
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_count = 0

            for start in range(0, len(states), max(1, int(batch_size))):
                idx = perm[start : start + max(1, int(batch_size))]
                batch_states = torch.FloatTensor(states[idx]).to(self.device)
                batch_actions = torch.LongTensor(actions[idx]).to(self.device)

                logits = self.q_network(batch_states)
                branch_losses = [
                    loss_fn(logits[:, branch, :], batch_actions[:, branch])
                    for branch in range(self.n_branches)
                ]
                loss = torch.stack(branch_losses).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
                self.optimizer.step()

                preds = logits.argmax(dim=2)
                epoch_correct += int((preds == batch_actions).sum().item())
                epoch_count += int(batch_actions.numel())
                epoch_loss += float(loss.item()) * int(batch_actions.shape[0])

            losses.append(epoch_loss / max(epoch_count, 1))
            accuracies.append(epoch_correct / max(epoch_count, 1))

        # Keep target network aligned with supervised warm-started Q-network.
        self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "bc_loss": float(np.mean(losses)),
            "bc_accuracy": float(np.mean(accuracies)),
            "bc_samples": int(len(states)),
            "bc_epochs": int(max(1, int(epochs))),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = _load_checkpoint_file(path, self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.train_step_count = ckpt["train_step_count"]


class ActorCriticNetwork(nn.Module):
    """Shared Actor-Critic backbone used by A2C and PPO policies."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, n_branches: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_branches = max(1, int(n_branches))

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, self.n_branches * action_dim)  # policy logits
        self.critic = nn.Linear(hidden_dim, 1)  # value estimate

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(state)
        action_logits = self.actor(x).view(-1, self.n_branches, self.action_dim)
        value = self.critic(x)
        return action_logits, value


class A2CPolicy(RLPolicy):
    """
    Advantage Actor-Critic (A2C) policy.
    Placeholder for Phase 2b implementation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_branches: int = 1,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_branches = max(1, int(n_branches))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim, self.n_branches).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[Union[int, np.ndarray], float]:
        """Sample action from policy distribution."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, value = self.network(state_t)
        probs = torch.softmax(action_logits, dim=2)
        action_dist = torch.distributions.Categorical(probs=probs)
        if training:
            action_t = action_dist.sample()
        else:
            action_t = torch.argmax(probs, dim=2)
        action = action_t.squeeze(0).cpu().numpy().astype(np.int64)
        if self.n_branches == 1:
            return int(action[0]), float(value.item())
        return action, float(value.item())

    def train_step(self, batch: Dict) -> Dict:
        """
        A2C training step on batch of transitions.
        
        Algorithm:
        1. Compute value estimates V(s) and V(s')
        2. Compute advantages A(s,a) = r + γ*V(s') - V(s)
        3. Actor loss: -log π(a|s) * A(s,a) (maximize advantage)
        4. Critic loss: MSE(V(s), r + γ*V(s'))
        5. Total loss: actor_loss + critic_loss
        """
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(1)

        if actions.shape[1] != self.n_branches:
            raise ValueError(
                f"Expected actions with {self.n_branches} branches, got shape {tuple(actions.shape)}"
            )

        # Forward pass: get action logits and values
        action_logits, values = self.network(states)
        with torch.no_grad():
            _, next_values = self.network(next_states)

        # Compute advantages: A(s,a) = r + γ*V(s') - V(s)
        target_values = rewards + self.gamma * next_values.squeeze(-1) * (1 - dones)
        advantages = (target_values - values.squeeze(-1)).detach()

        # Actor loss: -log π(a|s) * A(s,a)
        action_dist = torch.distributions.Categorical(probs=torch.softmax(action_logits, dim=2))
        log_probs = action_dist.log_prob(actions).sum(dim=1)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss: MSE(V(s), target_V(s))
        critic_loss = nn.functional.mse_loss(values.squeeze(-1), target_values)

        # Total loss with entropy regularization for exploration
        entropy = action_dist.entropy().mean()
        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1

        return {
            "loss": float(total_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": float(grad_norm.item()),
            "avg_advantage": float(advantages.mean().item()),
            "avg_value": float(values.mean().item()),
        }

    def behavior_clone_train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        epochs: int = 3,
        batch_size: int = 64,
    ) -> Dict:
        """Supervised warm-start of actor head using baseline actions."""
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 1:
            if self.n_branches != 1:
                raise ValueError(
                    "A2C warm-start data must provide multi-branch action vectors for this policy."
                )
            actions = actions.reshape(-1, 1)

        if actions.shape[1] != self.n_branches:
            raise ValueError(
                f"A2C warm-start action dimension mismatch: expected {self.n_branches}, got {actions.shape[1]}"
            )

        valid = np.all((actions >= 0) & (actions < self.action_dim), axis=1)
        if not np.any(valid):
            raise ValueError("No valid actions available for A2C warm-start.")

        states = states[valid]
        actions = actions[valid]
        if len(states) == 0:
            raise ValueError("Empty dataset after warm-start action filtering.")

        loss_fn = nn.CrossEntropyLoss()
        losses: List[float] = []
        accuracies: List[float] = []

        self.network.train()
        for _ in range(max(1, int(epochs))):
            perm = np.random.permutation(len(states))
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_count = 0

            for start in range(0, len(states), max(1, int(batch_size))):
                idx = perm[start : start + max(1, int(batch_size))]
                batch_states = torch.FloatTensor(states[idx]).to(self.device)
                batch_actions = torch.LongTensor(actions[idx]).to(self.device)

                action_logits, _ = self.network(batch_states)
                branch_losses = [
                    loss_fn(action_logits[:, branch, :], batch_actions[:, branch])
                    for branch in range(self.n_branches)
                ]
                loss = torch.stack(branch_losses).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                preds = action_logits.argmax(dim=2)
                epoch_correct += int((preds == batch_actions).sum().item())
                epoch_count += int(batch_actions.numel())
                epoch_loss += float(loss.item()) * int(batch_actions.shape[0])

            losses.append(epoch_loss / max(epoch_count, 1))
            accuracies.append(epoch_correct / max(epoch_count, 1))

        return {
            "bc_loss": float(np.mean(losses)),
            "bc_accuracy": float(np.mean(accuracies)),
            "bc_samples": int(len(states)),
            "bc_epochs": int(max(1, int(epochs))),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = _load_checkpoint_file(path, self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step_count = ckpt["train_step_count"]


class PPOPolicy(RLPolicy):
    """
    Proximal Policy Optimization policy with clipped surrogate loss.

    This implementation uses one-step bootstrapped returns from the replay batch.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_branches: int = 1,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_branches = max(1, int(n_branches))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim, self.n_branches).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[Union[int, np.ndarray], float]:
        """
        Select action from policy distribution.

        Returns:
            action index and log-probability of selected action.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, _ = self.network(state_t)
            probs = torch.softmax(action_logits, dim=2)
            dist = torch.distributions.Categorical(probs=probs)
            if training:
                action_t = dist.sample()
            else:
                action_t = torch.argmax(probs, dim=2)
            log_prob_t = dist.log_prob(action_t)

        action = action_t.squeeze(0).cpu().numpy().astype(np.int64)
        log_prob = float(log_prob_t.sum().item())
        if self.n_branches == 1:
            return int(action[0]), log_prob
        return action, log_prob

    def train_step(self, batch: Dict) -> Dict:
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(1)

        if actions.shape[1] != self.n_branches:
            raise ValueError(
                f"Expected actions with {self.n_branches} branches, got shape {tuple(actions.shape)}"
            )

        # Old log-probs are captured at action time by the controller.
        old_log_probs = batch.get("action_info")
        if old_log_probs is None:
            old_log_probs_t = None
        else:
            old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
            if torch.isnan(old_log_probs_t).any():
                old_log_probs_t = None

        action_logits, values = self.network(states)
        values = values.squeeze(-1)

        with torch.no_grad():
            _, next_values = self.network(next_states)
            next_values = next_values.squeeze(-1)
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values

        dist = torch.distributions.Categorical(probs=torch.softmax(action_logits, dim=2))
        new_log_probs = dist.log_prob(actions).sum(dim=1)

        if old_log_probs_t is None:
            old_log_probs_t = new_log_probs.detach()

        ratios = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1

        approx_kl = (old_log_probs_t - new_log_probs).mean().abs()
        clip_fraction = ((ratios - 1.0).abs() > self.clip_epsilon).float().mean()

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "grad_norm": float(grad_norm.item()),
            "avg_advantage": float(advantages.mean().item()),
            "avg_value": float(values.mean().item()),
        }

    def behavior_clone_train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        epochs: int = 3,
        batch_size: int = 64,
    ) -> Dict:
        """Supervised warm-start of PPO actor head from baseline actions."""
        actions = np.asarray(actions, dtype=np.int64)
        if actions.ndim == 1:
            if self.n_branches != 1:
                raise ValueError(
                    "PPO warm-start data must provide multi-branch action vectors for this policy."
                )
            actions = actions.reshape(-1, 1)

        if actions.shape[1] != self.n_branches:
            raise ValueError(
                f"PPO warm-start action dimension mismatch: expected {self.n_branches}, got {actions.shape[1]}"
            )

        valid = np.all((actions >= 0) & (actions < self.action_dim), axis=1)
        if not np.any(valid):
            raise ValueError("No valid actions available for PPO warm-start.")

        states = states[valid]
        actions = actions[valid]
        if len(states) == 0:
            raise ValueError("Empty dataset after warm-start action filtering.")

        loss_fn = nn.CrossEntropyLoss()
        losses: List[float] = []
        accuracies: List[float] = []

        self.network.train()
        for _ in range(max(1, int(epochs))):
            perm = np.random.permutation(len(states))
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_count = 0

            for start in range(0, len(states), max(1, int(batch_size))):
                idx = perm[start : start + max(1, int(batch_size))]
                batch_states = torch.FloatTensor(states[idx]).to(self.device)
                batch_actions = torch.LongTensor(actions[idx]).to(self.device)

                action_logits, _ = self.network(batch_states)
                branch_losses = [
                    loss_fn(action_logits[:, branch, :], batch_actions[:, branch])
                    for branch in range(self.n_branches)
                ]
                loss = torch.stack(branch_losses).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                preds = action_logits.argmax(dim=2)
                epoch_correct += int((preds == batch_actions).sum().item())
                epoch_count += int(batch_actions.numel())
                epoch_loss += float(loss.item()) * int(batch_actions.shape[0])

            losses.append(epoch_loss / max(epoch_count, 1))
            accuracies.append(epoch_correct / max(epoch_count, 1))

        return {
            "bc_loss": float(np.mean(losses)),
            "bc_accuracy": float(np.mean(accuracies)),
            "bc_samples": int(len(states)),
            "bc_epochs": int(max(1, int(epochs))),
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = _load_checkpoint_file(path, self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step_count = ckpt["train_step_count"]
