"""Rainbow admission policy: DQN + Double-Q target + Dueling architecture +
NoisyNet exploration + Prioritized Experience Replay, over the binary
admission action space (action_dim=2, matching dqn_admission.py /
a2c_admission.py). Nothing here is reused from oranslice_drl -- confirmed
during Stage Zero design survey that no Rainbow implementation exists
anywhere in this codebase; every class below is new.

Table I (papers #1/#2, Rainbow column): gamma=0.95, alpha=0.001,
batch_size=16, target update every 10 episodes (on_episode_end(), same
convention as dqn_admission.py), PER alpha=0.6, PER beta_start=0.4 (beta
annealed to 1.0 over training -- standard PER practice, not itself a
published constant in the papers).
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .._oranslice_path import ensure_oranslice_drl_importable

ensure_oranslice_drl_importable()

from oranslice_drl.drl_policy import RLPolicy, _load_checkpoint_file  # noqa: E402

PAPER_TABLE_I_RAINBOW_DEFAULTS = dict(
    gamma=0.95,
    learning_rate=0.001,
    per_alpha=0.6,
    per_beta_start=0.4,
)


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer (Fortunato et al., 2017) --
    Table I's "NoisyNet" exploration mechanism for Rainbow, replacing
    epsilon-greedy."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features ** 0.5))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class DuelingQNetwork(nn.Module):
    """Shared trunk -> value stream + advantage stream, combined as
    Q = V + (A - mean(A)). Value/advantage heads use NoisyLinear."""

    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value_out = NoisyLinear(hidden_dim, 1)
        self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage_out = NoisyLinear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.trunk(state)
        v = self.value_out(torch.relu(self.value_hidden(x)))
        a = self.advantage_out(torch.relu(self.advantage_hidden(x)))
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self) -> None:
        for m in (self.value_hidden, self.value_out, self.advantage_hidden, self.advantage_out):
            m.reset_noise()


class PrioritizedReplayBuffer:
    """Proportional prioritization via a plain numpy priority array --
    O(n) sampling, acceptable at Stage Zero's scale (a few thousand
    transitions); a sum-tree would be the production-scale upgrade, noted
    as a possible follow-up rather than built here."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        max_priority = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
        transition = (state, int(action), reward, next_state, done)
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, Any]:
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} transitions, need {batch_size}")
        prios = self.priorities[: self.size] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "weights": weights.astype(np.float32),
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, eps: float = 1e-6) -> None:
        self.priorities[indices] = np.abs(td_errors) + eps

    def __len__(self) -> int:
        return self.size


class RainbowAdmissionPolicy(RLPolicy):
    def __init__(self, state_dim: int, hidden_dim: int = 128, device: str = "cpu", **overrides):
        params = {**PAPER_TABLE_I_RAINBOW_DEFAULTS, **overrides}
        self.state_dim = state_dim
        self.action_dim = 2
        self.gamma = params["gamma"]
        self.per_alpha = params["per_alpha"]
        self.per_beta = params["per_beta_start"]
        self.per_beta_end = 1.0
        self.device = torch.device(device)

        self.q_network = DuelingQNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target net's NoisyLinear layers stay deterministic (mu only)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=params["learning_rate"])
        self.train_step_count = 0

    def select_action(self, state: np.ndarray, training: bool = False) -> Tuple[int, float]:
        self.q_network.train(training)  # NoisyLinear noise active only while exploring
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        action = int(q_values.argmax(dim=1).item())
        return action, float(q_values.max().item())

    def train_step(self, batch: Dict) -> Dict:
        self.q_network.train(True)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device).view(-1)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        weights = torch.FloatTensor(batch["weights"]).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)  # Double-Q: online net picks
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (q_values - target_q).detach().cpu().numpy()
        loss = (weights * (q_values - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.q_network.reset_noise()

        self.train_step_count += 1
        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
            "td_errors": td_errors,
            "avg_target_q": float(target_q.mean().item()),
        }

    def on_episode_end(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def anneal_beta(self, fraction: float) -> None:
        """fraction in [0,1]: progress through the offline training
        schedule. Linearly anneals PER beta from beta_start to 1.0 --
        standard PER practice; the papers give beta_start=0.4 but no
        end-of-schedule value, so 1.0 (full importance-sampling
        correction by the end of training) is our documented choice."""
        self.per_beta = self.per_beta + fraction * (self.per_beta_end - self.per_beta)

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step_count": self.train_step_count,
                "per_beta": self.per_beta,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = _load_checkpoint_file(path, self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step_count = ckpt["train_step_count"]
        self.per_beta = ckpt["per_beta"]
