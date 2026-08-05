"""Ablation 2 of M2: N independent per-gNB DQN policies, NO topology
sharing and NO parameter sharing -- each gNB's own DQNPolicy (reusing
framework/drl_slicing's existing DQNPolicy class unchanged, the same
class paper #4's single-agent arm uses) sees ONLY its own node's local
[prb_used_ratio, congestion_level, queue_len_norm]-per-slice features plus
its own request's slice one-hot -- the identical per-agent input the
GAT-CTDE arm's AgentQHead consumes (same node_feat_dim + context_dim),
so the only architectural difference between this ablation and GAT-CTDE
is the presence/absence of the GAT encoder and the centralized QMIX
mixer -- isolating the GAT/CTDE contribution specifically, not conflating
it with an input-feature difference.

This is intentionally NOT paper #4's single-agent-DQN-on-the-flattened-
joint-state arm (that is ablation 1, built by simply running the existing
qoe_oran_framework training pipeline unmodified against an N-gNB config --
no new code needed for it at all, see experiments/scripts/m2_run_experiment.py).
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../framework, for oranslice_drl
from oranslice_drl.drl_policy import DQNPolicy  # noqa: E402


class IndependentPerGnbDqnPolicy:
    """Owns n_agents separate DQNPolicy instances, addressed by gNB index.
    No cross-agent communication anywhere -- each policy's select_action/
    train_step call only ever sees that one agent's own local state.

    Hyperparameters default to paper #4's own tuned schedule
    (qoe_oran_framework.policies.dqn_admission.PAPER_TABLE_I_DQN_DEFAULTS:
    gamma=0.95, epsilon_decay=0.985 applied ONCE PER EPISODE via
    on_episode_end(), not per train_step() call), matching
    single_agent_dqn's DQNAdmissionPolicy exactly -- fixed here after a
    real hyperparameter-parity bug was found during M2 hardening: this
    class originally left the base DQNPolicy's raw defaults in place
    (gamma=0.99, epsilon_decay=0.995 applied per train_step() call, which
    this project's own dqn_admission.py module docstring already
    documents collapses epsilon to its floor within ~600 steps -- under
    10 episodes of a 300-episode run, leaving ~97% of training nearly
    greedy with no real exploration). GatCtdeMarlPolicy had the identical
    bug and was fixed the same way. Passing epsilon_decay=1.0 to the
    underlying DQNPolicy freezes its own per-call decay; the real decay
    happens in on_episode_end() below, called explicitly by
    marl_training.run_episodes_marl at episode boundaries -- mirroring
    mc_runner.run_single's identical explicit on_episode_end() call for
    algorithm in ("dqn", "rainbow")."""

    def __init__(self, n_agents: int, node_feat_dim: int, context_dim: int, action_dim: int,
                 learning_rate: float = 1e-3, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.985,
                 target_sync_every_episodes: int = 10, device: str = "cpu"):
        self.n_agents = n_agents
        self.state_dim = node_feat_dim + context_dim
        self.action_dim = action_dim
        self._per_episode_epsilon_decay = epsilon_decay
        self._target_sync_every_episodes = target_sync_every_episodes
        self._episode_count = 0
        self.agents: List[DQNPolicy] = [
            DQNPolicy(self.state_dim, action_dim, n_branches=1, learning_rate=learning_rate,
                      gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                      epsilon_decay=1.0, device=device)
            for _ in range(n_agents)
        ]

    def on_episode_end(self) -> None:
        self._episode_count += 1
        for agent in self.agents:
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * self._per_episode_epsilon_decay)
            if self._episode_count % self._target_sync_every_episodes == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())

    def select_actions(self, node_features: np.ndarray, requests: List[Tuple[int, np.ndarray]],
                        training: bool = False) -> List[int]:
        actions = []
        for agent_idx, context in requests:
            local_state = np.concatenate([node_features[agent_idx], context]).astype(np.float32)
            action, _ = self.agents[agent_idx].select_action(local_state, training=training)
            actions.append(int(action))
        return actions

    def train_step(self, batch: Dict) -> Dict:
        """Same ragged batch shape as GatCtdeMarlPolicy.train_step, but
        each agent trains ONLY on its own requests -- no shared network,
        no joint target, no mixer. Returns per-agent loss dict."""
        B = len(batch["rewards"])
        per_agent_states: Dict[int, list] = {i: [] for i in range(self.n_agents)}
        per_agent_next_states: Dict[int, list] = {i: [] for i in range(self.n_agents)}
        per_agent_actions: Dict[int, list] = {i: [] for i in range(self.n_agents)}
        per_agent_rewards: Dict[int, list] = {i: [] for i in range(self.n_agents)}
        per_agent_dones: Dict[int, list] = {i: [] for i in range(self.n_agents)}

        for b in range(B):
            nf = batch["node_features"][b]
            next_nf = batch["next_node_features"][b]
            agent_idxs = batch["agent_request_agent_idx"][b]
            contexts = batch["agent_request_context"][b]
            acts = batch["agent_request_action"][b]
            reward = batch["rewards"][b]
            done = batch["dones"][b]
            for i, agent_idx in enumerate(agent_idxs):
                local_state = np.concatenate([nf[agent_idx], contexts[i]]).astype(np.float32)
                local_next_state = np.concatenate([next_nf[agent_idx], contexts[i]]).astype(np.float32)
                per_agent_states[agent_idx].append(local_state)
                per_agent_next_states[agent_idx].append(local_next_state)
                per_agent_actions[agent_idx].append(acts[i])
                per_agent_rewards[agent_idx].append(reward)
                per_agent_dones[agent_idx].append(done)

        losses = {}
        for agent_idx in range(self.n_agents):
            if not per_agent_states[agent_idx]:
                continue
            agent_batch = {
                "states": np.stack(per_agent_states[agent_idx]),
                "actions": np.array(per_agent_actions[agent_idx], dtype=np.int64),
                "rewards": np.array(per_agent_rewards[agent_idx], dtype=np.float32),
                "next_states": np.stack(per_agent_next_states[agent_idx]),
                "dones": np.array(per_agent_dones[agent_idx], dtype=np.float32),
            }
            info = self.agents[agent_idx].train_step(agent_batch)
            losses[f"agent{agent_idx}_loss"] = info["loss"]
        return losses

    def save_checkpoint(self, path: str) -> None:
        import torch
        torch.save({f"agent{i}": self.agents[i].q_network.state_dict() for i in range(self.n_agents)}
                   | {f"agent{i}_target": self.agents[i].target_network.state_dict() for i in range(self.n_agents)},
                   path)

    def load_checkpoint(self, path: str) -> None:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for i in range(self.n_agents):
            self.agents[i].q_network.load_state_dict(ckpt[f"agent{i}"])
            self.agents[i].target_network.load_state_dict(ckpt[f"agent{i}_target"])
