"""Centralized-training, decentralized-execution (CTDE) MARL policy over
the gNB topology graph, on the exact same discrete per-request
accept/reject action space paper #4 validated (action_dim=2, mapped onto a
PRB-ratio ceiling nudge by the existing, unmodified
qoe_oran_framework.action_mapping.AdmissionGate) -- not a continuous
softmax-simplex allocation.

Design, revised after a real training-instability bug was found and
fixed (see below): a shared GATEncoder over the joint cluster state,
producing a per-gNB-node embedding (the "centralized" part -- the encoder
sees every node's raw features and the graph structure, trained end to
end from every agent's loss), feeding a shared per-agent Q-head
(parameters shared across gNBs -- homogeneous/interchangeable agents) that
maps [own-node GAT embedding, slice one-hot] to per-action Q-values for a
pending request at that node. At *action-selection* time each agent
consumes only its own embedding row -- the decentralized-execution half.

Bug found and fixed (M2 first pass): the initial version summed a gNB's
chosen-action Q-values across every request it handled THAT STEP, then
fed the sum into a QMIX-style monotonic mixer (Rashid et al. 2018) for
the TD target. Diagnosed directly (not assumed) by logging train_step's
own loss over 15 training episodes: mean loss rose from ~76 (first 20% of
steps) to ~409 (last 20%) -- diverging, not converging -- and doubling
the training budget (300->600 episodes) made held-out compliance flat-to-
worse, ruling out "just needs more training." Root cause: the number of
pending requests per step varies (0 up to `max_pending_per_step`), so
summing an agent's own per-request Q-values before mixing made the
learning target's scale swing with request count step to step -- an
ill-conditioned, non-stationary regression target no fixed learning rate
tracks, unlike paper #4's own DQN (which never aggregates: every request
is its own independent transition, sharing only the step's scalar reward,
exactly `_store_and_train`'s existing dqn branch in mc_runner.py).

**Fix:** train_step now mirrors that same granularity -- every pending
request in the batch is its own independent TD sample (own chosen action,
own gathered Q, own bootstrapped next-Q), with the step's shared reward
broadcast to every request in that step as its target's immediate reward
(never summed or divided), losses averaged over the batch. The GATEncoder
and QMixer classes are unchanged below; QMixer is kept for a future,
better-conditioned joint-value-factorisation attempt (e.g. mixing at most
one representative Q per agent per step, not a request-count-dependent
sum) but is NOT used in the current train_step -- centralization now
comes from the shared encoder's parameters being updated by every agent's
loss, not from a joint Q_tot target.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_encoder import GATEncoder


class AgentQHead(nn.Module):
    def __init__(self, embed_dim: int, context_dim: int, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, embed: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([embed, context], dim=-1))


class QMixer(nn.Module):
    """QMIX hypernetwork mixer: Q_tot = |W2| . ELU(|W1| . q_agents + b1) + b2,
    with W1/W2/b1/b2 all produced from the global state by small
    hypernetworks, and W1/W2 passed through abs() to enforce
    monotonicity (Rashid et al. 2018, eq. 4-5)."""

    def __init__(self, n_agents: int, global_state_dim: int, mixing_hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(global_state_dim, n_agents * mixing_hidden_dim)
        self.hyper_w2 = nn.Linear(global_state_dim, mixing_hidden_dim)
        self.hyper_b1 = nn.Linear(global_state_dim, mixing_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, mixing_hidden_dim), nn.ReLU(), nn.Linear(mixing_hidden_dim, 1)
        )
        self.mixing_hidden_dim = mixing_hidden_dim

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """agent_qs: [batch, n_agents]; global_state: [batch, global_state_dim].
        Returns Q_tot: [batch]."""
        batch = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch, self.n_agents, self.mixing_hidden_dim)
        b1 = self.hyper_b1(global_state).view(batch, 1, self.mixing_hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # [batch, 1, mixing_hidden_dim]

        w2 = torch.abs(self.hyper_w2(global_state)).view(batch, self.mixing_hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # [batch, 1, 1]
        return q_tot.view(batch)


class GatCtdeNetwork(nn.Module):
    def __init__(self, n_agents: int, node_feat_dim: int, context_dim: int, action_dim: int,
                 gat_hidden_dim: int = 16, gat_out_dim: int = 16, gat_heads: int = 4,
                 q_hidden_dim: int = 32, mixing_hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.encoder = GATEncoder(node_feat_dim, gat_hidden_dim, gat_out_dim, n_heads=gat_heads, n_layers=2)
        self.q_head = AgentQHead(gat_out_dim, context_dim, action_dim, hidden_dim=q_hidden_dim)
        global_state_dim = n_agents * node_feat_dim
        self.mixer = QMixer(n_agents, global_state_dim, mixing_hidden_dim=mixing_hidden_dim)

    def embed(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """node_features: [batch, n_agents, node_feat_dim] -> [batch, n_agents, gat_out_dim]."""
        return self.encoder(node_features, adjacency)

    def agent_q_values(self, embed_row: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """embed_row: [batch, gat_out_dim] (one agent's embedding);
        context: [batch, context_dim] (slice one-hot). Returns [batch, action_dim]."""
        return self.q_head(embed_row, context)


class GatCtdeMarlPolicy:
    """Orchestrates GatCtdeNetwork for action selection (decentralized,
    per-request) and QMIX-style joint training (centralized). Mirrors
    drl_slicing.oranslice_drl.drl_policy.DQNPolicy's epsilon-greedy /
    save_checkpoint / load_checkpoint conventions where they carry over,
    so it plugs into the same experiment-running patterns."""

    def __init__(self, n_agents: int, node_feat_dim: int, context_dim: int, action_dim: int,
                 adjacency: np.ndarray, learning_rate: float = 1e-4, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.985,
                 target_sync_every_episodes: int = 10, device: str = "cpu"):
        """gamma/epsilon_decay default to paper #4's own tuned schedule
        (qoe_oran_framework.policies.dqn_admission.PAPER_TABLE_I_DQN_DEFAULTS),
        matching single_agent_dqn's DQNAdmissionPolicy exactly -- fixed
        during M2 hardening after a real hyperparameter-parity bug was
        found (this class originally used the raw DQNPolicy-style
        defaults, gamma=0.99 and a per-train_step-call epsilon decay that
        collapses to the exploration floor within ~10 episodes of a
        300-episode run -- see independent_dqn_ablation.py's identical
        fix for the full explanation). epsilon now decays ONLY via
        on_episode_end(), called explicitly by
        marl_training.run_episodes_marl at episode boundaries."""
        self.n_agents = n_agents
        self.node_feat_dim = node_feat_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self._per_episode_epsilon_decay = epsilon_decay
        self._target_sync_every_episodes = target_sync_every_episodes
        self._episode_count = 0
        self.device = torch.device(device)
        self.adjacency = torch.tensor(adjacency, dtype=torch.float32, device=self.device)

        self.online = GatCtdeNetwork(n_agents, node_feat_dim, context_dim, action_dim).to(self.device)
        self.target = GatCtdeNetwork(n_agents, node_feat_dim, context_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.train_step_count = 0

    def on_episode_end(self) -> None:
        self._episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self._per_episode_epsilon_decay)
        if self._episode_count % self._target_sync_every_episodes == 0:
            self.target.load_state_dict(self.online.state_dict())

    def select_actions(self, node_features: np.ndarray, requests: List[Tuple[int, np.ndarray]],
                        training: bool = False) -> List[int]:
        """node_features: [n_agents, node_feat_dim] this step's joint state.
        requests: list of (agent_idx, context_onehot) for every pending
        request this step, in the SAME order RANEnv.step() expects actions.
        Returns one discrete action per request (decentralized: each
        action only used that request's own agent's embedding row)."""
        if not requests:
            return []
        with torch.no_grad():
            nf = torch.tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            embeds = self.online.embed(nf, self.adjacency).squeeze(0)  # [n_agents, gat_out_dim]

        actions = []
        for agent_idx, context in requests:
            if training and np.random.rand() < self.epsilon:
                actions.append(int(np.random.randint(0, self.action_dim)))
                continue
            with torch.no_grad():
                ctx_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.online.agent_q_values(embeds[agent_idx].unsqueeze(0), ctx_t)
            actions.append(int(q.argmax(dim=-1).item()))
        return actions

    def train_step(self, batch: Dict) -> Dict:
        """batch keys (all numpy arrays, batch dim first):
        node_features [B, n_agents, node_feat_dim]
        next_node_features [B, n_agents, node_feat_dim]
        agent_request_agent_idx: List[List[int]] length B (ragged: which agent each request in that step belongs to)
        agent_request_context: List[np.ndarray] length B (ragged: [n_requests_b, context_dim])
        agent_request_action: List[np.ndarray] length B (ragged: [n_requests_b])
        rewards [B]; dones [B]
        """
        B = len(batch["rewards"])
        nf = torch.tensor(np.stack(batch["node_features"]), dtype=torch.float32, device=self.device)
        next_nf = torch.tensor(np.stack(batch["next_node_features"]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        embeds = self.online.embed(nf, self.adjacency)  # [B, n_agents, gat_out_dim]
        with torch.no_grad():
            next_embeds = self.target.embed(next_nf, self.adjacency)

        # Per-request TD samples (fix: see module docstring "Bug found and
        # fixed"). Every pending request across the whole batch becomes its
        # own independent (embedding, context, action, reward, next-max-Q,
        # done) sample -- no per-step summation, matching paper #4's own
        # DQN granularity exactly.
        chosen_list, target_list = [], []
        for b in range(B):
            agent_idxs = batch["agent_request_agent_idx"][b]
            contexts = batch["agent_request_context"][b]
            acts = batch["agent_request_action"][b]
            if len(agent_idxs) == 0:
                continue
            ctx_t = torch.tensor(np.asarray(contexts), dtype=torch.float32, device=self.device)
            emb_rows = embeds[b, agent_idxs]  # [n_requests, gat_out_dim]
            q_vals = self.online.agent_q_values(emb_rows, ctx_t)  # [n_requests, action_dim]
            acts_t = torch.tensor(acts, dtype=torch.long, device=self.device)
            chosen = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)  # [n_requests]

            with torch.no_grad():
                next_emb_rows = next_embeds[b, agent_idxs]
                next_q_vals = self.target.agent_q_values(next_emb_rows, ctx_t)
                next_max = next_q_vals.max(dim=-1)[0]  # [n_requests]
                request_target = rewards[b] + self.gamma * next_max * (1 - dones[b])

            chosen_list.append(chosen)
            target_list.append(request_target)

        if not chosen_list:
            return {"loss": 0.0, "grad_norm": 0.0, "epsilon": self.epsilon}

        chosen_all = torch.cat(chosen_list)
        target_all = torch.cat(target_list)
        # Huber/smooth-L1, not MSE: the GAT encoder is a deeper, more
        # expressive function approximator than DQNPolicy's plain MLP, and
        # was found (empirically, via the loss-divergence diagnostic
        # documented above) to be more prone to occasional large TD-error
        # blowups under raw squared error at gamma=0.99's long bootstrap
        # horizon -- the standard DQN-family fix (Mnih et al. 2015 already
        # clips/uses Huber-style loss for exactly this reason).
        loss = F.smooth_l1_loss(chosen_all, target_all)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        # epsilon decay happens ONLY in on_episode_end() now (see __init__
        # docstring) -- this per-100-train-step target sync stays as a
        # harmless safety net on top of on_episode_end()'s per-episode
        # sync, matching DQNAdmissionPolicy's own documented convention.
        if self.train_step_count % 100 == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {"loss": float(loss.item()), "grad_norm": float(grad_norm.item()), "epsilon": self.epsilon}

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.train_step_count = ckpt["train_step_count"]
