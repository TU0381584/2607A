"""M3: federated variant of the GAT-CTDE arm (docs/PAPER5_M3_fl_dp.md).

Architecture: each gNB is an FL client with its OWN private copy of a
LocalGatQNetwork (the same GATEncoder + AgentQHead pair gat_ctde.py's
GatCtdeNetwork uses, minus the QMixer -- which the centralized arm's own
train_step doesn't use either, see ctde_policy.py's module docstring, so
dropping it here isolates "federated local training + periodic FedAvg"
as the ONLY architectural difference from gat_ctde, exactly the isolation
principle M2's independent_dqn ablation already established for a
different axis). Every client's GAT encoder still consumes the FULL joint
node-feature snapshot as input (same environment observability every
other arm gets -- RANEnv broadcasts cluster state to whichever policy is
running, this is a modelling given, not a privacy leak: it is a snapshot
of already-public per-slice PRB/queue occupancy, not another client's raw
admission-decision training data). What never crosses a client boundary
is: (a) any client's own replay transitions (this class enforces the same
per-request agent-ownership partition IndependentPerGnbDqnPolicy already
demonstrates -- an agent's train_step contribution is built only from
requests belonging to its own gNB), and (b) any client's own gradients
(clipped and, optionally, DP-noised in place via dp_sgd.clip_and_noise_
BEFORE that client's own local optimizer.step() -- so even the local
update itself is privatized, not just the round-boundary aggregate).

Federation happens at fixed intervals (`local_steps_per_round` train_step
calls): the server (fl_aggregation.fedavg_aggregate) averages all n_agents
clients' online state_dicts and broadcasts the average back to every
client, mirroring standard FedAvg/DP-FedAvg round structure (McMahan et
al. 2017; Abadi et al. 2016 for the per-client DP-SGD step). FedProx (Li
et al. 2020) is available as `aggregator="fedprox"`: identical server-side
averaging, but each client's local loss gains a proximal term pulling its
weights toward the last-broadcast global snapshot (see _local_loss below)
-- the standard way FedProx differs from FedAvg, entirely on the client
side.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dp_sgd
from .fl_aggregation import fedavg_aggregate
from .gat_encoder import GATEncoder
from .ctde_policy import AgentQHead


class LocalGatQNetwork(nn.Module):
    def __init__(self, node_feat_dim: int, context_dim: int, action_dim: int,
                 gat_hidden_dim: int = 16, gat_out_dim: int = 16, gat_heads: int = 4,
                 q_hidden_dim: int = 32):
        super().__init__()
        self.encoder = GATEncoder(node_feat_dim, gat_hidden_dim, gat_out_dim, n_heads=gat_heads, n_layers=2)
        self.q_head = AgentQHead(gat_out_dim, context_dim, action_dim, hidden_dim=q_hidden_dim)

    def embed(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_features, adjacency)

    def agent_q_values(self, embed_row: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.q_head(embed_row, context)


class FederatedGatPolicy:
    """Same select_actions(node_features, requests, training) /
    train_step(batch) / on_episode_end() interface as GatCtdeMarlPolicy
    and IndependentPerGnbDqnPolicy, so it plugs into
    marl_training.run_episodes_marl unchanged -- no new training-loop
    code needed for this arm."""

    def __init__(self, n_agents: int, node_feat_dim: int, context_dim: int, action_dim: int,
                 adjacency: np.ndarray, learning_rate: float = 1e-4, gamma: float = 0.95,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.985,
                 target_sync_every_episodes: int = 10,
                 aggregator: str = "fedavg", fedprox_mu: float = 0.0,
                 local_steps_per_round: int = 50,
                 dp_clip_norm: float = 1.0, dp_noise_multiplier: float = 0.0,
                 dp_seed: Optional[int] = None, device: str = "cpu"):
        if aggregator not in ("fedavg", "fedprox"):
            raise ValueError(f"unknown aggregator {aggregator!r}")
        if aggregator == "fedavg" and fedprox_mu != 0.0:
            raise ValueError("fedprox_mu must be 0.0 under aggregator='fedavg'")

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self._per_episode_epsilon_decay = epsilon_decay
        self._target_sync_every_episodes = target_sync_every_episodes
        self._episode_count = 0
        self.device = torch.device(device)
        self.adjacency = torch.tensor(adjacency, dtype=torch.float32, device=self.device)

        self.aggregator = aggregator
        self.fedprox_mu = fedprox_mu
        self.local_steps_per_round = local_steps_per_round
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self._dp_generator = None
        if dp_seed is not None:
            self._dp_generator = torch.Generator(device=self.device)
            self._dp_generator.manual_seed(dp_seed)

        self.clients = [
            LocalGatQNetwork(node_feat_dim, context_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ]
        # Server-initialized global model: every client starts from the
        # SAME weights (client 0's random init, broadcast) -- standard FL
        # round-0 setup, not independent random inits per client.
        init_state = self.clients[0].state_dict()
        for c in self.clients[1:]:
            c.load_state_dict(init_state)
        self.targets = [
            LocalGatQNetwork(node_feat_dim, context_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ]
        for t, c in zip(self.targets, self.clients):
            t.load_state_dict(c.state_dict())
            t.eval()

        self.optimizers = [torch.optim.Adam(c.parameters(), lr=learning_rate) for c in self.clients]
        self.global_snapshot = {k: v.clone().detach() for k, v in init_state.items()}

        self.local_steps_since_round = 0
        self.round_count = 0
        self.dp_step_count = [0 for _ in range(n_agents)]
        self.train_step_count = 0

    def on_episode_end(self) -> None:
        self._episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self._per_episode_epsilon_decay)
        if self._episode_count % self._target_sync_every_episodes == 0:
            for t, c in zip(self.targets, self.clients):
                t.load_state_dict(c.state_dict())

    def select_actions(self, node_features: np.ndarray, requests: List[Tuple[int, np.ndarray]],
                        training: bool = False) -> List[int]:
        if not requests:
            return []
        nf = torch.tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = []
        # Each client embeds the joint state with its OWN online network,
        # then keeps only its own node's embedding row (decentralized
        # execution: only the acting agent's own weights and own row are
        # used to pick that agent's action).
        own_row_by_agent: Dict[int, torch.Tensor] = {}
        for agent_idx, context in requests:
            if training and np.random.rand() < self.epsilon:
                actions.append(int(np.random.randint(0, self.action_dim)))
                continue
            if agent_idx not in own_row_by_agent:
                with torch.no_grad():
                    full_embed = self.clients[agent_idx].embed(nf, self.adjacency).squeeze(0)
                    own_row_by_agent[agent_idx] = full_embed[agent_idx]
            with torch.no_grad():
                ctx_t = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.clients[agent_idx].agent_q_values(own_row_by_agent[agent_idx].unsqueeze(0), ctx_t)
            actions.append(int(q.argmax(dim=-1).item()))
        return actions

    def _local_loss(self, agent_idx: int, chosen: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.smooth_l1_loss(chosen, target)
        if self.fedprox_mu > 0:
            prox = torch.zeros((), device=self.device)
            for name, p in self.clients[agent_idx].named_parameters():
                prox = prox + (p - self.global_snapshot[name]).pow(2).sum()
            loss = loss + 0.5 * self.fedprox_mu * prox
        return loss

    def train_step(self, batch: Dict) -> Dict:
        """Same ragged batch shape as GatCtdeMarlPolicy.train_step. Splits
        requests by owning agent (like IndependentPerGnbDqnPolicy), but
        each agent's forward pass still embeds the FULL joint node-feature
        snapshot with its OWN local network (same GAT-attention input
        every other arm gets) -- only the loss terms (which requests
        contribute) and the network weights are agent-private."""
        B = len(batch["rewards"])
        nf = torch.tensor(np.stack(batch["node_features"]), dtype=torch.float32, device=self.device)
        next_nf = torch.tensor(np.stack(batch["next_node_features"]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        per_agent_b: Dict[int, List[int]] = {i: [] for i in range(self.n_agents)}
        per_agent_ctx: Dict[int, List[np.ndarray]] = {i: [] for i in range(self.n_agents)}
        per_agent_act: Dict[int, List[int]] = {i: [] for i in range(self.n_agents)}
        for b in range(B):
            agent_idxs = batch["agent_request_agent_idx"][b]
            contexts = batch["agent_request_context"][b]
            acts = batch["agent_request_action"][b]
            for i, agent_idx in enumerate(agent_idxs):
                per_agent_b[agent_idx].append(b)
                per_agent_ctx[agent_idx].append(contexts[i])
                per_agent_act[agent_idx].append(acts[i])

        losses: Dict[str, float] = {}
        any_update = False
        for agent_idx in range(self.n_agents):
            b_idxs = per_agent_b[agent_idx]
            if not b_idxs:
                continue
            any_update = True
            b_t = torch.tensor(b_idxs, dtype=torch.long, device=self.device)
            embeds = self.clients[agent_idx].embed(nf[b_t], self.adjacency)[:, agent_idx]
            with torch.no_grad():
                next_embeds = self.targets[agent_idx].embed(next_nf[b_t], self.adjacency)[:, agent_idx]

            ctx_t = torch.tensor(np.asarray(per_agent_ctx[agent_idx]), dtype=torch.float32, device=self.device)
            q_vals = self.clients[agent_idx].agent_q_values(embeds, ctx_t)
            acts_t = torch.tensor(per_agent_act[agent_idx], dtype=torch.long, device=self.device)
            chosen = q_vals.gather(1, acts_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_vals = self.targets[agent_idx].agent_q_values(next_embeds, ctx_t)
                next_max = next_q_vals.max(dim=-1)[0]
                request_target = rewards[b_t] + self.gamma * next_max * (1 - dones[b_t])

            loss = self._local_loss(agent_idx, chosen, request_target)
            self.optimizers[agent_idx].zero_grad()
            loss.backward()
            if self.dp_noise_multiplier > 0:
                dp_sgd.clip_and_noise_(
                    self.clients[agent_idx].parameters(), self.dp_clip_norm,
                    self.dp_noise_multiplier, generator=self._dp_generator,
                )
                self.dp_step_count[agent_idx] += 1
            else:
                nn.utils.clip_grad_norm_(self.clients[agent_idx].parameters(), max_norm=self.dp_clip_norm)
            self.optimizers[agent_idx].step()
            losses[f"agent{agent_idx}_loss"] = float(loss.item())

        if not any_update:
            return {"loss": 0.0, "epsilon": self.epsilon, "round": self.round_count}

        self.train_step_count += 1
        self.local_steps_since_round += 1
        if self.local_steps_since_round >= self.local_steps_per_round:
            self._aggregate_round()
            self.local_steps_since_round = 0

        losses["epsilon"] = self.epsilon
        losses["round"] = self.round_count
        return losses

    def _aggregate_round(self) -> None:
        state_dicts = [c.state_dict() for c in self.clients]
        avg = fedavg_aggregate(state_dicts)
        for c in self.clients:
            c.load_state_dict(avg)
        self.global_snapshot = {k: v.clone().detach() for k, v in avg.items()}
        self.round_count += 1

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "clients": [c.state_dict() for c in self.clients],
            "targets": [t.state_dict() for t in self.targets],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "epsilon": self.epsilon,
            "train_step_count": self.train_step_count,
            "round_count": self.round_count,
            "dp_step_count": self.dp_step_count,
            "aggregator": self.aggregator,
            "fedprox_mu": self.fedprox_mu,
            "dp_clip_norm": self.dp_clip_norm,
            "dp_noise_multiplier": self.dp_noise_multiplier,
            "local_steps_per_round": self.local_steps_per_round,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for c, sd in zip(self.clients, ckpt["clients"]):
            c.load_state_dict(sd)
        for t, sd in zip(self.targets, ckpt["targets"]):
            t.load_state_dict(sd)
        for o, sd in zip(self.optimizers, ckpt["optimizers"]):
            o.load_state_dict(sd)
        self.epsilon = ckpt["epsilon"]
        self.train_step_count = ckpt["train_step_count"]
        self.round_count = ckpt["round_count"]
        self.dp_step_count = ckpt["dp_step_count"]
