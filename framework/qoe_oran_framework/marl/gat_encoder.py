"""Graph Attention Network encoder (Velickovic et al. 2018, cited in
paper_conf/refs.bib as `gat`) over the gNB topology graph, implemented
directly in torch (no torch_geometric dependency -- confirmed not
installed in this project's venv, and the graphs here are small enough
-- a handful of gNB nodes -- that a dense, masked-attention implementation
is simpler and just as fast as a sparse-message-passing one).

Standard multi-head GAT layer: for each node i, attention coefficients
over its neighbours j (per adjacency mask) are computed from a shared
linear projection plus a learned attention vector, softmax-normalised per
node, then used to aggregate neighbour features. Multiple heads are
concatenated (hidden layers) or averaged (final layer), matching the
original paper's convention.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, concat: bool = True,
                 dropout: float = 0.0, leaky_relu_slope: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """node_features: [batch, N, in_dim]; adjacency: [N, N] (or [batch, N, N])
        binary mask, 1 where an edge (or self-loop) exists. Returns
        [batch, N, n_heads*out_dim] if concat else [batch, N, out_dim]
        (heads averaged)."""
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        batch, n_nodes, _ = node_features.shape
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(batch, -1, -1)

        # [batch, heads, N, out_dim]
        h = torch.einsum("bnf,hfo->bhno", node_features, self.W)

        src_score = torch.einsum("bhno,ho->bhn", h, self.a_src)  # [batch, heads, N]
        dst_score = torch.einsum("bhno,ho->bhn", h, self.a_dst)  # [batch, heads, N]
        # e[b,h,i,j] = LeakyReLU(a_src . h_i + a_dst . h_j)
        e = self.leaky_relu(src_score.unsqueeze(-1) + dst_score.unsqueeze(-2))  # [batch, heads, N, N]

        mask = adjacency.unsqueeze(1) > 0  # [batch, 1, N, N] -> broadcasts over heads
        e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated node (no neighbours at all): softmax of all -inf
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.einsum("bhij,bhjo->bhio", alpha, h)  # [batch, heads, N, out_dim]
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(batch, n_nodes, self.n_heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out


class GATEncoder(nn.Module):
    """Stacks GATLayer(s): hidden layers concat heads + ELU, final layer
    averages heads (standard GAT convention) to produce a fixed-width
    per-node embedding regardless of head count."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert n_layers >= 1
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers.append(GATLayer(d, hidden_dim, n_heads=n_heads, concat=True, dropout=dropout))
            d = hidden_dim * n_heads
        layers.append(GATLayer(d, out_dim, n_heads=n_heads, concat=False, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.out_dim = out_dim

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = node_features
        for i, layer in enumerate(self.layers):
            x = layer(x, adjacency)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x  # [batch, N, out_dim]
