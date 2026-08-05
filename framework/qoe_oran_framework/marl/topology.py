"""Graph topology over gNB nodes for the GAT encoder.

No real multi-gNB physical topology (inter-site distances, backhaul graph,
neighbor relations) exists anywhere in this repo's configs or docs -- this
project's live rig is single-gNB (paper #4), and paper #2's offline
multi-gNB LB-extension config treats every gNB as an interchangeable peer
with no notion of adjacency at all (see qoe_oran_framework/env.py's
encode_state(), which concatenates every gNB into one flat vector with no
graph structure). Absent a specified real topology, this defaults to
fully-connected (every gNB attends to every other gNB) -- the least
assumption-laden choice, and documented here as a choice, not a measured
fact. `build_adjacency` accepts an explicit edge list if a real topology
is ever supplied.
"""
from typing import List, Optional, Tuple

import numpy as np


def build_adjacency(
    n_nodes: int, edges: Optional[List[Tuple[int, int]]] = None, self_loops: bool = True,
) -> np.ndarray:
    """Returns an [n_nodes, n_nodes] binary adjacency matrix.

    edges=None -> fully-connected (default, see module docstring).
    edges=[(i,j), ...] -> undirected edges only at the given (i,j) pairs,
    for when a real topology is supplied later.
    """
    if edges is None:
        adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
    else:
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    else:
        np.fill_diagonal(adj, 0.0)
    return adj
