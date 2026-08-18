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

M6 (docs/PAPER5_M6_topology.md): to test whether GAT-CTDE's edge over
single-agent DQN depends on topology sparsity, not just node count, two
sparser generators are added below, both producing an explicit edge list
consumed by the same `build_adjacency` -- no change to that function or to
anything M2/M3/M4 already depend on:
  - `ring_edges`: each node connects to its two neighbours in an N-cycle.
    Defined for any N. Degenerate at N=3 (a 3-cycle already touches every
    pair, so ring == fully-connected there) -- documented, not a bug;
    ring only produces a sparser graph than fully-connected once N>=4.
  - `hex_grid_edges`: concentric hexagonal rings around one centre cell
    (ring 1 = 6 cells -> N=7 total; ring 2 = +12 cells -> N=19 total),
    the standard cellular frequency-reuse cluster sizes -- not a
    coincidence that M6's own N in {7, 19} matches them exactly. Two
    cells are adjacent iff their axial hex coordinates differ by one of
    the 6 unit hex directions (a cell sharing a physical edge). Defined
    only for N=7 and N=19 (raises otherwise) -- no ad hoc guess at what
    "hex-grid" means for an unlisted N.
"""
from typing import List, Optional, Tuple

import numpy as np

# The 6 unit axial-coordinate steps between adjacent hex cells (pointy-top
# axial convention; any consistent convention gives an isomorphic graph).
_HEX_DIRECTIONS: List[Tuple[int, int]] = [
    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
]


def _hex_ring_coords(ring: int) -> List[Tuple[int, int]]:
    """Axial (q, r) coordinates of every cell exactly `ring` steps from the
    centre (0, 0), in walk order -- the standard hex-ring-traversal
    algorithm (start at ring steps along direction 4, then walk `ring`
    cells along each of the 6 directions in turn)."""
    if ring == 0:
        return [(0, 0)]
    q, r = _HEX_DIRECTIONS[4][0] * ring, _HEX_DIRECTIONS[4][1] * ring
    coords = []
    for direction in range(6):
        dq, dr = _HEX_DIRECTIONS[direction]
        for _ in range(ring):
            coords.append((q, r))
            q, r = q + dq, r + dr
    return coords


def hex_cluster_coords(n_nodes: int) -> List[Tuple[int, int]]:
    """Axial coordinates for the standard n_nodes-cell hex cluster (centre
    + concentric rings), n_nodes in {7, 19} only -- see module docstring."""
    if n_nodes == 7:
        rings = 1
    elif n_nodes == 19:
        rings = 2
    else:
        raise ValueError(
            f"hex_cluster_coords: no standard hex cluster has {n_nodes} cells "
            "(defined only for the 7-cell and 19-cell clusters, M6's own N values)"
        )
    coords: List[Tuple[int, int]] = []
    for ring in range(rings + 1):
        coords.extend(_hex_ring_coords(ring))
    assert len(coords) == n_nodes, f"internal error: got {len(coords)} cells, expected {n_nodes}"
    return coords


def ring_edges(n_nodes: int) -> List[Tuple[int, int]]:
    """Node i <-> node (i+1) mod n_nodes -- an N-cycle. Degenerate (==
    fully-connected) at N=3; see module docstring."""
    if n_nodes < 3:
        raise ValueError(f"ring_edges: need at least 3 nodes for a cycle, got {n_nodes}")
    return [(i, (i + 1) % n_nodes) for i in range(n_nodes)]


def hex_grid_edges(n_nodes: int) -> List[Tuple[int, int]]:
    """Edges between node indices i, j (indices into the same order
    `hex_cluster_coords` returns) wherever their axial coordinates are one
    hex step apart. n_nodes in {7, 19} only."""
    coords = hex_cluster_coords(n_nodes)
    index_of = {c: i for i, c in enumerate(coords)}
    edges: List[Tuple[int, int]] = []
    seen = set()
    for i, (q, r) in enumerate(coords):
        for dq, dr in _HEX_DIRECTIONS:
            neighbor = (q + dq, r + dr)
            j = index_of.get(neighbor)
            if j is not None and (j, i) not in seen:
                edges.append((i, j))
                seen.add((i, j))
    return edges


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
