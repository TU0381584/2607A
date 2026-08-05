"""Feature-extraction glue between the existing, UNMODIFIED
qoe_oran_framework.env.RANEnv (already multi-gNB capable -- see its
encode_state()/ClusterState.per_gnb, exercised today by paper #2's
LB-extension configs) and the new GAT/CTDE policy layer.

No new environment class is needed: RANEnv.step(actions) already accepts
one action per env.pending_requests() entry regardless of which gNB a
request belongs to, and ClusterState.per_gnb already carries a clean
per-node feature set. This module only adds the two small, pure functions
that turn RANEnv's existing observation objects into the graph-node
tensor and per-request (agent_idx, context) pairs the CTDE policy needs --
nothing here touches env.py.
"""
from typing import List, Tuple

import numpy as np

from ..config import SacLbExperimentConfig
from ..types import AdmissionRequest, ClusterState

NODE_FEAT_DIM_PER_SLICE = 3  # [prb_used_ratio, congestion_level, queue_len_norm], matches env.encode_state


def node_feature_dim(cfg: SacLbExperimentConfig) -> int:
    return len(cfg.slices) * NODE_FEAT_DIM_PER_SLICE


def extract_node_features(cluster_state: ClusterState, cfg: SacLbExperimentConfig) -> np.ndarray:
    """Returns [n_gnb, node_feature_dim]: row i is gNB cfg.gnb_ids[i]'s own
    per-slice [prb_used_ratio, congestion_level, queue_len_norm], in
    cfg.slices order -- the exact same per-(gNB,slice) triple
    env.encode_state() concatenates into one flat vector, just kept as a
    per-node matrix here instead of flattened, since the GAT encoder needs
    node structure, not a flat vector."""
    n_gnb = len(cfg.gnb_ids)
    dim = node_feature_dim(cfg)
    out = np.zeros((n_gnb, dim), dtype=np.float32)
    for i, gnb_id in enumerate(cfg.gnb_ids):
        slice_states = cluster_state.per_gnb.get(gnb_id, {})
        offset = 0
        for spec in cfg.slices:
            agg = slice_states.get(spec.slice_id)
            if agg is not None:
                out[i, offset:offset + 3] = [agg.prb_used_ratio, agg.congestion_level, agg.queue_len_norm]
            offset += 3
    return out


def request_context_dim(cfg: SacLbExperimentConfig) -> int:
    return len(cfg.slices)


def request_to_agent_context(request: AdmissionRequest, cfg: SacLbExperimentConfig) -> Tuple[int, np.ndarray]:
    """Returns (agent_idx, slice_onehot). agent_idx replaces the gNB
    one-hot RANEnv's single-agent encode_request_context() used -- here,
    gNB identity IS which agent acts, not a feature fed to one shared
    policy, so only the slice one-hot remains as context."""
    agent_idx = cfg.gnb_ids.index(request.gnb_id)
    onehot = np.array([1.0 if request.slice_id == s.slice_id else 0.0 for s in cfg.slices], dtype=np.float32)
    return agent_idx, onehot


def requests_to_agent_contexts(
    pending: List[AdmissionRequest], cfg: SacLbExperimentConfig
) -> List[Tuple[int, np.ndarray]]:
    return [request_to_agent_context(r, cfg) for r in pending]
