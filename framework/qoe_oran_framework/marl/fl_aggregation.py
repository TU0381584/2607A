"""Server-side FedAvg aggregation (McMahan et al. 2017, cited in
Papers_4-5/Paper_4/refs.bib as `fedavg`): a plain (optionally weighted) average of
client state_dicts. FedAvg and FedProx (Li et al. 2020) share this exact
server-side step -- the only difference between them is in the CLIENT's
local objective (FedProx adds a proximal term pulling local weights toward
the last-broadcast global model during local training; see
FederatedGatPolicy._local_loss in fl_ctde_policy.py), not in how the
server combines the resulting weights. So there is deliberately only one
aggregation function here, reused by both aggregator modes.
"""
from typing import Dict, List, Optional, Sequence

import torch


def fedavg_aggregate(
    state_dicts: Sequence[Dict[str, torch.Tensor]], weights: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Returns a new state_dict: the (optionally weighted) mean of every
    tensor across `state_dicts`, keyed identically (all clients share the
    same architecture, so key sets match exactly). weights=None -> equal
    weighting (McMahan et al. 2017's convention when clients contribute
    equal-sized local datasets, the case here: every gNB runs the same
    number of local episodes per round)."""
    if not state_dicts:
        raise ValueError("fedavg_aggregate: no state_dicts to aggregate")
    n = len(state_dicts)
    if weights is None:
        w = [1.0 / n] * n
    else:
        total = float(sum(weights))
        w = [x / total for x in weights]

    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for sd, wi in zip(state_dicts, w):
            acc += wi * sd[key].to(torch.float32)
        out[key] = acc.to(state_dicts[0][key].dtype)
    return out
