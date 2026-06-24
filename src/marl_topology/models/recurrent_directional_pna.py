"""Phase 11b (Technical-Spec S7.12 items 2 & 4): directional message passing + recurrent shared update.

The decentralized PNA actor backbone: K rounds of finite neighbour communication.

  - DIRECTIONAL message passing: each DIRECTED edge ``u -> v`` carries a message ``msg([H[u], e_{u->v}])``
    built from the SOURCE state and the DIRECTION-SPECIFIC edge features (Spec S7.13: direction-specific
    delivery / SINR / ...). Messages are aggregated at the DESTINATION, so ``u->v`` and ``v->u`` (separate
    rows with separate features) contribute differently -- the layer is genuinely directional.
  - PNA aggregation: the incoming directed messages at each node are reduced by the 4 PNA aggregators
    (mean/max/min/std) and the 3 degree scalers (Phase 11a ``pna_combine``), using the node's IN-degree.
  - RECURRENT SHARED update: a SINGLE ``GRUCell`` (shared across ALL nodes and ALL K rounds, parameter-
    sharing per Spec) updates each node state from its PNA readout. Finite K rounds (K-hop receptive
    field).

Permutation-equivariant in the node ordering, device/dtype-preserving, variable N/E, no node IDs, no
global state (Spec S7.13) -- a fully DECENTRALIZED backbone (each node uses only its own state + the
messages of its physical in-neighbours). Lives in models/ (the actor; torch allowed -- only the decode
path is torch-free).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from marl_topology.models.pna_aggregation import PNA_SCALER_ALPHAS, pna_combine


def scatter_directional_pna(messages: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    """Aggregate directed-edge ``messages`` ``[E, F]`` at their destination ``dst`` ``[E]`` with the 4 PNA
    aggregators -> ``[N, 4, F]`` (mean, max, min, std of the INCOMING messages per node). Isolated nodes
    (no incoming edge) -> all-zero rows (matches :func:`pna_aggregators` at k=0). Device-preserving,
    vectorized via ``scatter_reduce`` (no node loop)."""
    if messages.dim() != 2:
        raise ValueError("messages must be [E, F]")
    f = messages.shape[-1]
    idx = dst.long().unsqueeze(-1).expand(-1, f)
    z = messages.new_zeros(num_nodes, f)
    if messages.shape[0] == 0:
        return torch.stack([z, z, z, z], dim=1)
    mean = z.clone().scatter_reduce(0, idx, messages, reduce="mean", include_self=False)
    mx = z.clone().scatter_reduce(0, idx, messages, reduce="amax", include_self=False)
    mn = z.clone().scatter_reduce(0, idx, messages, reduce="amin", include_self=False)
    msq = z.clone().scatter_reduce(0, idx, messages * messages, reduce="mean", include_self=False)
    std = (msq - mean * mean).clamp_min(0.0).sqrt()
    # scatter_reduce leaves untouched (isolated) destinations at the init 0 for mean/msq; amax/amin leave
    # them at 0 too (include_self=False on an empty segment keeps the init) -> isolated rows are zeros.
    return torch.stack([mean, mx, mn, std], dim=1)


class RecurrentDirectionalPNA(nn.Module):
    """K rounds of directional message passing + PNA aggregation + a SHARED recurrent (GRUCell) update.

    forward(node_features [N, node_dim], edge_index [E, 2] (src, dst), edge_features [E, edge_dim]) ->
    node embeddings [N, hidden]. ``edge_index`` is DIRECTED (pass both u->v and v->u as separate rows for
    an undirected physical graph, each with its own direction-specific features). ``delta`` is the PNA
    training log-degree mean (Phase 11a)."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, rounds: int = 3,
                 delta: float = 1.0, alphas=PNA_SCALER_ALPHAS):
        super().__init__()
        if rounds < 0:
            raise ValueError("rounds must be >= 0")
        self.hidden = hidden
        self.rounds = rounds
        self.delta = float(delta)
        self.alphas = tuple(alphas)
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        # directed-edge message from [source state, direction-specific edge features]
        self.msg = nn.Sequential(nn.Linear(hidden + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        # PNA readout dim = 4 aggregators * len(scalers) * hidden -> project to the GRU input
        pna_dim = 4 * len(self.alphas) * hidden
        self.pna_proj = nn.Linear(pna_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)        # SHARED across all nodes and all rounds

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        n = node_features.shape[0]
        h = self.encoder(node_features)              # [N, hidden]
        if edge_index.shape[0] == 0:                 # no edges -> in-degree 0 everywhere
            src = dst = node_features.new_zeros(0, dtype=torch.long)
            in_deg = node_features.new_zeros(n)
        else:
            src, dst = edge_index[:, 0].long(), edge_index[:, 1].long()
            in_deg = node_features.new_zeros(n).scatter_add(
                0, dst, node_features.new_ones(dst.shape[0]))
        for _ in range(self.rounds):                 # finite K rounds, SHARED params
            if edge_index.shape[0] == 0:
                pna = h.new_zeros(n, 4 * len(self.alphas) * self.hidden)
            else:
                m = self.msg(torch.cat([h[src], edge_features], dim=-1))   # directed message per edge
                agg = scatter_directional_pna(m, dst, n)                   # [N, 4, hidden]
                pna = pna_combine(agg, in_deg, self.delta, self.alphas)    # [N, 4*S*hidden]
            h = self.gru(self.pna_proj(pna), h)      # shared recurrent update
        return h
