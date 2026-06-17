"""K-round bidirectional message-passing GNN edge scorer (the production actor).

This is the actor that produced the project's best validated result (held-out decentralized
feasibility 0.82, N=8 0.957, under the full TR 37.885 stochastic stack at the 4-RSU / 20 dBm
operating point; see ``docs/URBAN_V2X_RESEARCH_LOG.md`` 1'/2'/Step-3). It is a genuine graph
neural network: ``K`` rounds of bidirectional neighbour message passing over the candidate
graph, emitting a per-edge activation logit.

Decentralized-with-communication (NOT a global-state shortcut)
--------------------------------------------------------------
Each round is one hop of local neighbour-to-neighbour signalling (realistic V2V / V2I message
exchange): every node sends its state to its neighbours, messages are aggregated to BOTH
endpoints (``_scatter_add(m, dst) + _scatter_add(m, src)``), then node and edge states are
updated with a residual + LayerNorm. After ``K`` rounds a node's receptive field is ``K`` hops,
so the policy can reason about the global backbone while every message stays local. This is the
standard GNN-as-communication view of a Dec-POMDP actor; the actor never reads global adjacency
as a feature. The hard topology activation is owned by the decentralized mutual-acceptance
decoder (``policies/decentralized_mutual_acceptance.py``); this module outputs edge scores only.

The class attribute names are kept byte-stable so the frozen ``_artifacts_step3.pt`` /
``_artifacts_phase2p.pt`` state dicts load directly into it (recovered, not re-derived).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID = "kround_message_passing_graph_edge_scorer_v1"


def _scatter_add(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros((values.shape[0], num_nodes, values.shape[-1]), dtype=values.dtype)
    out.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]), values)
    return out


@dataclass(frozen=True, slots=True)
class MessagePassingGraphEdgeScorerConfig:
    node_input_dim: int = 8
    edge_input_dim: int = 8
    hidden: int = 64
    rounds: int = 4
    dropout: float = 0.0


class MessagePassingGraphEdgeScorer(nn.Module):
    """K-round bidirectional message-passing edge scorer (decentralized-with-communication)."""

    model_id = MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, rounds: int = 4, dropout: float = 0.0):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.rounds = rounds
        self.drop = nn.Dropout(dropout)
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.msg = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(rounds)
        )
        self.node_update = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(rounds)
        )
        self.node_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(rounds))
        self.edge_norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(rounds))
        self.edge_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    @classmethod
    def from_config(cls, config: MessagePassingGraphEdgeScorerConfig) -> "MessagePassingGraphEdgeScorer":
        return cls(config.node_input_dim, config.edge_input_dim, config.hidden, config.rounds, config.dropout)

    def forward(self, node_features, edge_features, edge_index, node_mask, edge_mask):
        # shapes: node [B,N,Dn], edge [B,E,De], edge_index [B,E,2], masks [B,N]/[B,E]
        B, N, _ = node_features.shape
        node_state = self.node_encoder(node_features) * node_mask.unsqueeze(-1)
        edge_state = self.edge_encoder(edge_features) * edge_mask.unsqueeze(-1)
        bidx = torch.arange(B).unsqueeze(1)
        src = edge_index[..., 0].clamp(0, N - 1)
        dst = edge_index[..., 1].clamp(0, N - 1)
        for r in range(self.rounds):
            s_state = node_state[bidx, src]
            d_state = node_state[bidx, dst]
            m = self.msg[r](torch.cat((s_state, d_state, edge_state), dim=-1)) * edge_mask.unsqueeze(-1)
            agg = _scatter_add(m, dst, N) + _scatter_add(m, src, N)  # bidirectional
            upd = self.node_update[r](torch.cat((node_state, agg), dim=-1))
            node_state = self.node_norm[r](node_state + self.drop(upd)) * node_mask.unsqueeze(-1)
            edge_state = self.edge_norm[r](edge_state + self.drop(m)) * edge_mask.unsqueeze(-1)
        s_state = node_state[bidx, src]
        d_state = node_state[bidx, dst]
        logit = self.edge_head(torch.cat((edge_state, s_state * d_state, torch.abs(s_state - d_state)), dim=-1))
        return logit.squeeze(-1)  # [B, E]

    def boundary_report(self) -> dict[str, object]:
        """Deployment boundary: K-hop local message passing, no global-state feature."""

        return {
            "model_id": self.model_id,
            "global_topology_used": False,
            "receptive_field_hops": self.rounds,
            "decentralized_with_communication": True,
            "outputs": "per_edge_activation_logit",
            "activation_owned_by_decoder": True,
        }
