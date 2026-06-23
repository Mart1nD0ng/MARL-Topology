"""Centralized graph value critic for the single-step CTDE trunk (Engineering-Plan Phase 7).

A TRAINING-ONLY centralized critic: it sees the WHOLE candidate graph (all nodes + edges) -- legal
in CTDE training -- and predicts a scalar ``V(scene)`` in reward units, used as the PPO advantage
baseline. It has its OWN graph encoder with INDEPENDENT weights (never shared with the actor
``MessagePassingGraphEdgeScorer``), so the critic gradient never touches the actor and the two are
trained by separate optimizers. The critic is NEVER reachable from the deployed actor (D1): it is
constructed and called only inside the trunk's ``graph-mappo`` branch; no deployed module (under
``policies/``, ``protocol/``, ``data/``, ``evaluation/``) imports it. Lives under ``models/``
(torch-exempt under the deployment-purity gates).

Architecture: the same proven K-round bidirectional message-passing primitive as the actor (its
own independent weights), then a permutation-invariant masked mean + max readout over nodes into a
scalar value head (vector heads V_E / V_L / V_C are a dormant, flag-gated extension deferred to a
later increment -- the seam is shaped so they are additive).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


CENTRALIZED_GRAPH_CRITIC_MODEL_ID = "centralized_graph_value_critic_v1"


def _scatter_add(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # local copy of the message-passing primitive so the critic carries no reference to the actor
    out = torch.zeros((values.shape[0], num_nodes, values.shape[-1]), dtype=values.dtype)
    out.scatter_add_(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]), values)
    return out


@dataclass(frozen=True, slots=True)
class CentralizedGraphCriticConfig:
    node_input_dim: int = 8
    edge_input_dim: int = 8
    hidden: int = 64
    rounds: int = 4
    vector_heads: bool = False
    critic_sees_action: bool = False


class CentralizedGraphCritic(nn.Module):
    """Centralized graph value critic V(scene) (training-only; separate from the actor)."""

    model_id = CENTRALIZED_GRAPH_CRITIC_MODEL_ID

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int = 64,
        rounds: int = 4,
        vector_heads: bool = False,
        critic_sees_action: bool = False,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.rounds = rounds
        self.vector_heads = vector_heads
        self.critic_sees_action = critic_sees_action
        edge_in = edge_dim + (1 if critic_sees_action else 0)
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
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
        out_dim = 3 if vector_heads else 1
        # readout: masked mean + max pooling over BOTH nodes and edges -> [B, 4*hidden] -> value head
        # (pooling edges too keeps the final edge state in the value path -- a topology value depends
        # on the edges -- and leaves no dead message-passing weights).
        self.value_head = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    @classmethod
    def from_config(cls, config: CentralizedGraphCriticConfig) -> "CentralizedGraphCritic":
        return cls(config.node_input_dim, config.edge_input_dim, config.hidden, config.rounds,
                   config.vector_heads, config.critic_sees_action)

    def forward(self, node_features, edge_features, edge_index, node_mask, edge_mask, active_edge_onehot=None):
        # shapes: node [B,N,Dn], edge [B,E,De], edge_index [B,E,2], masks [B,N]/[B,E]
        B, N, _ = node_features.shape
        if self.critic_sees_action:
            if active_edge_onehot is None:
                active_edge_onehot = torch.zeros(edge_features.shape[:-1], dtype=edge_features.dtype)
            # the realized joint action is a DETACHED conditioning input (toward the Phase-8 V->Q step)
            edge_features = torch.cat((edge_features, active_edge_onehot.detach().unsqueeze(-1)), dim=-1)
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
            node_state = self.node_norm[r](node_state + upd) * node_mask.unsqueeze(-1)
            edge_state = self.edge_norm[r](edge_state + m) * edge_mask.unsqueeze(-1)
        # permutation-invariant masked mean + max pooling over BOTH nodes and edges (centralized readout)
        floor = torch.finfo(node_state.dtype).min
        nmask = node_mask.unsqueeze(-1)
        ncount = node_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        node_mean = (node_state * nmask).sum(dim=1) / ncount
        node_max = (node_state + (1.0 - nmask) * floor).max(dim=1).values
        emask = edge_mask.unsqueeze(-1)
        ecount = edge_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        edge_mean = (edge_state * emask).sum(dim=1) / ecount
        edge_max = (edge_state + (1.0 - emask) * floor).max(dim=1).values
        value = self.value_head(torch.cat((node_mean, node_max, edge_mean, edge_max), dim=-1))  # [B, out_dim]
        return value if self.vector_heads else value.squeeze(-1)

    def boundary_report(self) -> dict[str, object]:
        """Training-only boundary: centralized graph value, never a deployment module (D1)."""

        return {
            "model_id": self.model_id,
            "role": "centralized_value_critic_training_only",
            "uses_global_graph": True,
            "reachable_from_deployed_actor": False,
            "outputs": "vector_value" if self.vector_heads else "scalar_value",
            "shares_weights_with_actor": False,
        }
