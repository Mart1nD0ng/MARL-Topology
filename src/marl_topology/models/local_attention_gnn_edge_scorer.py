"""Local attention (GAT-style) message-passing GNN edge scorer.

A drop-in actor that replaces v3's symmetric MEAN ego-message aggregation with a learned
attention: each ego node weights its incident edges by a per-layer segment-softmax instead
of averaging them, so it can focus on the most informative links (e.g. the few reliable
edges in a dense ego-graph) rather than diluting them. Everything else (the v3 encoders,
residual+norm message passing, the score head, the actor-safe edge-score-only boundary) is
unchanged. Outputs one logit per directed edge, exactly like v3.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .local_gnn_edge_scorer import (
    LocalGNNV3ResidualNormConfig,
    LocalMessagePassingGNNV3ResidualNorm,
)

LOCAL_ATTENTION_GNN_EDGE_SCORER_MODEL_ID = "local_attention_gnn_edge_scorer_v1"


def _attention_by_group(
    values: torch.Tensor,
    attn_logits: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """Segment-softmax attention aggregation. For each ego group, softmax the per-edge
    attention logits over the group's incident edges and return the attention-weighted
    message broadcast back to each edge -- shape [E, H], matching ``_mean_by_group``.

    The softmax uses a global max shift, which is exactly segment-softmax-invariant (a
    constant shift cancels in each group's normalization) and prevents overflow.
    """
    unique_groups, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    num_groups = unique_groups.shape[0]
    weights = torch.exp(attn_logits - attn_logits.max())  # [E, 1], stable
    denom = torch.zeros((num_groups, 1), dtype=values.dtype, device=values.device)
    denom.index_add_(0, inverse, weights)
    weights = weights / denom[inverse].clamp_min(1e-12)  # per-group softmax weights
    weighted = values * weights  # [E, H]
    out = torch.zeros((num_groups, values.shape[1]), dtype=values.dtype, device=values.device)
    out.index_add_(0, inverse, weighted)
    return out[inverse]


@dataclass(frozen=True, slots=True)
class LocalAttentionGNNEdgeScorerConfig(LocalGNNV3ResidualNormConfig):
    model_id: str = LOCAL_ATTENTION_GNN_EDGE_SCORER_MODEL_ID


class LocalAttentionGNNEdgeScorer(LocalMessagePassingGNNV3ResidualNorm):
    """v3 local message-passing GNN with GAT-style attention ego-message aggregation."""

    def __init__(self, config: LocalAttentionGNNEdgeScorerConfig | None = None) -> None:
        super().__init__(config or LocalAttentionGNNEdgeScorerConfig())
        hidden = self.config.hidden_dim
        layers = self.config.message_passing_layers
        # one attention scorer per message-passing layer: edge message -> attention logit.
        self.attention_scorers = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(layers))
        for scorer in self.attention_scorers:
            nn.init.xavier_uniform_(scorer.weight)
            nn.init.zeros_(scorer.bias)

    def _aggregate_message(
        self,
        edge_message: torch.Tensor,
        group_ids: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        attn_logits = self.attention_scorers[layer_index](edge_message)  # [E, 1]
        return _attention_by_group(edge_message, attn_logits, group_ids)

    def boundary_report(self) -> dict[str, object]:
        report = super().boundary_report()
        report.update(
            {
                "model_id": self.config.model_id,
                "ego_message_aggregation": "gat_segment_softmax_attention",
                "attention_layers": len(self.attention_scorers),
            }
        )
        return report
