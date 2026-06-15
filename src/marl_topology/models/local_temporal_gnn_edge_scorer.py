"""Local temporal message-passing GNN actor (workstream 3, Part B).

A drop-in actor that adds a recurrent temporal stage in front of the v3 local
message-passing GNN: it encodes each candidate edge's W-step history (the leakage-safe
``[E, W, F]`` window from A4) with a gated recurrent encoder, fuses the temporal summary
(as a ZERO-initialised residual) into the current-frame edge features, then runs the
UNCHANGED v3 message passing -> one logit per directed edge. So the Plackett-Luce
sampler, the deployment assembler, and the centralized critic are untouched.

When a batch carries no history (the static / single-frame path) the actor degrades
EXACTLY to the plain v3 GNN -- the recurrent + fusion stages are bypassed. And because
the fusion is zero-initialised as a residual, even WITH history the actor starts
byte-identical to v3 and only earns temporal influence through training gradients,
preserving the v3 unsaturated-init fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from marl_topology.models.local_gnn_edge_scorer import (
    LocalGNNV3ResidualNormConfig,
    LocalMessagePassingGNNV3ResidualNorm,
)
from marl_topology.models.tensorizers import ActorEdgeTensorBatch

LOCAL_TEMPORAL_GNN_EDGE_SCORER_MODEL_ID = "local_temporal_gnn_edge_scorer_v1"


@dataclass(frozen=True, slots=True)
class LocalTemporalGNNEdgeScorerConfig(LocalGNNV3ResidualNormConfig):
    model_id: str = LOCAL_TEMPORAL_GNN_EDGE_SCORER_MODEL_ID
    temporal_dim: int = 24

    def __post_init__(self) -> None:
        LocalGNNV3ResidualNormConfig.__post_init__(self)
        if self.temporal_dim < 1:
            raise ValueError("temporal_dim must be >= 1")


class LocalTemporalGNNEdgeScorer(LocalMessagePassingGNNV3ResidualNorm):
    """v3 local message-passing GNN with a recurrent temporal pre-encoder over edge history."""

    def __init__(self, config: LocalTemporalGNNEdgeScorerConfig | None = None) -> None:
        super().__init__(config or LocalTemporalGNNEdgeScorerConfig())
        input_dim = self.config.input_dim
        temporal_dim = int(getattr(self.config, "temporal_dim", 24))
        # Recurrent encoder over the [E, W, F] history (batch=E, seq=W, feature=F).
        self.temporal_encoder = nn.GRU(
            input_size=input_dim, hidden_size=temporal_dim, batch_first=True
        )
        # Residual fusion of the temporal summary into the current edge features,
        # ZERO-initialised so at init the temporal contribution is 0 -> byte-identical
        # to the plain v3 GNN; training gradients then let the temporal path earn
        # influence (mirrors the v3 unsaturated-init philosophy).
        self.temporal_fusion = nn.Linear(input_dim + temporal_dim, input_dim)
        nn.init.zeros_(self.temporal_fusion.weight)
        nn.init.zeros_(self.temporal_fusion.bias)

    def score_tensor_batch(self, batch: ActorEdgeTensorBatch) -> torch.Tensor:
        history = getattr(batch, "history", None)
        if history is None:
            # static / single-frame path: byte-identical to the plain v3 GNN.
            return super().score_tensor_batch(batch)
        return self.forward_with_history(history, batch.edge_features, batch.group_ids)

    def forward_with_history(
        self,
        history: torch.Tensor,
        edge_features: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        edge_count = edge_features.shape[0]
        if edge_count == 0:
            return super().forward_with_groups(edge_features, group_ids)
        if (
            history.ndim != 3
            or history.shape[0] != edge_count
            or history.shape[2] != self.config.input_dim
        ):
            raise ValueError(
                "history must be [edge_count, window, input_dim] aligned to edge_features"
            )
        # Encode each edge's W-step history; take the last hidden state as the summary.
        _sequence_out, last_hidden = self.temporal_encoder(history)
        temporal_summary = last_hidden[-1]  # [E, temporal_dim]
        fused = edge_features + self.temporal_fusion(
            torch.cat((edge_features, temporal_summary), dim=1)
        )
        return super().forward_with_groups(fused, group_ids)

    def boundary_report(self) -> dict[str, object]:
        report = super().boundary_report()
        report.update(
            {
                "temporal_history_consumed": True,
                "temporal_encoder_kind": "gated_recurrent_edge_history_encoder",
                "temporal_dim": int(getattr(self.config, "temporal_dim", 24)),
                "degrades_to_static_gnn_without_history": True,
                "outputs_edge_scores_only": True,
                "full_message_passing_gnn": True,
            }
        )
        return report
