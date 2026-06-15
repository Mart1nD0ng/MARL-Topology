"""Forward-only local MLP edge scorer model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tensorizers import ACTOR_EDGE_FEATURE_FIELDS, ActorEdgeTensorBatch


LOCAL_MLP_EDGE_SCORER_MODEL_ID = "local_mlp_edge_scorer_v2_model_package"
LOCAL_MLP_EDGE_SCORE_SOURCE = "stage9_forward_local_mlp_edge_scorer"


@dataclass(frozen=True, slots=True)
class LocalMLPEdgeScorerConfig:
    model_id: str = LOCAL_MLP_EDGE_SCORER_MODEL_ID
    input_dim: int = len(ACTOR_EDGE_FEATURE_FIELDS)
    hidden_dim: int = 32
    output_dim: int = 1
    freeze_parameters: bool = False
    include_probability: bool = True

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be declared")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim != 1:
            raise ValueError("output_dim must be 1")


class LocalMLPEdgeScorer(nn.Module):
    """Shared directed-edge MLP over actor-safe local edge tensors."""

    def __init__(self, config: LocalMLPEdgeScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalMLPEdgeScorerConfig()
        self.network = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )
        if self.config.freeze_parameters:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.input_dim:
            raise ValueError("edge_features must have shape [edge_count, input_dim]")
        return self.network(edge_features).reshape(-1)

    def score_tensor_batch(self, batch: ActorEdgeTensorBatch) -> torch.Tensor:
        return self.forward(batch.edge_features)

    def boundary_report(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "feature_fields": list(ACTOR_EDGE_FEATURE_FIELDS)
            if self.config.input_dim == len(ACTOR_EDGE_FEATURE_FIELDS)
            else f"custom_actor_feature_dim_{self.config.input_dim}",
            "actor_input_boundary": "actor_safe_local_edges_only",
            "output_schema": "actor_policy_local_edge_score_output_v1",
            "outputs_final_topology": False,
            "outputs_activate": False,
            "training_only_fields_consumed": False,
        }
