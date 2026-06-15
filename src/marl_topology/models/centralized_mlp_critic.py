"""Training-only centralized MLP critic baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tensorizers import CRITIC_GLOBAL_FEATURE_FIELDS, CriticTensorBatch


CENTRALIZED_MLP_CRITIC_BASELINE_ID = "centralized_mlp_critic_baseline_v1"
CRITIC_HEAD_NAMES = (
    "value",
    "feasibility",
    "consensus_success_probability",
    "latency",
    "energy",
    "edge_delta_add",
    "edge_delta_remove",
    "edge_delta_keep",
)


@dataclass(frozen=True, slots=True)
class CentralizedMLPCriticConfig:
    model_id: str = CENTRALIZED_MLP_CRITIC_BASELINE_ID
    input_dim: int = len(CRITIC_GLOBAL_FEATURE_FIELDS)
    hidden_dim: int = 64
    edge_output_dim: int = 1
    training_only: bool = True

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be declared")
        if self.input_dim != len(CRITIC_GLOBAL_FEATURE_FIELDS):
            raise ValueError("input_dim must match critic feature schema")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.edge_output_dim <= 0:
            raise ValueError("edge_output_dim must be positive")
        if not self.training_only:
            raise ValueError("centralized critic baseline must remain training-only")


@dataclass(frozen=True, slots=True)
class CentralizedCriticTensorOutput:
    value: torch.Tensor
    feasibility: torch.Tensor
    consensus_success_probability: torch.Tensor
    latency: torch.Tensor
    energy: torch.Tensor
    edge_delta_add: torch.Tensor
    edge_delta_remove: torch.Tensor
    edge_delta_keep: torch.Tensor
    training_only: bool = True

    @property
    def head_names(self) -> tuple[str, ...]:
        return CRITIC_HEAD_NAMES

    def assert_shapes(self, *, batch_size: int, edge_count: int) -> None:
        scalar_shape = (batch_size,)
        edge_shape = (batch_size, edge_count)
        for name in (
            "value",
            "feasibility",
            "consensus_success_probability",
            "latency",
            "energy",
        ):
            if getattr(self, name).shape != scalar_shape:
                raise ValueError(f"{name} head shape mismatch")
        for name in ("edge_delta_add", "edge_delta_remove", "edge_delta_keep"):
            if getattr(self, name).shape != edge_shape:
                raise ValueError(f"{name} head shape mismatch")


class CentralizedMLPCriticBaseline(nn.Module):
    """Centralized training-only critic with scalar and edge-delta heads."""

    def __init__(self, config: CentralizedMLPCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(config.hidden_dim, 1)
        self.feasibility_head = nn.Linear(config.hidden_dim, 1)
        self.consensus_head = nn.Linear(config.hidden_dim, 1)
        self.latency_head = nn.Linear(config.hidden_dim, 1)
        self.energy_head = nn.Linear(config.hidden_dim, 1)
        self.edge_delta_add_head = nn.Linear(config.hidden_dim, config.edge_output_dim)
        self.edge_delta_remove_head = nn.Linear(config.hidden_dim, config.edge_output_dim)
        self.edge_delta_keep_head = nn.Linear(config.hidden_dim, config.edge_output_dim)

    def forward(self, global_features: torch.Tensor) -> CentralizedCriticTensorOutput:
        if global_features.ndim != 2 or global_features.shape[1] != self.config.input_dim:
            raise ValueError("global_features must have shape [batch, input_dim]")
        hidden = self.trunk(global_features)
        return CentralizedCriticTensorOutput(
            value=self.value_head(hidden).reshape(-1),
            feasibility=torch.sigmoid(self.feasibility_head(hidden).reshape(-1)),
            consensus_success_probability=torch.sigmoid(
                self.consensus_head(hidden).reshape(-1)
            ),
            latency=torch.relu(self.latency_head(hidden).reshape(-1)),
            energy=torch.relu(self.energy_head(hidden).reshape(-1)),
            edge_delta_add=self.edge_delta_add_head(hidden),
            edge_delta_remove=self.edge_delta_remove_head(hidden),
            edge_delta_keep=self.edge_delta_keep_head(hidden),
        )

    def predict_tensor_batch(self, batch: CriticTensorBatch) -> CentralizedCriticTensorOutput:
        output = self.forward(batch.global_features)
        output.assert_shapes(batch_size=batch.batch_size, edge_count=batch.edge_count)
        return output

    def boundary_report(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "feature_fields": list(CRITIC_GLOBAL_FEATURE_FIELDS),
            "head_names": list(CRITIC_HEAD_NAMES),
            "training_only": True,
            "deployment_actor_receives_critic_output": False,
        }
