"""Enriched centralized value critic candidates for Stage 27."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from marl_topology.training.critic_features import VALUE_CRITIC_FEATURE_FIELDS


ENRICHED_CENTRALIZED_MLP_CRITIC_ID = "enriched_centralized_mlp_value_critic_v1"
ENRICHED_CRITIC_OUTPUT_SCHEMA_ID = "stage27_normalized_value_with_aux_heads_v1"


@dataclass(frozen=True, slots=True)
class EnrichedCentralizedMLPCriticConfig:
    model_id: str = ENRICHED_CENTRALIZED_MLP_CRITIC_ID
    input_dim: int = len(VALUE_CRITIC_FEATURE_FIELDS)
    hidden_dim: int = 128
    training_only: bool = True

    def __post_init__(self) -> None:
        if self.input_dim != len(VALUE_CRITIC_FEATURE_FIELDS):
            raise ValueError("input_dim must match Stage 27 value features")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not self.training_only:
            raise ValueError("critic must remain training-only")


@dataclass(frozen=True, slots=True)
class EnrichedCriticOutput:
    normalized_value: torch.Tensor
    feasibility_logit: torch.Tensor
    consensus_proxy: torch.Tensor
    latency_proxy: torch.Tensor
    energy_proxy: torch.Tensor
    training_only: bool = True

    def assert_shapes(self, batch_size: int) -> None:
        for name in (
            "normalized_value",
            "feasibility_logit",
            "consensus_proxy",
            "latency_proxy",
            "energy_proxy",
        ):
            if getattr(self, name).shape != (batch_size,):
                raise ValueError(f"{name} shape mismatch")


class EnrichedCentralizedMLPCritic(nn.Module):
    """Training-only MLP that predicts normalized centralized value targets."""

    def __init__(self, config: EnrichedCentralizedMLPCriticConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnrichedCentralizedMLPCriticConfig()
        self.trunk = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
        )
        head_dim = self.config.hidden_dim // 2
        self.value_head = nn.Linear(head_dim, 1)
        self.feasibility_head = nn.Linear(head_dim, 1)
        self.consensus_head = nn.Linear(head_dim, 1)
        self.latency_head = nn.Linear(head_dim, 1)
        self.energy_head = nn.Linear(head_dim, 1)

    def forward(self, value_features: torch.Tensor) -> EnrichedCriticOutput:
        if value_features.ndim != 2 or value_features.shape[1] != self.config.input_dim:
            raise ValueError("value_features must have shape [batch, input_dim]")
        hidden = self.trunk(value_features)
        output = EnrichedCriticOutput(
            normalized_value=self.value_head(hidden).reshape(-1),
            feasibility_logit=self.feasibility_head(hidden).reshape(-1),
            consensus_proxy=self.consensus_head(hidden).reshape(-1),
            latency_proxy=self.latency_head(hidden).reshape(-1),
            energy_proxy=self.energy_head(hidden).reshape(-1),
        )
        output.assert_shapes(int(value_features.shape[0]))
        return output

    def boundary_report(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "input_schema_id": "stage27_pre_action_value_critic_features_v1",
            "output_schema_id": ENRICHED_CRITIC_OUTPUT_SCHEMA_ID,
            "training_only": True,
            "deployment_actor_receives_critic_output": False,
            "predicts_normalized_value": True,
            "feature_count": len(VALUE_CRITIC_FEATURE_FIELDS),
        }
