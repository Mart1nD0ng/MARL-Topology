"""Centralized message-passing graph value critic for Stage 27."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from marl_topology.models.quorum_tail_pool import QUORUM_TAIL_FEATURE_DIM, QuorumTailReadout


CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID = (
    "centralized_message_passing_graph_value_critic_v1"
)
POOLING_MEAN = "mean"
POOLING_QUORUM_TAIL = "quorum_tail"


@dataclass(frozen=True, slots=True)
class GraphCriticBatch:
    node_features: torch.Tensor
    edge_features: torch.Tensor
    edge_index: torch.Tensor
    node_mask: torch.Tensor
    edge_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.node_features.ndim != 3:
            raise ValueError("node_features must be [batch, nodes, node_dim]")
        if self.edge_features.ndim != 3:
            raise ValueError("edge_features must be [batch, edges, edge_dim]")
        if self.edge_index.shape[:2] != self.edge_features.shape[:2]:
            raise ValueError("edge_index must align with edge_features")
        if self.edge_index.shape[-1] != 2:
            raise ValueError("edge_index must store source and target ids")
        if self.node_mask.shape != self.node_features.shape[:2]:
            raise ValueError("node_mask shape mismatch")
        if self.edge_mask.shape != self.edge_features.shape[:2]:
            raise ValueError("edge_mask shape mismatch")
        for name in ("node_features", "edge_features"):
            if not torch.isfinite(getattr(self, name)).all().item():
                raise ValueError(f"{name} must be finite")

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])


@dataclass(frozen=True, slots=True)
class CentralizedMessagePassingGraphCriticConfig:
    model_id: str = CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID
    node_feature_dim: int = 8
    edge_feature_dim: int = 8
    hidden_dim: int = 64
    message_layers: int = 2
    training_only: bool = True
    # pooling selects the graph readout (opt-in, default byte-identical):
    # - "mean": permutation-invariant masked mean over nodes/edges (unchanged).
    # - "quorum_tail": the BFT-matched readout -- a per-node soft readiness gate pooled
    #   through the differentiable quorum-tail at k(N)=2f+1 (the PBFT fault-tolerant quorum),
    #   carrying the consensus order-statistic as an architectural prior instead of a mean.
    pooling: str = POOLING_MEAN
    # distributional (opt-in, default byte-identical): under a STOCHASTIC channel a topology's
    # consensus is a DISTRIBUTION over shadowing/NLOSv realizations, not a point. When on, two
    # extra heads predict the realization-distribution summary the deployment cares about:
    #   robust_feasibility_logit -- P(consensus >= tau ACROSS realizations) (trained on the
    #     M-draw robust rate), the cliff-edge-aware selection signal a single-draw feasibility
    #     head is structurally blind to; and
    #   consensus_low_quantile  -- the pessimistic (e.g. 10th-percentile) consensus, a margin.
    # The point heads (consensus/feasibility) are unchanged. Default off => byte-identical.
    distributional: bool = False

    def __post_init__(self) -> None:
        if self.node_feature_dim <= 0 or self.edge_feature_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.message_layers < 2:
            raise ValueError("graph critic requires at least two message layers")
        if not self.training_only:
            raise ValueError("graph critic must remain training-only")
        if self.pooling not in (POOLING_MEAN, POOLING_QUORUM_TAIL):
            raise ValueError("pooling must be 'mean' or 'quorum_tail'")


@dataclass(frozen=True, slots=True)
class GraphCriticOutput:
    normalized_value: torch.Tensor
    feasibility_logit: torch.Tensor
    consensus_proxy: torch.Tensor
    latency_proxy: torch.Tensor
    energy_proxy: torch.Tensor
    training_only: bool = True
    # populated only when the critic config is distributional (else None).
    robust_feasibility_logit: torch.Tensor | None = None
    consensus_low_quantile: torch.Tensor | None = None

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
        for name in ("robust_feasibility_logit", "consensus_low_quantile"):
            value = getattr(self, name)
            if value is not None and value.shape != (batch_size,):
                raise ValueError(f"{name} shape mismatch")


class CentralizedMessagePassingGraphCritic(nn.Module):
    """Training-only graph value critic with permutation-invariant pooling."""

    def __init__(
        self,
        config: CentralizedMessagePassingGraphCriticConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CentralizedMessagePassingGraphCriticConfig()
        hidden = self.config.hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(self.config.node_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.config.edge_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.message_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden * 3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(self.config.message_layers)
        )
        self.node_update_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(self.config.message_layers)
        )
        self.quorum_tail_readout = (
            QuorumTailReadout(hidden)
            if self.config.pooling == POOLING_QUORUM_TAIL
            else None
        )
        graph_head_in = hidden * 2 + (
            QUORUM_TAIL_FEATURE_DIM if self.config.pooling == POOLING_QUORUM_TAIL else 0
        )
        self.graph_head = nn.Sequential(
            nn.Linear(graph_head_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.feasibility_head = nn.Linear(hidden, 1)
        self.consensus_head = nn.Linear(hidden, 1)
        self.latency_head = nn.Linear(hidden, 1)
        self.energy_head = nn.Linear(hidden, 1)
        # distributional heads appended LAST so the default-path parameter-init order (and
        # thus the byte-identical default) is unchanged when distributional is off.
        if self.config.distributional:
            self.robust_feasibility_head = nn.Linear(hidden, 1)
            self.consensus_low_quantile_head = nn.Linear(hidden, 1)
        else:
            self.robust_feasibility_head = None
            self.consensus_low_quantile_head = None

    def forward(self, batch: GraphCriticBatch) -> GraphCriticOutput:
        node_state = self.node_encoder(batch.node_features)
        edge_state = self.edge_encoder(batch.edge_features)
        batch_size, node_count, hidden = node_state.shape
        edge_count = edge_state.shape[1]
        batch_ids = torch.arange(batch_size, device=node_state.device).unsqueeze(1)
        for message_layer, update_layer in zip(
            self.message_layers,
            self.node_update_layers,
            strict=True,
        ):
            source_index = batch.edge_index[..., 0].clamp(0, node_count - 1)
            target_index = batch.edge_index[..., 1].clamp(0, node_count - 1)
            source_state = node_state[batch_ids, source_index]
            target_state = node_state[batch_ids, target_index]
            message_input = torch.cat((source_state, target_state, edge_state), dim=-1)
            messages = message_layer(message_input) * batch.edge_mask.unsqueeze(-1)
            aggregated = torch.zeros(
                (batch_size, node_count, hidden),
                dtype=node_state.dtype,
                device=node_state.device,
            )
            expanded_target = target_index.unsqueeze(-1).expand(batch_size, edge_count, hidden)
            aggregated.scatter_add_(1, expanded_target, messages)
            reverse_target = source_index.unsqueeze(-1).expand(batch_size, edge_count, hidden)
            aggregated.scatter_add_(1, reverse_target, messages)
            node_state = update_layer(torch.cat((node_state, aggregated), dim=-1))
            node_state = node_state * batch.node_mask.unsqueeze(-1)
            edge_state = edge_state + messages
        edge_pool = _masked_mean(edge_state, batch.edge_mask)
        if self.quorum_tail_readout is not None:
            node_pool, tail_features = self.quorum_tail_readout(node_state, batch.node_mask)
            graph_state = self.graph_head(
                torch.cat((node_pool, edge_pool, tail_features), dim=-1)
            )
        else:
            node_pool = _masked_mean(node_state, batch.node_mask)
            graph_state = self.graph_head(torch.cat((node_pool, edge_pool), dim=-1))
        robust = (
            self.robust_feasibility_head(graph_state).reshape(-1)
            if self.robust_feasibility_head is not None
            else None
        )
        low_quantile = (
            self.consensus_low_quantile_head(graph_state).reshape(-1)
            if self.consensus_low_quantile_head is not None
            else None
        )
        output = GraphCriticOutput(
            normalized_value=self.value_head(graph_state).reshape(-1),
            feasibility_logit=self.feasibility_head(graph_state).reshape(-1),
            consensus_proxy=self.consensus_head(graph_state).reshape(-1),
            latency_proxy=self.latency_head(graph_state).reshape(-1),
            energy_proxy=self.energy_head(graph_state).reshape(-1),
            robust_feasibility_logit=robust,
            consensus_low_quantile=low_quantile,
        )
        output.assert_shapes(batch.batch_size)
        return output

    def boundary_report(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "input_schema_id": "stage27_centralized_graph_value_features_v1",
            "output_schema_id": "stage27_normalized_value_with_aux_heads_v1",
            "training_only": True,
            "deployment_actor_receives_critic_output": False,
            "message_layers": self.config.message_layers,
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return total / count
