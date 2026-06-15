"""Local ego-graph message-passing GNN edge scorer.

The deployment actor still outputs edge scores only. Hard topology activation
remains owned by the environment-side assembler.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tensorizers import ACTOR_EDGE_FEATURE_FIELDS, ActorEdgeTensorBatch


LOCAL_GNN_EDGE_SCORER_MODEL_ID = "local_message_passing_gnn_edge_scorer_v2"
LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID = (
    "local_message_passing_gnn_edge_scorer_v3_residual_norm"
)
LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID = "local_role_resource_aware_gnn_edge_scorer_v3"
ACTIVE_STAGE33_GNN_MODEL_ID = LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
STAGE33_ARCHIVED_GNN_VARIANT_IDS = (
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
)

FULL_GNN_FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "global_topology",
        "selected_topology",
        "oracle_label",
        "oracle_membership",
        "consensus_success_probability",
        "latency",
        "energy",
        "reward_surrogate",
        "future_outcome",
        "edge_delta_target",
        "critic_output",
    }
)

# F1 (Stage 34): final edge-score layer init scale. Sized so the v3 /
# role-resource actors start UNSATURATED (mean edge probability ~0.5, far from the
# saturated ~0.96 "select all edges" start that collapsed under shared-spectrum
# interference in every Stage 33 seed) yet COMMITTED enough to learn. The original
# 0.01 was too small: it left the policy near-uniform-random (entropy ~ log(num
# candidates)), so the actor trained stably but never learned (validated: v3 ran
# the full update budget with tau pinned at 0 and entropy at ~max). ~0.2 (between v2's
# saturating 1.0 and 0.01) keeps the init input-dependent with enough score spread
# to commit during training, while the symmetric Xavier init keeps the mean ~0.5.
V3_OUTPUT_HEAD_INIT_GAIN = 0.2
# F2 (Stage 34): role-only vs resource-only feature subsets make the two
# role/resource ablations genuinely distinct models rather than one network
# registered under two names.
ROLE_RESOURCE_FEATURE_MODES = ("role_and_resource", "role_only", "resource_only")


@dataclass(frozen=True, slots=True)
class LocalGNNEdgeScorerConfig:
    model_id: str = LOCAL_GNN_EDGE_SCORER_MODEL_ID
    input_dim: int = len(ACTOR_EDGE_FEATURE_FIELDS)
    hidden_dim: int = 32
    output_dim: int = 1
    message_passing_layers: int = 2
    residual: bool = True

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be declared")
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim != 1:
            raise ValueError("output_dim must be 1")
        if self.message_passing_layers < 2:
            raise ValueError("full GNN requires at least two message-passing layers")


class LocalGNNEdgeScorer(nn.Module):
    """Message-passing scorer over local ego candidate graphs.

    `group_ids` partition candidate edges into local ego graphs. Within each
    group, every candidate edge has one ego node and one neighbor node. Message
    passing updates edge, ego-node, and neighbor-node states using only local
    actor-safe feature rows.
    """

    def __init__(self, config: LocalGNNEdgeScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalGNNEdgeScorerConfig()
        hidden = self.config.hidden_dim
        self.edge_encoder = _mlp(self.config.input_dim, hidden, hidden)
        self.ego_node_encoder = _mlp(self.config.input_dim, hidden, hidden)
        self.neighbor_node_encoder = _mlp(self.config.input_dim, hidden, hidden)
        self.edge_message_layers = nn.ModuleList(
            _mlp(hidden * 3, hidden, hidden)
            for _ in range(self.config.message_passing_layers)
        )
        self.ego_update_layers = nn.ModuleList(
            _mlp(hidden * 2, hidden, hidden)
            for _ in range(self.config.message_passing_layers)
        )
        self.neighbor_update_layers = nn.ModuleList(
            _mlp(hidden * 2, hidden, hidden)
            for _ in range(self.config.message_passing_layers)
        )
        self.edge_update_layers = nn.ModuleList(
            _mlp(hidden * 3, hidden, hidden)
            for _ in range(self.config.message_passing_layers)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.config.output_dim),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.input_dim:
            raise ValueError("edge_features must have shape [edge_count, input_dim]")
        group_ids = torch.arange(edge_features.shape[0], device=edge_features.device)
        return self.forward_with_groups(edge_features, group_ids)

    def forward_with_groups(
        self,
        edge_features: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score flattened local ego graphs.

        The aggregation is permutation invariant within each local graph because
        node messages are reduced by `group_ids`.
        """

        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.input_dim:
            raise ValueError("edge_features must have shape [edge_count, input_dim]")
        if group_ids.shape != (edge_features.shape[0],):
            raise ValueError("group_ids must match edge_count")
        if edge_features.shape[0] == 0:
            return edge_features.new_zeros((0,))

        edge_state = self.edge_encoder(edge_features)
        neighbor_state = self.neighbor_node_encoder(edge_features)
        ego_seed = self.ego_node_encoder(edge_features)
        ego_state = _mean_by_group(ego_seed, group_ids)

        for message_layer, ego_update, neighbor_update, edge_update in zip(
            self.edge_message_layers,
            self.ego_update_layers,
            self.neighbor_update_layers,
            self.edge_update_layers,
        ):
            edge_message = message_layer(
                torch.cat((ego_state, neighbor_state, edge_state), dim=1)
            )
            ego_message = _mean_by_group(edge_message, group_ids)
            next_ego = ego_update(torch.cat((ego_state, ego_message), dim=1))
            next_neighbor = neighbor_update(torch.cat((neighbor_state, edge_message), dim=1))
            next_edge = edge_update(torch.cat((next_ego, next_neighbor, edge_state), dim=1))
            if self.config.residual:
                ego_state = ego_state + next_ego
                neighbor_state = neighbor_state + next_neighbor
                edge_state = edge_state + next_edge
            else:
                ego_state = next_ego
                neighbor_state = next_neighbor
                edge_state = next_edge

        logits = self.score_head(
            torch.cat(
                (
                    ego_state,
                    neighbor_state,
                    edge_state,
                    ego_state * neighbor_state,
                    torch.abs(ego_state - neighbor_state),
                ),
                dim=1,
            )
        )
        return logits.reshape(-1)

    def forward_padded(
        self,
        edge_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score padded local ego graphs with shape [batch, edges, features]."""

        if edge_features.ndim != 3 or edge_features.shape[2] != self.config.input_dim:
            raise ValueError("edge_features must have shape [batch, edge_count, input_dim]")
        if mask.shape != edge_features.shape[:2]:
            raise ValueError("mask must have shape [batch, edge_count]")
        active = mask.to(dtype=torch.bool)
        logits = edge_features.new_zeros(edge_features.shape[:2])
        if not bool(active.any()):
            return logits
        batch_size, edge_count, _feature_dim = edge_features.shape
        flat_features = edge_features.reshape(batch_size * edge_count, self.config.input_dim)
        flat_mask = active.reshape(batch_size * edge_count)
        group_ids = (
            torch.arange(batch_size, device=edge_features.device)
            .repeat_interleave(edge_count)[flat_mask]
        )
        active_logits = self.forward_with_groups(flat_features[flat_mask], group_ids)
        logits.reshape(batch_size * edge_count)[flat_mask] = active_logits
        return logits

    def score_tensor_batch(self, batch: ActorEdgeTensorBatch) -> torch.Tensor:
        return self.forward_with_groups(batch.edge_features, batch.group_ids)

    def boundary_report(self) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "actor_graph_scope": "local_ego_candidate_graph_only",
            "message_passing_layers": self.config.message_passing_layers,
            "uses_edge_to_node_messages": True,
            "uses_node_to_edge_updates": True,
            "permutation_invariant_aggregation": "mean_by_local_graph_group",
            "mask_support": True,
            "global_topology_used": False,
            "oracle_labels_used": False,
            "critic_outputs_used": False,
            "outputs_edge_scores_only": True,
            "full_message_passing_gnn": True,
        }

    @staticmethod
    def validate_input_field_names(field_names: tuple[str, ...]) -> None:
        forbidden = sorted(set(field_names) & FULL_GNN_FORBIDDEN_INPUT_FIELDS)
        if forbidden:
            raise ValueError(f"forbidden GNN actor input fields: {forbidden}")


@dataclass(frozen=True, slots=True)
class LocalGNNV3ResidualNormConfig(LocalGNNEdgeScorerConfig):
    model_id: str = LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
    hidden_dim: int = 48
    message_passing_layers: int = 3
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        LocalGNNEdgeScorerConfig.__post_init__(self)
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")


class LocalMessagePassingGNNV3ResidualNorm(LocalGNNEdgeScorer):
    """Full local message-passing actor with residual normalization controls."""

    def __init__(self, config: LocalGNNV3ResidualNormConfig | None = None) -> None:
        super().__init__(config or LocalGNNV3ResidualNormConfig())
        hidden = self.config.hidden_dim
        layers = self.config.message_passing_layers
        self.edge_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ego_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.neighbor_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(float(self.config.dropout_probability))
        self._reset_v3_parameters()

    def forward_with_groups(
        self,
        edge_features: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.input_dim:
            raise ValueError("edge_features must have shape [edge_count, input_dim]")
        if group_ids.shape != (edge_features.shape[0],):
            raise ValueError("group_ids must match edge_count")
        if edge_features.shape[0] == 0:
            return edge_features.new_zeros((0,))

        edge_state = self.edge_encoder(edge_features)
        neighbor_state = self.neighbor_node_encoder(edge_features)
        ego_seed = self.ego_node_encoder(edge_features)
        ego_state = _mean_by_group(ego_seed, group_ids)

        for layer_index, (message_layer, ego_update, neighbor_update, edge_update) in enumerate(
            zip(
                self.edge_message_layers,
                self.ego_update_layers,
                self.neighbor_update_layers,
                self.edge_update_layers,
                strict=True,
            )
        ):
            edge_message = message_layer(
                torch.cat((ego_state, neighbor_state, edge_state), dim=1)
            )
            ego_message = self._aggregate_message(edge_message, group_ids, layer_index)
            next_ego = ego_update(torch.cat((ego_state, ego_message), dim=1))
            next_neighbor = neighbor_update(torch.cat((neighbor_state, edge_message), dim=1))
            next_edge = edge_update(torch.cat((next_ego, next_neighbor, edge_state), dim=1))
            if self.config.residual:
                ego_state = self.ego_norms[layer_index](ego_state + self.dropout(next_ego))
                neighbor_state = self.neighbor_norms[layer_index](
                    neighbor_state + self.dropout(next_neighbor)
                )
                edge_state = self.edge_norms[layer_index](edge_state + self.dropout(next_edge))
            else:
                ego_state = self.ego_norms[layer_index](next_ego)
                neighbor_state = self.neighbor_norms[layer_index](next_neighbor)
                edge_state = self.edge_norms[layer_index](next_edge)

        logits = self.score_head(
            torch.cat(
                (
                    ego_state,
                    neighbor_state,
                    edge_state,
                    ego_state * neighbor_state,
                    torch.abs(ego_state - neighbor_state),
                ),
                dim=1,
            )
        )
        return logits.reshape(-1)

    def _aggregate_message(
        self,
        edge_message: torch.Tensor,
        group_ids: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        """Reduce each ego group's incident-edge messages into a per-edge ego message.
        v3 (and the role/resource subclass) use the symmetric MEAN; an attention subclass
        overrides this with a learned segment-softmax. Default keeps v3 byte-identical."""
        return _mean_by_group(edge_message, group_ids)

    def boundary_report(self) -> dict[str, object]:
        report = super().boundary_report()
        report.update(
            {
                "model_id": self.config.model_id,
                "residual_connections": bool(self.config.residual),
                "normalization": "layer_norm_per_message_passing_layer",
                "dropout_probability": float(self.config.dropout_probability),
                "message_passing_depth_diagnostic": self.config.message_passing_layers,
                "stage33_active_production_gnn": self.config.model_id == ACTIVE_STAGE33_GNN_MODEL_ID,
            }
        )
        return report

    def _reset_v3_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # F1 fix: re-initialize the final edge-score layer at a small scale so
        # initial logits are ~0 (edge probability ~0.5, near-maximal entropy).
        # The per-layer LayerNorm normalizes node/edge states to unit variance,
        # which inflates the strictly non-negative head interaction block
        # |ego - neighbor| to O(1); with a full-scale Xavier output layer this
        # drove v3 to initialize at probability ~0.96 / entropy ~0.15 and
        # collapse. A small symmetric output init recenters the start to ~0.5
        # while keeping it input-dependent (gradients still flow).
        final_score_layer = self.score_head[-1]
        nn.init.xavier_uniform_(final_score_layer.weight, gain=V3_OUTPUT_HEAD_INIT_GAIN)
        nn.init.zeros_(final_score_layer.bias)


@dataclass(frozen=True, slots=True)
class LocalRoleResourceAwareGNNV3Config(LocalGNNV3ResidualNormConfig):
    model_id: str = LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID
    hidden_dim: int = 48
    message_passing_layers: int = 3
    feature_mode: str = "role_and_resource"

    def __post_init__(self) -> None:
        LocalGNNV3ResidualNormConfig.__post_init__(self)
        if self.feature_mode not in ROLE_RESOURCE_FEATURE_MODES:
            raise ValueError(f"unknown role/resource feature_mode: {self.feature_mode}")


class LocalRoleResourceAwareGNNV3(LocalMessagePassingGNNV3ResidualNorm):
    """Full local GNN with explicit role/resource gates from actor-safe fields."""

    def __init__(self, config: LocalRoleResourceAwareGNNV3Config | None = None) -> None:
        super().__init__(config or LocalRoleResourceAwareGNNV3Config())
        hidden = self.config.hidden_dim
        self._feature_mode = getattr(self.config, "feature_mode", "role_and_resource")
        # The first four actor tensor fields are endpoint role flags; the final
        # four selected fields are local link/resource context (distance, link
        # success probability, latency, energy). The role-only and resource-only
        # ablations consume disjoint feature subsets, so they are genuinely
        # distinct models rather than the same network under two names.
        role_resource_feature_dim = 8 if self._feature_mode == "role_and_resource" else 4
        self.role_resource_encoder = nn.Sequential(
            nn.Linear(role_resource_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.role_resource_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Sigmoid(),
        )
        self._reset_v3_parameters()

    def forward_with_groups(
        self,
        edge_features: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        if edge_features.ndim != 2 or edge_features.shape[1] != self.config.input_dim:
            raise ValueError("edge_features must have shape [edge_count, input_dim]")
        role_resource = self._role_resource_features(edge_features)
        role_resource_state = self.role_resource_encoder(role_resource)
        edge_state_seed = self.edge_encoder(edge_features)
        gated_edge_state = edge_state_seed * self.role_resource_gate(
            torch.cat((edge_state_seed, role_resource_state), dim=1)
        )
        return self._forward_from_edge_state(edge_features, group_ids, gated_edge_state)

    def _forward_from_edge_state(
        self,
        edge_features: torch.Tensor,
        group_ids: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> torch.Tensor:
        if group_ids.shape != (edge_features.shape[0],):
            raise ValueError("group_ids must match edge_count")
        if edge_features.shape[0] == 0:
            return edge_features.new_zeros((0,))

        neighbor_state = self.neighbor_node_encoder(edge_features)
        ego_seed = self.ego_node_encoder(edge_features)
        ego_state = _mean_by_group(ego_seed, group_ids)
        for layer_index, (message_layer, ego_update, neighbor_update, edge_update) in enumerate(
            zip(
                self.edge_message_layers,
                self.ego_update_layers,
                self.neighbor_update_layers,
                self.edge_update_layers,
                strict=True,
            )
        ):
            edge_message = message_layer(
                torch.cat((ego_state, neighbor_state, edge_state), dim=1)
            )
            ego_message = self._aggregate_message(edge_message, group_ids, layer_index)
            next_ego = ego_update(torch.cat((ego_state, ego_message), dim=1))
            next_neighbor = neighbor_update(torch.cat((neighbor_state, edge_message), dim=1))
            next_edge = edge_update(torch.cat((next_ego, next_neighbor, edge_state), dim=1))
            ego_state = self.ego_norms[layer_index](ego_state + self.dropout(next_ego))
            neighbor_state = self.neighbor_norms[layer_index](
                neighbor_state + self.dropout(next_neighbor)
            )
            edge_state = self.edge_norms[layer_index](edge_state + self.dropout(next_edge))

        logits = self.score_head(
            torch.cat(
                (
                    ego_state,
                    neighbor_state,
                    edge_state,
                    ego_state * neighbor_state,
                    torch.abs(ego_state - neighbor_state),
                ),
                dim=1,
            )
        )
        return logits.reshape(-1)

    def _role_resource_features(self, edge_features: torch.Tensor) -> torch.Tensor:
        role_cols = edge_features[:, :4]
        tail = edge_features[:, -4:]
        if tail.shape[1] < 4:
            tail = torch.nn.functional.pad(tail, (0, 4 - tail.shape[1]))
        if self._feature_mode == "role_only":
            return role_cols
        if self._feature_mode == "resource_only":
            return tail
        return torch.cat((role_cols, tail), dim=1)

    def boundary_report(self) -> dict[str, object]:
        report = super().boundary_report()
        report.update(
            {
                "model_id": self.config.model_id,
                "feature_mode": self._feature_mode,
                "role_features_encoded": self._feature_mode in ("role_and_resource", "role_only"),
                "resource_context_features_encoded": self._feature_mode
                in ("role_and_resource", "resource_only"),
                "endpoint_contention_or_projection_history_encoded": "actor_safe_local_context_only",
                "stage33_active_production_gnn": self.config.model_id == ACTIVE_STAGE33_GNN_MODEL_ID,
            }
        )
        return report


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def _mean_by_group(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    unique_groups, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    sums = torch.zeros(
        (unique_groups.shape[0], values.shape[1]),
        dtype=values.dtype,
        device=values.device,
    )
    counts = torch.zeros(
        (unique_groups.shape[0], 1),
        dtype=values.dtype,
        device=values.device,
    )
    sums.index_add_(0, inverse, values)
    counts.index_add_(
        0,
        inverse,
        torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
    )
    return sums[inverse] / counts[inverse].clamp_min(1.0)
