"""Model registry for controlled model-family lookup."""

from __future__ import annotations

from dataclasses import dataclass

from .local_mlp_edge_scorer import LOCAL_MLP_EDGE_SCORER_MODEL_ID
from .local_gnn_edge_scorer import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
    LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
)
from .message_passing_graph_edge_scorer import MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
    model_id: str
    family: str
    role: str
    input_schema_id: str
    output_schema_id: str
    training_only: bool
    allowed_stage: str
    forbidden_exports: tuple[str, ...]
    active_for_future_value_baseline: bool = False
    active_for_stage33_production: bool = False
    diagnostic_baseline_only: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "role": self.role,
            "input_schema_id": self.input_schema_id,
            "output_schema_id": self.output_schema_id,
            "training_only": self.training_only,
            "allowed_stage": self.allowed_stage,
            "forbidden_exports": list(self.forbidden_exports),
            "active_for_future_value_baseline": self.active_for_future_value_baseline,
            "active_for_stage33_production": self.active_for_stage33_production,
            "diagnostic_baseline_only": self.diagnostic_baseline_only,
        }


def build_model_registry() -> dict[str, ModelRegistryEntry]:
    return {
        LOCAL_MLP_EDGE_SCORER_MODEL_ID: ModelRegistryEntry(
            model_id=LOCAL_MLP_EDGE_SCORER_MODEL_ID,
            family="local_mlp",
            role="deployment_actor_edge_scorer",
            input_schema_id="actor_local_edge_tensor_v1",
            output_schema_id="actor_policy_local_edge_score_output_v1",
            training_only=False,
            allowed_stage="stage_9_forward_only",
            forbidden_exports=(
                "activate",
                "selected_topology",
                "global_topology",
                "oracle_label",
                "critic_output",
            ),
            diagnostic_baseline_only=True,
        ),
        LOCAL_GNN_EDGE_SCORER_MODEL_ID: ModelRegistryEntry(
            model_id=LOCAL_GNN_EDGE_SCORER_MODEL_ID,
            family="local_message_passing_gnn",
            role="archived_stage33_diagnostic_gnn_edge_scorer",
            input_schema_id="actor_local_edge_tensor_v1_with_group_ids",
            output_schema_id="actor_policy_local_edge_score_output_v1",
            training_only=False,
            allowed_stage="stage_22_full_message_passing_gnn_repair",
            forbidden_exports=(
                "activate",
                "selected_topology",
                "global_topology",
                "oracle_label",
                "critic_output",
            ),
            diagnostic_baseline_only=True,
        ),
        LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID: ModelRegistryEntry(
            model_id=LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
            family="local_message_passing_gnn",
            role="deployment_actor_edge_scorer",
            input_schema_id="actor_local_edge_tensor_v1_with_group_ids",
            output_schema_id="actor_policy_local_edge_score_output_v1",
            training_only=False,
            allowed_stage="stage_33_gnn_stability_and_mappo_loop_unification",
            forbidden_exports=(
                "activate",
                "selected_topology",
                "global_topology",
                "oracle_label",
                "critic_output",
            ),
            active_for_stage33_production=(
                ACTIVE_STAGE33_GNN_MODEL_ID == LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID
            ),
        ),
        MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID: ModelRegistryEntry(
            model_id=MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID,
            family="kround_message_passing_gnn",
            # The decentralized production deployment actor: the K-round message-passing GNN
            # edge scorer recovered as the project trunk (it produced the validated 0.82 result;
            # see docs/URBAN_V2X_RESEARCH_LOG.md). Deployed via the local mutual-acceptance
            # decoder; the centralized graph critic is training-only (CTDE). Not gated by the
            # legacy Stage-33 ego-graph gate, so active_for_stage33_production stays False.
            role="decentralized_production_actor_edge_scorer",
            input_schema_id="actor_graph_payload_v1",
            output_schema_id="actor_policy_local_edge_score_output_v1",
            training_only=False,
            allowed_stage="decentralized_distillation_production",
            forbidden_exports=(
                "activate",
                "selected_topology",
                "global_topology",
                "oracle_label",
                "critic_output",
            ),
            active_for_stage33_production=False,
            diagnostic_baseline_only=False,
        ),
        LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID: ModelRegistryEntry(
            model_id=LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
            family="local_role_resource_aware_gnn",
            role="inactive_stage33_gnn_variant",
            input_schema_id="actor_local_edge_tensor_v1_with_group_ids",
            output_schema_id="actor_policy_local_edge_score_output_v1",
            training_only=False,
            allowed_stage="stage_33_gnn_stability_and_mappo_loop_unification",
            forbidden_exports=(
                "activate",
                "selected_topology",
                "global_topology",
                "oracle_label",
                "critic_output",
            ),
            active_for_stage33_production=(
                ACTIVE_STAGE33_GNN_MODEL_ID == LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID
            ),
            diagnostic_baseline_only=True,
        ),
    }


def active_stage33_production_gnn_entries() -> tuple[ModelRegistryEntry, ...]:
    return tuple(
        entry for entry in build_model_registry().values() if entry.active_for_stage33_production
    )
