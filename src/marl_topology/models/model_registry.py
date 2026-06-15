"""Model registry for controlled model-family lookup."""

from __future__ import annotations

from dataclasses import dataclass

from marl_topology.training.critic_features import VALUE_CRITIC_FEATURE_FIELDS

from .centralized_message_passing_graph_critic import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
)
from .centralized_mlp_critic import CENTRALIZED_MLP_CRITIC_BASELINE_ID, CRITIC_HEAD_NAMES
from .enriched_centralized_mlp_critic import ENRICHED_CENTRALIZED_MLP_CRITIC_ID
from .local_mlp_edge_scorer import LOCAL_MLP_EDGE_SCORER_MODEL_ID
from .local_gnn_edge_scorer import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
    LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
)
from .tensorizers import ACTOR_EDGE_FEATURE_FIELDS, CRITIC_GLOBAL_FEATURE_FIELDS


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
        CENTRALIZED_MLP_CRITIC_BASELINE_ID: ModelRegistryEntry(
            model_id=CENTRALIZED_MLP_CRITIC_BASELINE_ID,
            family="centralized_mlp",
            role="training_only_centralized_critic",
            input_schema_id="centralized_critic_global_tensor_v1",
            output_schema_id="centralized_critic_multi_head_v1",
            training_only=True,
            allowed_stage="stage_9_forward_only",
            forbidden_exports=tuple(ACTOR_EDGE_FEATURE_FIELDS)
            + tuple(CRITIC_GLOBAL_FEATURE_FIELDS)
            + CRITIC_HEAD_NAMES,
        ),
        ENRICHED_CENTRALIZED_MLP_CRITIC_ID: ModelRegistryEntry(
            model_id=ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
            family="enriched_centralized_mlp",
            role="training_only_value_critic",
            input_schema_id="stage27_pre_action_value_critic_features_v1",
            output_schema_id="stage27_normalized_value_with_aux_heads_v1",
            training_only=True,
            allowed_stage="stage_27_critic_baseline_repair",
            forbidden_exports=tuple(ACTOR_EDGE_FEATURE_FIELDS)
            + tuple(VALUE_CRITIC_FEATURE_FIELDS)
            + ("normalized_value", "feasibility_logit", "consensus_proxy"),
            active_for_future_value_baseline=False,
        ),
        CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID: ModelRegistryEntry(
            model_id=CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
            family="centralized_message_passing_graph",
            role="training_only_value_critic_candidate",
            input_schema_id="stage27_centralized_graph_value_features_v1",
            output_schema_id="stage27_normalized_value_with_aux_heads_v1",
            training_only=True,
            allowed_stage="stage_27_critic_baseline_repair",
            forbidden_exports=tuple(ACTOR_EDGE_FEATURE_FIELDS)
            + ("normalized_value", "graph_state", "centralized_graph_features"),
            active_for_future_value_baseline=True,
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
