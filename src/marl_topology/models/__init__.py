"""Learnable model scaffolds for MARL topology control."""

from .local_mlp_edge_scorer import (
    LOCAL_MLP_EDGE_SCORER_MODEL_ID,
    LocalMLPEdgeScorer,
    LocalMLPEdgeScorerConfig,
)
from .local_gnn_edge_scorer import (
    ACTIVE_STAGE33_GNN_MODEL_ID,
    LOCAL_GNN_EDGE_SCORER_MODEL_ID,
    LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID,
    LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID,
    STAGE33_ARCHIVED_GNN_VARIANT_IDS,
    LocalGNNEdgeScorer,
    LocalGNNEdgeScorerConfig,
    LocalGNNV3ResidualNormConfig,
    LocalMessagePassingGNNV3ResidualNorm,
    LocalRoleResourceAwareGNNV3,
    LocalRoleResourceAwareGNNV3Config,
)
from .message_passing_graph_edge_scorer import (
    MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID,
    MessagePassingGraphEdgeScorer,
    MessagePassingGraphEdgeScorerConfig,
)
from .model_registry import (
    ModelRegistryEntry,
    active_stage33_production_gnn_entries,
    build_model_registry,
)
from .tensorizers import (
    ACTOR_EDGE_FEATURE_FIELDS,
    CRITIC_GLOBAL_FEATURE_FIELDS,
    ActorEdgeTensorBatch,
    CriticTensorBatch,
    TensorizerViolation,
    tensorize_actor_history_sequence,
    tensorize_actor_policy_inputs,
    tensorize_actor_policy_rows,
    tensorize_critic_evidence_rows,
)

__all__ = [
    "ACTOR_EDGE_FEATURE_FIELDS",
    "ACTIVE_STAGE33_GNN_MODEL_ID",
    "MESSAGE_PASSING_GRAPH_EDGE_SCORER_MODEL_ID",
    "MessagePassingGraphEdgeScorer",
    "MessagePassingGraphEdgeScorerConfig",
    "CRITIC_GLOBAL_FEATURE_FIELDS",
    "LOCAL_MLP_EDGE_SCORER_MODEL_ID",
    "LOCAL_GNN_EDGE_SCORER_MODEL_ID",
    "LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID",
    "LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID",
    "STAGE33_ARCHIVED_GNN_VARIANT_IDS",
    "ActorEdgeTensorBatch",
    "CriticTensorBatch",
    "LocalMLPEdgeScorer",
    "LocalMLPEdgeScorerConfig",
    "LocalGNNEdgeScorer",
    "LocalGNNEdgeScorerConfig",
    "LocalGNNV3ResidualNormConfig",
    "LocalMessagePassingGNNV3ResidualNorm",
    "LocalRoleResourceAwareGNNV3",
    "LocalRoleResourceAwareGNNV3Config",
    "ModelRegistryEntry",
    "TensorizerViolation",
    "active_stage33_production_gnn_entries",
    "build_model_registry",
    "tensorize_actor_history_sequence",
    "tensorize_actor_policy_inputs",
    "tensorize_actor_policy_rows",
    "tensorize_critic_evidence_rows",
]
