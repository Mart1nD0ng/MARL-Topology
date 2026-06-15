"""Learnable model scaffolds for MARL topology control."""

from .centralized_mlp_critic import (
    CENTRALIZED_MLP_CRITIC_BASELINE_ID,
    CRITIC_HEAD_NAMES,
    CentralizedCriticTensorOutput,
    CentralizedMLPCriticBaseline,
    CentralizedMLPCriticConfig,
)
from .centralized_message_passing_graph_critic import (
    CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID,
    CentralizedMessagePassingGraphCritic,
    CentralizedMessagePassingGraphCriticConfig,
    GraphCriticBatch,
    GraphCriticOutput,
)
from .enriched_centralized_mlp_critic import (
    ENRICHED_CENTRALIZED_MLP_CRITIC_ID,
    EnrichedCentralizedMLPCritic,
    EnrichedCentralizedMLPCriticConfig,
    EnrichedCriticOutput,
)
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
from .local_attention_gnn_edge_scorer import (
    LOCAL_ATTENTION_GNN_EDGE_SCORER_MODEL_ID,
    LocalAttentionGNNEdgeScorer,
    LocalAttentionGNNEdgeScorerConfig,
)
from .local_temporal_gnn_edge_scorer import (
    LOCAL_TEMPORAL_GNN_EDGE_SCORER_MODEL_ID,
    LocalTemporalGNNEdgeScorer,
    LocalTemporalGNNEdgeScorerConfig,
)
from .local_gru_edge_scorer import (
    LOCAL_GRU_EDGE_SCORER_MODEL_ID,
    LocalGRUEdgeScorer,
    LocalGRUEdgeScorerConfig,
)
from .local_lstm_edge_scorer import (
    LOCAL_LSTM_EDGE_SCORER_MODEL_ID,
    LocalLSTMEdgeScorer,
    LocalLSTMEdgeScorerConfig,
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
    "CENTRALIZED_MLP_CRITIC_BASELINE_ID",
    "CENTRALIZED_MESSAGE_PASSING_GRAPH_CRITIC_ID",
    "CRITIC_GLOBAL_FEATURE_FIELDS",
    "CRITIC_HEAD_NAMES",
    "LOCAL_MLP_EDGE_SCORER_MODEL_ID",
    "LOCAL_GNN_EDGE_SCORER_MODEL_ID",
    "LOCAL_GNN_V3_RESIDUAL_NORM_MODEL_ID",
    "LOCAL_GRU_EDGE_SCORER_MODEL_ID",
    "LOCAL_LSTM_EDGE_SCORER_MODEL_ID",
    "LOCAL_ROLE_RESOURCE_GNN_V3_MODEL_ID",
    "STAGE33_ARCHIVED_GNN_VARIANT_IDS",
    "ActorEdgeTensorBatch",
    "CentralizedCriticTensorOutput",
    "CentralizedMessagePassingGraphCritic",
    "CentralizedMessagePassingGraphCriticConfig",
    "CentralizedMLPCriticBaseline",
    "CentralizedMLPCriticConfig",
    "CriticTensorBatch",
    "ENRICHED_CENTRALIZED_MLP_CRITIC_ID",
    "EnrichedCentralizedMLPCritic",
    "EnrichedCentralizedMLPCriticConfig",
    "EnrichedCriticOutput",
    "GraphCriticBatch",
    "GraphCriticOutput",
    "LocalMLPEdgeScorer",
    "LocalMLPEdgeScorerConfig",
    "LocalGNNEdgeScorer",
    "LocalGNNEdgeScorerConfig",
    "LocalGNNV3ResidualNormConfig",
    "LocalGRUEdgeScorer",
    "LocalGRUEdgeScorerConfig",
    "LocalLSTMEdgeScorer",
    "LocalLSTMEdgeScorerConfig",
    "LOCAL_ATTENTION_GNN_EDGE_SCORER_MODEL_ID",
    "LOCAL_TEMPORAL_GNN_EDGE_SCORER_MODEL_ID",
    "LocalAttentionGNNEdgeScorer",
    "LocalAttentionGNNEdgeScorerConfig",
    "LocalMessagePassingGNNV3ResidualNorm",
    "LocalTemporalGNNEdgeScorer",
    "LocalTemporalGNNEdgeScorerConfig",
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
