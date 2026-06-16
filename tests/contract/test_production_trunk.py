"""Contract: the production trunk is the decentralized CTDE MARL stack, not the legacy
Stage-33 ego-graph + global-decode path."""

from marl_topology.models import build_model_registry
from marl_topology.models.local_khop_gnn_edge_scorer import (
    LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID,
)
from marl_topology.training.policy_gradient.decentralized_sampler import (
    DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID,
)
from marl_topology.training.decentralized_marl import DECENTRALIZED_CTDE_FLOW_ID
from marl_topology.training.production_trunk import (
    DIAGNOSTIC_BASELINE_ACTOR_MODEL_ID,
    PRODUCTION_TRUNK_ACTOR_MODEL_ID,
    PRODUCTION_TRUNK_FLOW_ID,
    PRODUCTION_TRUNK_SAMPLER_ID,
    production_trunk_designation,
)


def test_production_trunk_is_decentralized_marl_stack() -> None:
    designation = production_trunk_designation()
    assert PRODUCTION_TRUNK_ACTOR_MODEL_ID == LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID
    assert PRODUCTION_TRUNK_SAMPLER_ID == DECENTRALIZED_PER_NODE_MUTUAL_SAMPLER_ID
    assert PRODUCTION_TRUNK_FLOW_ID == DECENTRALIZED_CTDE_FLOW_ID
    assert designation["production_execution_is_decentralized"] is True
    assert designation["production_decode_is_decentralized"] is True
    assert designation["production_is_sequential_mdp"] is True
    assert designation["production_is_bandit"] is False


def test_legacy_stage33_path_is_only_the_diagnostic_baseline() -> None:
    designation = production_trunk_designation()
    # The legacy ego-graph v3 actor is the diagnostic baseline, distinct from production.
    assert DIAGNOSTIC_BASELINE_ACTOR_MODEL_ID != PRODUCTION_TRUNK_ACTOR_MODEL_ID
    assert designation["legacy_stage33_active_gnn_is_diagnostic_only"] is True


def test_production_trunk_actor_is_registered_as_production_deployment_actor() -> None:
    registry = build_model_registry()
    entry = registry[LOCAL_KHOP_GNN_EDGE_SCORER_V1_MODEL_ID]
    assert entry.role == "decentralized_production_actor_edge_scorer"
    assert entry.training_only is False
    assert entry.diagnostic_baseline_only is False
    # It is the production trunk via the decentralized flow, NOT via the legacy Stage-33 gate.
    assert entry.active_for_stage33_production is False
    assert "global_topology" in entry.forbidden_exports
