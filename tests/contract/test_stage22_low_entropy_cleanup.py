from pathlib import Path

import yaml

from marl_topology.models import LOCAL_GNN_EDGE_SCORER_MODEL_ID, build_model_registry
from marl_topology.policies import (
    ACTIVE_ACTION_SEMANTICS_ID,
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    build_active_action_semantics_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_stage22_active_registry_contains_only_selected_physical_semantics() -> None:
    active = build_active_action_semantics_registry()

    assert ACTIVE_ACTION_SEMANTICS_ID == UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
    assert set(active) == {UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID}


def test_stage22_model_registry_promotes_full_gnn_not_toy_v1() -> None:
    registry = build_model_registry()
    entry = registry[LOCAL_GNN_EDGE_SCORER_MODEL_ID]

    assert LOCAL_GNN_EDGE_SCORER_MODEL_ID == "local_message_passing_gnn_edge_scorer_v2"
    assert entry.family == "local_message_passing_gnn"
    assert entry.allowed_stage == "stage_22_full_message_passing_gnn_repair"


