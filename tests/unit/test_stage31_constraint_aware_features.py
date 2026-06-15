"""Stage 31 Phase D: constraint-aware (v2) actor features make the actor budget-aware."""

from __future__ import annotations

import json
from pathlib import Path

from marl_topology.models import LocalMLPEdgeScorer, LocalMLPEdgeScorerConfig
from marl_topology.models.tensorizers import (
    ACTOR_EDGE_FEATURE_FIELDS,
    ACTOR_EDGE_FEATURE_FIELDS_V2,
    ACTOR_EDGE_FEATURE_SCHEMA_ID_V2,
    tensorize_actor_policy_inputs,
    tensorize_actor_policy_inputs_v2,
)
from marl_topology.policies import ActorPolicyInput

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT
    / "result_save"
    / "evidence_dataset_only"
    / "stage7_completion_learning_evidence_dataset_v1"
    / "learning_evidence.json"
)


def _policy_inputs() -> tuple[ActorPolicyInput, ...]:
    row = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))["rows"][0]
    return tuple(ActorPolicyInput.from_actor_safe_row(r) for r in row["actor_safe_rows"])


def test_v2_schema_extends_v1_with_two_constraint_fields() -> None:
    assert ACTOR_EDGE_FEATURE_FIELDS_V2[: len(ACTOR_EDGE_FEATURE_FIELDS)] == ACTOR_EDGE_FEATURE_FIELDS
    assert ACTOR_EDGE_FEATURE_FIELDS_V2[-2:] == (
        "local_incident_edge_count",
        "endpoint_contention",
    )


def test_v1_tensorizer_is_unchanged() -> None:
    v1 = tensorize_actor_policy_inputs(_policy_inputs())
    assert v1.edge_features.shape[1] == len(ACTOR_EDGE_FEATURE_FIELDS)


def test_v2_tensorizer_adds_local_contention_features() -> None:
    inputs = _policy_inputs()
    v1 = tensorize_actor_policy_inputs(inputs)
    v2 = tensorize_actor_policy_inputs_v2(inputs)
    assert v2.feature_schema_id == ACTOR_EDGE_FEATURE_SCHEMA_ID_V2
    assert v2.edge_features.shape[1] == len(ACTOR_EDGE_FEATURE_FIELDS_V2)
    assert v2.edge_count == v1.edge_count
    # First 11 columns identical to v1.
    assert v2.edge_features[:, : len(ACTOR_EDGE_FEATURE_FIELDS)].equal(v1.edge_features)

    # The two extra columns are consistent with the per-agent incident degree.
    degree_by_agent: dict[str, int] = {}
    for ref in v2.records:
        degree_by_agent[ref.agent_id] = degree_by_agent.get(ref.agent_id, 0) + 1
    for index, ref in enumerate(v2.records):
        degree = float(degree_by_agent[ref.agent_id])
        expected_contention = 0.0 if degree <= 1.0 else 1.0 - 1.0 / degree
        assert float(v2.edge_features[index, -2].item()) == degree
        assert abs(float(v2.edge_features[index, -1].item()) - expected_contention) < 1e-6


def test_contention_is_zero_for_degree_one_and_positive_for_high_degree() -> None:
    v2 = tensorize_actor_policy_inputs_v2(_policy_inputs())
    contention = v2.edge_features[:, -1]
    degree = v2.edge_features[:, -2]
    for c, d in zip(contention.tolist(), degree.tolist()):
        if d <= 1.0:
            assert c == 0.0
        else:
            assert 0.0 < c < 1.0


def test_v2_features_are_consumable_by_a_scorer() -> None:
    v2 = tensorize_actor_policy_inputs_v2(_policy_inputs())
    actor = LocalMLPEdgeScorer(
        LocalMLPEdgeScorerConfig(
            input_dim=len(ACTOR_EDGE_FEATURE_FIELDS_V2), freeze_parameters=True
        )
    )
    logits = actor.score_tensor_batch(v2)
    assert logits.reshape(-1).shape[0] == v2.edge_count
