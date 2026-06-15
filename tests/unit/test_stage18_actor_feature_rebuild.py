import pytest

from marl_topology.data.actor_feature_rebuild import (
    STAGE18_ACTOR_SAFE_FEATURE_SPECS,
    Stage18ActorFeatureViolation,
    build_stage18_actor_safe_feature_rows,
    stage18_feature_schema,
    validate_stage18_actor_safe_feature_row,
)
from marl_topology.data.learning_evidence_stage16 import build_stage16_learning_evidence_dataset


def test_stage18_feature_schema_declares_required_actor_safe_groups() -> None:
    schema = stage18_feature_schema()

    required = [
        "edge_active_prev",
        "edge_active_current_local",
        "local_outgoing_degree_prev",
        "tx_budget_used",
        "tx_budget_remaining",
        "last_projection_rejection_reason",
        "local_conflict_group_occupancy",
        "estimated_link_success_probability",
        "estimated_deadline_delivery_probability",
        "estimated_p2p_latency",
        "estimated_p2p_energy",
        "recent_message_success_rate_local",
        "local_history_embedding_fields",
    ]
    missing = [field for field in required if field not in schema["features"]]
    assert not missing
    assert all(spec.actor_safe for spec in STAGE18_ACTOR_SAFE_FEATURE_SPECS.values())
    assert {spec.source for spec in STAGE18_ACTOR_SAFE_FEATURE_SPECS.values()} >= {
        "local_history",
        "local_projection_feedback",
        "local_message",
        "local_link_estimate",
    }


def test_stage18_actor_safe_feature_rows_add_local_topology_and_resource_context() -> None:
    row = build_stage16_learning_evidence_dataset().rows[0]
    actor_rows = build_stage18_actor_safe_feature_rows(row)

    assert actor_rows
    sample = actor_rows[0]
    for field in (
        "edge_active_current_local",
        "local_selected_edge_count",
        "tx_budget_remaining",
        "rx_capacity_estimate_for_neighbor",
        "local_channel_slot_occupancy",
        "local_conflict_group_occupancy",
        "previous_selected_neighbor_summary",
    ):
        assert field in sample
    validate_stage18_actor_safe_feature_row(sample)


def test_stage18_actor_safe_feature_row_rejects_forbidden_global_fields() -> None:
    row = build_stage16_learning_evidence_dataset().rows[0]
    sample = dict(build_stage18_actor_safe_feature_rows(row)[0])
    sample["consensus_success_probability"] = 0.99

    with pytest.raises(Stage18ActorFeatureViolation):
        validate_stage18_actor_safe_feature_row(sample)


def test_stage18_actor_safe_feature_row_does_not_expose_global_topology_or_oracle() -> None:
    row = build_stage16_learning_evidence_dataset().rows[0]
    sample = build_stage18_actor_safe_feature_rows(row)[0]
    text = repr(sample)

    forbidden = [
        "global_topology",
        "selected_edges",
        "selected_edge_ids",
        "oracle_label",
        "consensus_success_probability",
        "reward_surrogate",
        "future_outcome",
    ]
    assert not [term for term in forbidden if term in text]

