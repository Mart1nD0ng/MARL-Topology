from marl_topology.data import (
    ALL_REGISTERED_REPLAY_COLUMNS,
    CENTRALIZED_TRAINING_ONLY_COLUMNS,
    DEPLOYMENT_ACTOR_INPUT_COLUMNS,
    EVALUATION_ONLY_COLUMNS,
    LOCAL_ACTION_COLUMNS,
    REWARD_TRAINING_ONLY_COLUMNS,
    ReplayColumnViolation,
    classify_replay_columns,
    project_actor_input_row,
    validate_deployment_actor_input_columns,
    validate_registered_replay_columns,
)


def test_deployment_actor_input_columns_pass() -> None:
    validate_deployment_actor_input_columns(DEPLOYMENT_ACTOR_INPUT_COLUMNS)


def test_global_oracle_metric_action_reward_and_future_columns_fail_actor_input() -> None:
    forbidden_groups = (
        {"global_topology"},
        {"oracle_status"},
        {"consensus_success", "latency"},
        {"action_neighbor_id", "action_activate"},
        {"reward"},
        {"reward_surrogate"},
        {"future_consensus_outcome"},
    )

    for group in forbidden_groups:
        try:
            validate_deployment_actor_input_columns(
                set(DEPLOYMENT_ACTOR_INPUT_COLUMNS) | group
            )
        except ReplayColumnViolation:
            pass
        else:
            raise AssertionError(f"forbidden actor columns accepted: {group}")


def test_registered_mixed_replay_columns_classify_by_boundary() -> None:
    columns = {
        "agent_id",
        "agent_kind",
        "time_step",
        "local_neighbor_observations",
        "action_neighbor_id",
        "action_activate",
        "scenario_id",
        "selected_edge_ids",
        "oracle_status",
        "metric_name",
        "metric_value",
        "consensus_success",
        "latency",
        "reward_surrogate",
        "reward_reliability_penalty",
    }

    classification = classify_replay_columns(columns)

    assert set(classification["deployment_actor_input"]) == {
        "agent_id",
        "agent_kind",
        "time_step",
        "local_neighbor_observations",
    }
    assert set(classification["local_action"]) == {"action_neighbor_id", "action_activate"}
    assert set(classification["centralized_training_only"]) == {
        "scenario_id",
        "selected_edge_ids",
        "oracle_status",
    }
    assert set(classification["evaluation_only"]) == {
        "metric_name",
        "metric_value",
        "consensus_success",
        "latency",
    }
    assert set(classification["reward_training_only"]) == {
        "reward_surrogate",
        "reward_reliability_penalty",
    }


def test_actor_projection_excludes_centralized_action_and_evaluation_columns() -> None:
    row = {
        "agent_id": "veh_0",
        "agent_kind": "vehicle",
        "time_step": 0,
        "local_position_m": (0.0, 0.0, 1.5),
        "local_neighbor_observations": (),
        "local_messages": (),
        "local_history": {},
        "action_neighbor_id": "rsu_0",
        "action_activate": True,
        "scenario_id": "demo_stage2",
        "selected_edge_ids": ("rsu_0--veh_0",),
        "oracle_status": "feasible",
        "consensus_success": 1,
        "metric_name": "consensus_success",
        "metric_value": 1,
    }

    actor_row = project_actor_input_row(row)

    assert set(actor_row) == set(DEPLOYMENT_ACTOR_INPUT_COLUMNS)
    validate_deployment_actor_input_columns(actor_row.keys())
    assert "oracle_status" not in actor_row
    assert "consensus_success" not in actor_row
    assert "action_activate" not in actor_row


def test_unknown_and_unsupported_replay_columns_fail() -> None:
    invalid_sets = (
        {"agent_id", "unknown_feature"},
        {"agent_id", "reward"},
        {"agent_id", "advantage"},
        {"agent_id", "value_target"},
    )

    for columns in invalid_sets:
        try:
            validate_registered_replay_columns(columns)
        except ReplayColumnViolation:
            pass
        else:
            raise AssertionError(f"invalid replay columns accepted: {columns}")


def test_registered_replay_column_sets_are_disjoint_where_required() -> None:
    assert DEPLOYMENT_ACTOR_INPUT_COLUMNS <= ALL_REGISTERED_REPLAY_COLUMNS
    assert LOCAL_ACTION_COLUMNS <= ALL_REGISTERED_REPLAY_COLUMNS
    assert CENTRALIZED_TRAINING_ONLY_COLUMNS <= ALL_REGISTERED_REPLAY_COLUMNS
    assert EVALUATION_ONLY_COLUMNS <= ALL_REGISTERED_REPLAY_COLUMNS
    assert REWARD_TRAINING_ONLY_COLUMNS <= ALL_REGISTERED_REPLAY_COLUMNS
    assert not (DEPLOYMENT_ACTOR_INPUT_COLUMNS & LOCAL_ACTION_COLUMNS)
    assert not (DEPLOYMENT_ACTOR_INPUT_COLUMNS & CENTRALIZED_TRAINING_ONLY_COLUMNS)
    assert not (DEPLOYMENT_ACTOR_INPUT_COLUMNS & EVALUATION_ONLY_COLUMNS)
    assert not (DEPLOYMENT_ACTOR_INPUT_COLUMNS & REWARD_TRAINING_ONLY_COLUMNS)
