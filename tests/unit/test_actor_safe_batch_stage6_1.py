from marl_topology.data import (
    ACTOR_BATCH_REQUIRED_FIELDS,
    STAGE6_1_ACTOR_SAFE_BATCH_STAGE_ID,
    STAGE6_1_VERDICT,
    STAGE6_RECOMMENDED_NEXT_STAGE,
    ActorBatchViolation,
    ActorSafeBatch,
    build_actor_safe_batch_from_mixed_rows,
    build_actor_safe_batch_from_observations,
    build_stage6_1_actor_safe_batch_report,
)
from marl_topology.data.replay_schema import ReplayColumnViolation
from marl_topology.env import ActorObservation


def _observation(agent_id: str = "veh_0", time_step: int = 0) -> ActorObservation:
    return ActorObservation(
        agent_id=agent_id,
        agent_kind="vehicle",
        time_step=time_step,
        local_position_m=(0.0, 0.0, 1.5),
    )


def _mixed_row(agent_id: str = "veh_0") -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "agent_kind": "vehicle",
        "time_step": 0,
        "local_position_m": (0.0, 0.0, 1.5),
        "local_neighbor_observations": (),
        "local_messages": (),
        "local_history": {},
        "action_agent_id": agent_id,
        "action_neighbor_id": "rsu_0",
        "action_activate": True,
        "scenario_id": "scenario_alpha",
        "selected_edge_ids": ("rsu_0--veh_0",),
        "oracle_status": "diagnostic_only",
        "metric_name": "consensus_success_probability",
        "metric_value": 0.9,
        "consensus_success_probability": 0.9,
        "reward_surrogate": -0.1,
        "reward_config_id": "diagnostic_only",
    }


def test_stage6_1_builds_actor_safe_batch_from_observations() -> None:
    batch = build_actor_safe_batch_from_observations(
        (_observation("veh_0"), _observation("rsu_0")),
    )

    assert batch.row_count == 2
    assert batch.field_names == ACTOR_BATCH_REQUIRED_FIELDS
    assert all(set(row) == set(ACTOR_BATCH_REQUIRED_FIELDS) for row in batch.rows)
    assert batch.summary()["deployment_actor_safe"] is True
    assert batch.summary()["writes_performed"] is False


def test_stage6_1_projects_mixed_rows_to_actor_safe_fields() -> None:
    batch = build_actor_safe_batch_from_mixed_rows((_mixed_row(),))

    row = batch.rows[0]
    assert set(row) == set(ACTOR_BATCH_REQUIRED_FIELDS)
    assert "oracle_status" not in row
    assert "consensus_success_probability" not in row
    assert "action_activate" not in row
    assert "reward_surrogate" not in row


def test_stage6_1_missing_actor_field_fails() -> None:
    row = _mixed_row()
    del row["local_messages"]

    try:
        build_actor_safe_batch_from_mixed_rows((row,))
    except ActorBatchViolation as exc:
        assert "missing actor batch fields" in str(exc)
    else:
        raise AssertionError("missing actor field was accepted")


def test_stage6_1_direct_extra_actor_field_fails() -> None:
    row = _observation().to_payload()
    row["global_topology"] = "leak"

    try:
        ActorSafeBatch(rows=(row,), source="direct_test")
    except ActorBatchViolation as exc:
        assert "non-actor batch fields" in str(exc)
    else:
        raise AssertionError("extra global field was accepted")


def test_stage6_1_unsupported_future_target_column_fails_before_projection() -> None:
    row = _mixed_row()
    row["advantage"] = 1.0

    try:
        build_actor_safe_batch_from_mixed_rows((row,))
    except ReplayColumnViolation as exc:
        assert "unsupported replay columns" in str(exc)
    else:
        raise AssertionError("unsupported future target column was accepted")


def test_stage6_1_duplicate_agent_time_rows_fail() -> None:
    try:
        build_actor_safe_batch_from_observations(
            (_observation("veh_0", 0), _observation("veh_0", 0)),
        )
    except ActorBatchViolation as exc:
        assert "duplicate actor batch row" in str(exc)
    else:
        raise AssertionError("duplicate actor/time row was accepted")


def test_stage6_1_report_is_batch_ready_but_execution_blocked() -> None:
    report = build_stage6_1_actor_safe_batch_report(
        (_observation("veh_0"), _observation("rsu_0")),
        stack_ready=True,
    )

    assert report["stage"] == STAGE6_1_ACTOR_SAFE_BATCH_STAGE_ID
    assert report["verdict"] == STAGE6_1_VERDICT
    assert report["batch_ready"] is True
    assert report["writes_performed"] is False
    assert report["dataset_export_allowed"] is False
    assert report["checkpoint_creation_allowed"] is False
    assert report["training_execution_allowed"] is False
    assert report["model_implementation_allowed"] is False
    assert report["weight_calibration_allowed"] is False
    assert report["final_tau_selection_allowed"] is False
    assert report["v5_code_migrated"] is False
    assert report["recommended_next_stage"] == STAGE6_RECOMMENDED_NEXT_STAGE
