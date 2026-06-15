"""Actor-safe batch projection for Stage 6.

This module creates in-memory batches from deployment-safe actor observations
or mixed rows after projection. It does not write datasets, instantiate
models, create checkpoints, or execute any run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from marl_topology.env import ActorObservation, validate_actor_observation_payload

from .replay_schema import (
    DEPLOYMENT_ACTOR_INPUT_COLUMNS,
    project_actor_input_row,
    validate_deployment_actor_input_columns,
)


STAGE6_1_ACTOR_SAFE_BATCH_STAGE_ID = (
    "stage_6_1_actor_safe_batch_builder_without_model_or_training"
)
STAGE6_1_VERDICT = "actor_safe_batch_builder_ready_model_training_blocked"
STAGE6_RECOMMENDED_NEXT_STAGE = (
    "stage_7_0_local_actor_policy_interface_contract_with_owner_approval"
)
ACTOR_BATCH_REQUIRED_FIELDS = tuple(sorted(DEPLOYMENT_ACTOR_INPUT_COLUMNS))


class ActorBatchViolation(ValueError):
    """Raised when an actor-safe batch boundary is violated."""


@dataclass(frozen=True)
class ActorSafeBatch:
    """In-memory deployment actor batch with no training side effects."""

    rows: tuple[dict[str, object], ...]
    source: str
    batch_id: str = "actor_safe_batch_stage6_1"

    def __post_init__(self) -> None:
        if not self.source:
            raise ActorBatchViolation("source must be declared")
        seen: set[tuple[str, int]] = set()
        for row in self.rows:
            _validate_complete_actor_row(row)
            key = (str(row["agent_id"]), int(row["time_step"]))
            if key in seen:
                raise ActorBatchViolation(
                    f"duplicate actor batch row for agent/time: {key}"
                )
            seen.add(key)

    @property
    def field_names(self) -> tuple[str, ...]:
        return ACTOR_BATCH_REQUIRED_FIELDS

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def summary(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "source": self.source,
            "row_count": self.row_count,
            "field_names": list(self.field_names),
            "agent_ids": sorted(str(row["agent_id"]) for row in self.rows),
            "time_steps": sorted({int(row["time_step"]) for row in self.rows}),
            "deployment_actor_safe": True,
            "writes_performed": False,
            "dataset_export_allowed": False,
            "model_input_only": False,
        }


def build_actor_safe_batch_from_observations(
    observations: Iterable[ActorObservation],
    *,
    batch_id: str = "actor_safe_batch_from_observations_stage6_1",
) -> ActorSafeBatch:
    """Build an actor-safe batch directly from local observations."""

    rows = tuple(_row_from_observation(observation) for observation in observations)
    return ActorSafeBatch(rows=rows, source="actor_observation", batch_id=batch_id)


def build_actor_safe_batch_from_mixed_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    batch_id: str = "actor_safe_batch_from_projected_rows_stage6_1",
) -> ActorSafeBatch:
    """Project mixed rows to deployment-safe actor rows."""

    actor_rows = tuple(project_actor_input_row(row) for row in rows)
    return ActorSafeBatch(
        rows=actor_rows,
        source="mixed_row_projection",
        batch_id=batch_id,
    )


def build_stage6_1_actor_safe_batch_report(
    observations: Iterable[ActorObservation],
    *,
    stack_ready: bool,
) -> dict[str, object]:
    """Return a dry-run Stage 6.1 batch-builder report."""

    batch = build_actor_safe_batch_from_observations(observations)
    return {
        "stage": STAGE6_1_ACTOR_SAFE_BATCH_STAGE_ID,
        "verdict": STAGE6_1_VERDICT if stack_ready else "actor_safe_batch_guard_blocked",
        "stack_ready": stack_ready,
        "batch_ready": stack_ready and batch.row_count > 0,
        "batch_summary": batch.summary(),
        "dataset_export_allowed": False,
        "checkpoint_creation_allowed": False,
        "training_execution_allowed": False,
        "model_implementation_allowed": False,
        "weight_calibration_allowed": False,
        "final_tau_selection_allowed": False,
        "writes_performed": False,
        "v5_code_migrated": False,
        "recommended_next_stage": STAGE6_RECOMMENDED_NEXT_STAGE,
    }


def _row_from_observation(observation: ActorObservation) -> dict[str, object]:
    payload = observation.to_payload()
    validate_actor_observation_payload(payload)
    _validate_complete_actor_row(payload)
    return payload


def _validate_complete_actor_row(row: Mapping[str, object]) -> None:
    field_set = set(row)
    required = set(ACTOR_BATCH_REQUIRED_FIELDS)
    missing = sorted(required - field_set)
    extra = sorted(field_set - required)
    if missing:
        raise ActorBatchViolation(f"missing actor batch fields: {missing}")
    if extra:
        raise ActorBatchViolation(f"non-actor batch fields: {extra}")
    validate_deployment_actor_input_columns(field_set)
    validate_actor_observation_payload(row)
