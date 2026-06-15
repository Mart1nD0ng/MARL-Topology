"""Replay dataset column boundary contract.

This module classifies future replay/report columns. It does not write replay
files, train models, compute rewards, or create actor/critic datasets.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from marl_topology.env import (
    ACTOR_ALLOWED_FIELDS,
    ACTOR_FORBIDDEN_FIELDS,
    CENTRALIZED_TRAINING_ONLY_FIELDS,
)
from marl_topology.metrics import REGISTERED_METRICS


class ReplayColumnViolation(ValueError):
    """Raised when replay columns violate deployment or registry boundaries."""


DEPLOYMENT_ACTOR_INPUT_COLUMNS = frozenset(ACTOR_ALLOWED_FIELDS)

LOCAL_ACTION_COLUMNS = frozenset(
    {
        "action_agent_id",
        "action_neighbor_id",
        "action_activate",
    }
)

CENTRALIZED_TRAINING_ONLY_COLUMNS = frozenset(CENTRALIZED_TRAINING_ONLY_FIELDS)

REWARD_TRAINING_ONLY_COLUMNS = frozenset(
    {
        "reward_surrogate",
        "reward_reliability_penalty",
        "reward_latency_penalty",
        "reward_energy_penalty",
        "reward_config_id",
    }
)

EVALUATION_ONLY_COLUMNS = frozenset(
    {
        "baseline_name",
        "evaluation_topology_id",
        "fixture_id",
        "is_exhaustive",
        "is_oracle",
        "metric_level",
        "metric_name",
        "metric_value",
        "oracle_name",
        "oracle_selected_edge_ids",
        "searched_topology_count",
        "topology_id",
        "used_for",
        *REGISTERED_METRICS.keys(),
    }
)

UNSUPPORTED_REPLAY_COLUMNS = frozenset(
    {
        "reward",
        "return",
        "advantage",
        "value_target",
        "full_committee_success",
        "future_channel_state",
        "future_consensus_outcome",
        "future_mobility",
    }
)

ALL_REGISTERED_REPLAY_COLUMNS = frozenset(
    DEPLOYMENT_ACTOR_INPUT_COLUMNS
    | LOCAL_ACTION_COLUMNS
    | CENTRALIZED_TRAINING_ONLY_COLUMNS
    | REWARD_TRAINING_ONLY_COLUMNS
    | EVALUATION_ONLY_COLUMNS
)

FORBIDDEN_DEPLOYMENT_ACTOR_COLUMNS = frozenset(
    ACTOR_FORBIDDEN_FIELDS
    | CENTRALIZED_TRAINING_ONLY_COLUMNS
    | REWARD_TRAINING_ONLY_COLUMNS
    | EVALUATION_ONLY_COLUMNS
    | LOCAL_ACTION_COLUMNS
    | UNSUPPORTED_REPLAY_COLUMNS
)


def validate_deployment_actor_input_columns(columns: Iterable[str]) -> None:
    """Require an actor-input batch to contain only deployment-safe columns."""

    column_set = {str(column) for column in columns}
    forbidden = sorted(column_set & FORBIDDEN_DEPLOYMENT_ACTOR_COLUMNS)
    if forbidden:
        raise ReplayColumnViolation(
            f"forbidden deployment actor input columns: {forbidden}"
        )
    unknown = sorted(column_set - DEPLOYMENT_ACTOR_INPUT_COLUMNS)
    if unknown:
        raise ReplayColumnViolation(
            f"unregistered deployment actor input columns: {unknown}"
        )


def validate_registered_replay_columns(columns: Iterable[str]) -> None:
    """Require all replay columns to be known to this contract."""

    column_set = {str(column) for column in columns}
    unsupported = sorted(column_set & UNSUPPORTED_REPLAY_COLUMNS)
    if unsupported:
        raise ReplayColumnViolation(f"unsupported replay columns: {unsupported}")
    unknown = sorted(column_set - ALL_REGISTERED_REPLAY_COLUMNS)
    if unknown:
        raise ReplayColumnViolation(f"unregistered replay columns: {unknown}")


def classify_replay_columns(columns: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Classify registered columns by information boundary."""

    column_set = {str(column) for column in columns}
    validate_registered_replay_columns(column_set)
    return {
        "deployment_actor_input": tuple(
            sorted(column_set & DEPLOYMENT_ACTOR_INPUT_COLUMNS)
        ),
        "local_action": tuple(sorted(column_set & LOCAL_ACTION_COLUMNS)),
        "centralized_training_only": tuple(
            sorted(column_set & CENTRALIZED_TRAINING_ONLY_COLUMNS)
        ),
        "reward_training_only": tuple(
            sorted(column_set & REWARD_TRAINING_ONLY_COLUMNS)
        ),
        "evaluation_only": tuple(sorted(column_set & EVALUATION_ONLY_COLUMNS)),
    }


def project_actor_input_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return only deployment actor input fields from a mixed replay row."""

    validate_registered_replay_columns(row.keys())
    actor_row = {
        key: row[key]
        for key in sorted(set(row) & DEPLOYMENT_ACTOR_INPUT_COLUMNS)
    }
    validate_deployment_actor_input_columns(actor_row.keys())
    return actor_row
