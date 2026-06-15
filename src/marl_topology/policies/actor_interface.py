"""Stage 8 actor interface scaffold without model code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from marl_topology.data.actor_batch import ACTOR_BATCH_REQUIRED_FIELDS

from .edge_scores import EdgeScoreBatch, EdgeScoreRecord
from .interface_contract import (
    ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID,
    ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS,
    ACTOR_POLICY_INPUT_SCHEMA_ID,
    PolicyInterfaceViolation,
    validate_actor_policy_input_row,
)


ACTOR_EDGE_SCORE_OUTPUT_FORBIDDEN_FIELDS = frozenset(
    {
        "global_topology",
        "selected_directed_edges",
        "selected_edge_ids",
        "joint_action",
        "oracle_label",
        "consensus_success_probability",
        "learning_targets",
        "edge_delta_targets",
        "reward_surrogate",
        "critic_output",
    }
)


@dataclass(frozen=True, slots=True)
class ActorPolicyInput:
    agent_id: str
    agent_kind: str
    time_step: int
    local_position_m: tuple[float, float, float]
    local_neighbor_observations: tuple[object, ...] = ()
    local_messages: tuple[object, ...] = ()
    local_history: Mapping[str, object] = field(default_factory=dict)
    schema_id: str = ACTOR_POLICY_INPUT_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != ACTOR_POLICY_INPUT_SCHEMA_ID:
            raise PolicyInterfaceViolation("actor input schema_id mismatch")
        validate_actor_policy_input_row(self.to_actor_safe_row())

    @classmethod
    def from_actor_safe_row(cls, row: Mapping[str, object]) -> "ActorPolicyInput":
        validate_actor_policy_input_row(row)
        return cls(
            agent_id=str(row["agent_id"]),
            agent_kind=str(row["agent_kind"]),
            time_step=int(row["time_step"]),
            local_position_m=tuple(row["local_position_m"]),  # type: ignore[arg-type]
            local_neighbor_observations=tuple(row["local_neighbor_observations"]),  # type: ignore[arg-type]
            local_messages=tuple(row["local_messages"]),  # type: ignore[arg-type]
            local_history=dict(row["local_history"]),  # type: ignore[arg-type]
        )

    def to_actor_safe_row(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "agent_kind": self.agent_kind,
            "time_step": self.time_step,
            "local_position_m": self.local_position_m,
            "local_neighbor_observations": self.local_neighbor_observations,
            "local_messages": self.local_messages,
            "local_history": dict(self.local_history),
        }

    @property
    def field_names(self) -> tuple[str, ...]:
        return ACTOR_BATCH_REQUIRED_FIELDS


@dataclass(frozen=True, slots=True)
class PolicyState:
    state_id: str
    local_memory: Mapping[str, object] = field(default_factory=dict)
    source: str = "actor_local_history_or_permitted_messages"

    def __post_init__(self) -> None:
        if not self.state_id:
            raise PolicyInterfaceViolation("policy state_id must be declared")
        forbidden = sorted(set(self.local_memory) & set(ACTOR_POLICY_FORBIDDEN_INPUT_FIELDS))
        if forbidden:
            raise PolicyInterfaceViolation(f"forbidden policy state fields: {forbidden}")
        object.__setattr__(self, "local_memory", dict(self.local_memory))


@dataclass(frozen=True, slots=True)
class ActorPolicyOutput:
    agent_id: str
    edge_scores: EdgeScoreBatch
    policy_state: PolicyState | None = None
    schema_id: str = ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID:
            raise PolicyInterfaceViolation("actor output schema_id mismatch")
        mismatched = [
            record.directed_edge_id
            for record in self.edge_scores.edge_scores
            if record.agent_id != self.agent_id
        ]
        if mismatched:
            raise PolicyInterfaceViolation(
                f"edge scores do not belong to actor {self.agent_id}: {mismatched}"
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "agent_id": self.agent_id,
            "edge_scores": self.edge_scores.to_payload(),
            "policy_state_id": self.policy_state.state_id if self.policy_state else None,
            "contains_final_topology": False,
            "contains_oracle_label": False,
            "contains_consensus_metric": False,
        }


def validate_actor_policy_output_payload(payload: Mapping[str, object]) -> None:
    forbidden = sorted(set(payload) & ACTOR_EDGE_SCORE_OUTPUT_FORBIDDEN_FIELDS)
    if forbidden:
        raise PolicyInterfaceViolation(f"forbidden actor output fields: {forbidden}")
    if "edge_scores" not in payload:
        raise PolicyInterfaceViolation("actor output must contain edge_scores")
