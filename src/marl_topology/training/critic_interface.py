"""Centralized critic interface scaffold for future training-only use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


CRITIC_OUTPUT_HEAD_NAMES = (
    "value",
    "feasibility",
    "consensus_success_probability",
    "latency",
    "energy",
    "edge_delta_add",
    "edge_delta_remove",
    "edge_delta_keep",
)
ACTOR_POLICY_FIELD_SET = frozenset(
    {
        "agent_id",
        "agent_kind",
        "time_step",
        "local_position_m",
        "local_neighbor_observations",
        "local_messages",
        "local_history",
    }
)


class CriticInterfaceViolation(ValueError):
    """Raised when critic fields are used outside the training-only boundary."""


@dataclass(frozen=True, slots=True)
class CentralizedCriticInput:
    scenario_id: str
    time_step: int
    node_ids: tuple[str, ...]
    candidate_directed_edges: tuple[str, ...]
    selected_directed_edges: tuple[str, ...]
    centralized_features: Mapping[str, object] = field(default_factory=dict)
    training_only: bool = True

    def __post_init__(self) -> None:
        if not self.training_only:
            raise CriticInterfaceViolation("centralized critic input is training-only")
        if not self.scenario_id:
            raise CriticInterfaceViolation("scenario_id must be declared")
        if self.time_step < 0:
            raise CriticInterfaceViolation("time_step must be nonnegative")
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "candidate_directed_edges", tuple(self.candidate_directed_edges))
        object.__setattr__(self, "selected_directed_edges", tuple(self.selected_directed_edges))
        object.__setattr__(self, "centralized_features", dict(self.centralized_features))

    def to_training_only_payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "time_step": self.time_step,
            "node_ids": self.node_ids,
            "candidate_directed_edges": self.candidate_directed_edges,
            "selected_directed_edges": self.selected_directed_edges,
            "centralized_features": dict(self.centralized_features),
            "training_only": True,
        }


@dataclass(frozen=True, slots=True)
class EdgeDeltaCriticOutput:
    directed_edge_id: str
    edge_delta_add: float
    edge_delta_remove: float
    edge_delta_keep: float
    training_only: bool = True

    def __post_init__(self) -> None:
        if not self.training_only:
            raise CriticInterfaceViolation("edge-delta critic output is training-only")
        tx_id, separator, rx_id = self.directed_edge_id.partition("->")
        if separator != "->":
            raise CriticInterfaceViolation("directed_edge_id must use tx->rx form")
        if not tx_id or not rx_id or tx_id == rx_id:
            raise CriticInterfaceViolation("directed_edge_id must use a non-self tx->rx form")


@dataclass(frozen=True, slots=True)
class CriticOutputHeads:
    value: float
    feasibility: float
    consensus_success_probability: float
    latency: float
    energy: float
    edge_delta_add: Mapping[str, float] = field(default_factory=dict)
    edge_delta_remove: Mapping[str, float] = field(default_factory=dict)
    edge_delta_keep: Mapping[str, float] = field(default_factory=dict)
    training_only: bool = True

    def __post_init__(self) -> None:
        if not self.training_only:
            raise CriticInterfaceViolation("critic output heads are training-only")
        if not 0.0 <= self.feasibility <= 1.0:
            raise CriticInterfaceViolation("feasibility must be in [0, 1]")
        if not 0.0 <= self.consensus_success_probability <= 1.0:
            raise CriticInterfaceViolation(
                "consensus_success_probability must be in [0, 1]"
            )
        if self.latency < 0 or self.energy < 0:
            raise CriticInterfaceViolation("latency and energy must be nonnegative")
        object.__setattr__(self, "edge_delta_add", dict(self.edge_delta_add))
        object.__setattr__(self, "edge_delta_remove", dict(self.edge_delta_remove))
        object.__setattr__(self, "edge_delta_keep", dict(self.edge_delta_keep))

    @property
    def head_names(self) -> tuple[str, ...]:
        return CRITIC_OUTPUT_HEAD_NAMES


def assert_critic_payload_cannot_be_actor_input(payload: Mapping[str, object]) -> None:
    if set(payload) != ACTOR_POLICY_FIELD_SET:
        return
    raise CriticInterfaceViolation("critic payload matched deployment actor input fields")
