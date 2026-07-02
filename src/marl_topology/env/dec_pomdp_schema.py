"""Dec-POMDP observation and action schema contracts.

This module defines information boundaries only. It does not implement an
environment transition function, actor, critic, replay buffer, or training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from marl_topology.link import LinkRecord
from marl_topology.scenario import Scene3D
from marl_topology.topology import CandidateGraph, canonical_edge_id


class SchemaViolation(ValueError):
    """Raised when deployment actor schema boundaries are violated."""


ACTOR_ALLOWED_FIELDS = frozenset(
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

ACTOR_FORBIDDEN_FIELDS = frozenset(
    {
        "all_node_positions",
        "candidate_edge_ids",
        "candidate_graph",
        "centralized_state",
        "consensus_success",
        "consensus_success_probability",
        "critic_features",
        "critic_state",
        "energy",
        "full_committee_success",
        "full_graph",
        "full_positions",
        "full_topology",
        "future_channel_state",
        "future_consensus_outcome",
        "future_mobility",
        "global_graph",
        "global_reward",
        "global_topology",
        "joint_action",
        "latency",
        "oracle_action",
        "oracle_feasibility",
        "oracle_label",
        "topology_diagnostics",
    }
)

CENTRALIZED_TRAINING_ONLY_FIELDS = frozenset(
    {
        "candidate_edge_ids",
        "global_graph",
        "global_topology",
        "joint_action",
        "metric_names",
        "node_ids",
        "oracle_status",
        "scenario_id",
        "selected_edge_ids",
    }
)


@dataclass(frozen=True, slots=True)
class LocalNeighborObservation:
    neighbor_id: str
    neighbor_kind: str
    edge_id: str
    distance_3d_m: float
    link_success_probability: float
    estimated_link_latency_s: float
    estimated_link_energy_j: float

    def __post_init__(self) -> None:
        if self.distance_3d_m < 0:
            raise ValueError("distance_3d_m must be nonnegative")
        if not 0.0 <= self.link_success_probability <= 1.0:
            raise ValueError("link_success_probability must be in [0, 1]")
        if self.estimated_link_latency_s < 0:
            raise ValueError("estimated_link_latency_s must be nonnegative")
        if self.estimated_link_energy_j < 0:
            raise ValueError("estimated_link_energy_j must be nonnegative")


@dataclass(frozen=True, slots=True)
class LocalMessage:
    sender_id: str
    message_type: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _reject_forbidden_keys(self.payload.keys(), context="local message payload")


@dataclass(frozen=True, slots=True)
class ActorObservation:
    agent_id: str
    agent_kind: str
    time_step: int
    local_position_m: tuple[float, float, float]
    local_neighbor_observations: tuple[LocalNeighborObservation, ...] = ()
    local_messages: tuple[LocalMessage, ...] = ()
    local_history: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_step < 0:
            raise ValueError("time_step must be nonnegative")
        if len(self.local_position_m) != 3:
            raise ValueError("local_position_m must contain three coordinates")
        _reject_forbidden_keys(self.local_history.keys(), context="local history")
        validate_actor_observation_payload(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "agent_kind": self.agent_kind,
            "time_step": self.time_step,
            "local_position_m": self.local_position_m,
            "local_neighbor_observations": self.local_neighbor_observations,
            "local_messages": self.local_messages,
            "local_history": dict(self.local_history),
        }


@dataclass(frozen=True, slots=True)
class EdgeActionDecision:
    agent_id: str
    neighbor_id: str
    activate: bool

    @property
    def edge_id(self) -> str:
        return canonical_edge_id(self.agent_id, self.neighbor_id)


@dataclass(frozen=True, slots=True)
class JointTopologyAction:
    selected_edge_ids: tuple[str, ...]
    proposal_count: int
    assembly_rule: str = "any_active_local_proposal"

    @classmethod
    def from_local_decisions(
        cls,
        graph: CandidateGraph,
        decisions: tuple[EdgeActionDecision, ...],
        require_mutual: bool = False,
    ) -> "JointTopologyAction":
        graph_edges = set(graph.edge_ids)
        proposals: dict[str, set[str]] = {}
        for decision in decisions:
            edge_id = decision.edge_id
            if edge_id not in graph_edges:
                raise SchemaViolation(f"local decision references non-candidate edge: {edge_id}")
            if decision.activate:
                proposals.setdefault(edge_id, set()).add(decision.agent_id)

        if require_mutual:
            selected = tuple(sorted(edge_id for edge_id, agents in proposals.items() if len(agents) >= 2))
            rule = "mutual_active_local_proposals"
        else:
            selected = tuple(sorted(proposals))
            rule = "any_active_local_proposal"
        return cls(selected_edge_ids=selected, proposal_count=len(decisions), assembly_rule=rule)


@dataclass(frozen=True, slots=True)
class CentralizedTrainingView:
    """Training-only CTDE view, explicitly excluded from actor deployment."""

    scenario_id: str
    node_ids: tuple[str, ...]
    candidate_edge_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    metric_names: tuple[str, ...]
    oracle_status: str | None = None

    def to_training_only_payload(self) -> dict[str, object]:
        payload = {
            "scenario_id": self.scenario_id,
            "node_ids": self.node_ids,
            "candidate_edge_ids": self.candidate_edge_ids,
            "selected_edge_ids": self.selected_edge_ids,
            "metric_names": self.metric_names,
            "oracle_status": self.oracle_status,
        }
        unknown = set(payload) - CENTRALIZED_TRAINING_ONLY_FIELDS
        if unknown:
            raise SchemaViolation(f"unknown centralized training fields: {sorted(unknown)}")
        return payload


def validate_actor_observation_payload(payload: Mapping[str, object]) -> None:
    fields = set(payload)
    forbidden = sorted(fields & ACTOR_FORBIDDEN_FIELDS)
    if forbidden:
        raise SchemaViolation(f"forbidden actor observation fields: {forbidden}")
    unknown = sorted(fields - ACTOR_ALLOWED_FIELDS)
    if unknown:
        raise SchemaViolation(f"unregistered actor observation fields: {unknown}")
    if "local_history" in payload and isinstance(payload["local_history"], Mapping):
        _reject_forbidden_keys(payload["local_history"].keys(), context="local history")


def build_actor_observation(
    scene: Scene3D,
    graph: CandidateGraph,
    link_records: Mapping[str, LinkRecord],
    agent_id: str,
    time_step: int,
    local_messages: tuple[LocalMessage, ...] = (),
    local_history: Mapping[str, object] | None = None,
) -> ActorObservation:
    node = scene.get_node(agent_id)
    neighbor_observations: list[LocalNeighborObservation] = []
    for edge in graph.edges:
        if edge.node_u != agent_id and edge.node_v != agent_id:
            continue
        neighbor_id = edge.node_v if edge.node_u == agent_id else edge.node_u
        neighbor = scene.get_node(neighbor_id)
        link = link_records[edge.edge_id]
        neighbor_observations.append(
            LocalNeighborObservation(
                neighbor_id=neighbor.node_id,
                neighbor_kind=neighbor.kind.value,
                edge_id=edge.edge_id,
                distance_3d_m=link.distance_3d_m,
                link_success_probability=link.link_success_probability,
                estimated_link_latency_s=link.latency_s,
                estimated_link_energy_j=link.energy_j,
            )
        )
    return ActorObservation(
        agent_id=node.node_id,
        agent_kind=node.kind.value,
        time_step=time_step,
        local_position_m=(node.position.x_m, node.position.y_m, node.position.z_m),
        local_neighbor_observations=tuple(sorted(neighbor_observations, key=lambda item: item.edge_id)),
        local_messages=local_messages,
        local_history=local_history or {},
    )


def _reject_forbidden_keys(keys: object, context: str) -> None:
    key_set = {str(key) for key in keys}
    forbidden = sorted(key_set & ACTOR_FORBIDDEN_FIELDS)
    if forbidden:
        raise SchemaViolation(f"forbidden {context} fields: {forbidden}")
