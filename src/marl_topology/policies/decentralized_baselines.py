"""Decentralized non-learning baselines.

Each decision rule consumes only ``ActorObservation`` and emits local
``EdgeActionDecision`` values. Environment-side helpers may assemble those
local decisions into a joint topology action for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Mapping

from marl_topology.env import (
    ActorObservation,
    EdgeActionDecision,
    JointTopologyAction,
    build_actor_observation,
)
from marl_topology.link import LinkRecord
from marl_topology.scenario import Scene3D
from marl_topology.topology import CandidateGraph

LocalDecisionRule = Callable[[ActorObservation], tuple[EdgeActionDecision, ...]]


@dataclass(frozen=True, slots=True)
class DecentralizedBaselineResult:
    name: str
    observations: tuple[ActorObservation, ...]
    decisions: tuple[EdgeActionDecision, ...]
    joint_action: JointTopologyAction


class DecentralizedPolicyBaselines:
    """Pure local, non-learning decision rules."""

    @staticmethod
    def no_edges(observation: ActorObservation) -> tuple[EdgeActionDecision, ...]:
        return tuple(
            EdgeActionDecision(observation.agent_id, neighbor.neighbor_id, False)
            for neighbor in observation.local_neighbor_observations
        )

    @staticmethod
    def all_local_edges(observation: ActorObservation) -> tuple[EdgeActionDecision, ...]:
        return tuple(
            EdgeActionDecision(observation.agent_id, neighbor.neighbor_id, True)
            for neighbor in observation.local_neighbor_observations
        )

    @staticmethod
    def reliability_threshold(
        observation: ActorObservation,
        min_success_probability: float,
    ) -> tuple[EdgeActionDecision, ...]:
        if not 0.0 <= min_success_probability <= 1.0:
            raise ValueError("min_success_probability must be in [0, 1]")
        return tuple(
            EdgeActionDecision(
                observation.agent_id,
                neighbor.neighbor_id,
                neighbor.link_success_probability >= min_success_probability,
            )
            for neighbor in observation.local_neighbor_observations
        )

    @staticmethod
    def top_k_reliability(
        observation: ActorObservation,
        k: int,
    ) -> tuple[EdgeActionDecision, ...]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        ranked = sorted(
            observation.local_neighbor_observations,
            key=lambda neighbor: (
                -neighbor.link_success_probability,
                neighbor.estimated_link_energy_j,
                neighbor.edge_id,
            ),
        )
        selected = {neighbor.edge_id for neighbor in ranked[:k]}
        return tuple(
            EdgeActionDecision(
                observation.agent_id,
                neighbor.neighbor_id,
                neighbor.edge_id in selected,
            )
            for neighbor in observation.local_neighbor_observations
        )

    @staticmethod
    def local_random(
        observation: ActorObservation,
        seed: int,
        edge_probability: float,
    ) -> tuple[EdgeActionDecision, ...]:
        if not 0.0 <= edge_probability <= 1.0:
            raise ValueError("edge_probability must be in [0, 1]")
        rng = Random(f"{seed}:{observation.agent_id}")
        return tuple(
            EdgeActionDecision(
                observation.agent_id,
                neighbor.neighbor_id,
                rng.random() < edge_probability,
            )
            for neighbor in observation.local_neighbor_observations
        )


def build_local_observations(
    scene: Scene3D,
    graph: CandidateGraph,
    link_records: Mapping[str, LinkRecord],
    time_step: int,
) -> tuple[ActorObservation, ...]:
    return tuple(
        build_actor_observation(
            scene=scene,
            graph=graph,
            link_records=link_records,
            agent_id=node_id,
            time_step=time_step,
        )
        for node_id in graph.node_ids
    )


def run_decentralized_baseline(
    name: str,
    observations: tuple[ActorObservation, ...],
    graph: CandidateGraph,
    rule: LocalDecisionRule,
    require_mutual: bool = False,
) -> DecentralizedBaselineResult:
    decisions = tuple(decision for obs in observations for decision in rule(obs))
    joint_action = JointTopologyAction.from_local_decisions(
        graph=graph,
        decisions=decisions,
        require_mutual=require_mutual,
    )
    return DecentralizedBaselineResult(
        name=name,
        observations=observations,
        decisions=decisions,
        joint_action=joint_action,
    )
