"""Minimal Dec-POMDP reset/step wrapper.

This wrapper connects the existing Stage 2 scene, graph, link records,
Dec-POMDP schema, and topology evaluator. It does not implement reward,
learning, replay storage, actor models, critic models, or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from marl_topology.env.dec_pomdp_schema import (
    ActorObservation,
    EdgeActionDecision,
    JointTopologyAction,
    SchemaViolation,
    build_actor_observation,
)
from marl_topology.scenario import Scene3D
from marl_topology.topology import CandidateGraph
from marl_topology.topology.evaluator import TopologyEvaluation, TopologyEvaluator


@dataclass(frozen=True, slots=True)
class DecPOMDPResetResult:
    time_step: int
    observations: tuple[ActorObservation, ...]
    info: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DecPOMDPStepResult:
    time_step: int
    observations: tuple[ActorObservation, ...]
    joint_action: JointTopologyAction
    evaluation: TopologyEvaluation
    terminated: bool
    truncated: bool
    info: Mapping[str, object]


class MinimalDecPOMDPEnv:
    """Deterministic wrapper around the current clean skeleton.

    The environment exposes local observations and accepts local edge decisions.
    Metrics are returned only through environment-side evaluation results.
    """

    def __init__(
        self,
        scene: Scene3D,
        graph: CandidateGraph,
        evaluator: TopologyEvaluator,
        horizon: int = 1,
        require_mutual_actions: bool = False,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if scene.scenario_id != graph.scenario_id:
            raise ValueError("scene and graph scenario_id must match")
        if graph != evaluator.graph:
            raise ValueError("evaluator graph must match environment graph")
        self.scene = scene
        self.graph = graph
        self.evaluator = evaluator
        self.horizon = horizon
        self.require_mutual_actions = require_mutual_actions
        self._time_step = 0
        self._has_reset = False

    @property
    def time_step(self) -> int:
        return self._time_step

    def reset(self, time_step: int = 0) -> DecPOMDPResetResult:
        if time_step < 0:
            raise ValueError("time_step must be nonnegative")
        self._time_step = time_step
        self._has_reset = True
        return DecPOMDPResetResult(
            time_step=self._time_step,
            observations=self._build_observations(self._time_step),
            info={
                "scenario_id": self.graph.scenario_id,
                "physics_regime": self.graph.physics_regime,
                "node_count": len(self.graph.node_ids),
                "candidate_edge_count": len(self.graph.edge_ids),
                "horizon": self.horizon,
            },
        )

    def step(
        self,
        decisions: Iterable[EdgeActionDecision],
        topology_id: str | None = None,
    ) -> DecPOMDPStepResult:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        decision_tuple = tuple(decisions)
        joint_action = JointTopologyAction.from_local_decisions(
            graph=self.graph,
            decisions=decision_tuple,
            require_mutual=self.require_mutual_actions,
        )
        evaluation = self.evaluator.evaluate(
            set(joint_action.selected_edge_ids),
            topology_id=topology_id or f"env_step:{self._time_step}",
        )
        self._time_step += 1
        terminated = self._time_step >= self.horizon
        return DecPOMDPStepResult(
            time_step=self._time_step,
            observations=self._build_observations(self._time_step),
            joint_action=joint_action,
            evaluation=evaluation,
            terminated=terminated,
            truncated=False,
            info={
                "assembly_rule": joint_action.assembly_rule,
                "proposal_count": joint_action.proposal_count,
                "selected_edge_count": len(joint_action.selected_edge_ids),
                "evaluation_topology_id": evaluation.topology_id,
            },
        )

    def _build_observations(self, time_step: int) -> tuple[ActorObservation, ...]:
        observations = []
        for node_id in self.graph.node_ids:
            observations.append(
                build_actor_observation(
                    scene=self.scene,
                    graph=self.graph,
                    link_records=self.evaluator.link_records,
                    agent_id=node_id,
                    time_step=time_step,
                )
            )
        return tuple(observations)


def ensure_actor_observations_do_not_contain_metrics(
    observations: Iterable[ActorObservation],
) -> None:
    forbidden_metric_fields = {
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    }
    for observation in observations:
        payload = observation.to_payload()
        leaked = sorted(set(payload) & forbidden_metric_fields)
        if leaked:
            raise SchemaViolation(f"metric fields leaked into actor observation: {leaked}")
