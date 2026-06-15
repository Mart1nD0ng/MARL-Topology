"""Build evaluation objects from deterministic scenario fixtures."""

from __future__ import annotations

from dataclasses import dataclass

from marl_topology.link import SimpleLinkModel
from marl_topology.protocol import ConsensusConfig
from marl_topology.scenario import ScenarioFixture
from marl_topology.scenario.scene import Scene3D
from marl_topology.topology import CandidateGraph
from marl_topology.topology.evaluator import TopologyEvaluator
from marl_topology.topology.oracle import TopologyOracle


@dataclass(frozen=True, slots=True)
class EvaluationStack:
    fixture: ScenarioFixture
    scene: Scene3D
    graph: CandidateGraph
    evaluator: TopologyEvaluator
    oracle: TopologyOracle


def build_fixture_stack(fixture: ScenarioFixture) -> EvaluationStack:
    graph = CandidateGraph.from_scene(
        fixture.scene,
        max_distance_m=fixture.max_candidate_distance_m,
    )
    link_records = SimpleLinkModel(
        reference_distance_m=fixture.link_reference_distance_m,
    ).evaluate_graph(graph)
    evaluator = TopologyEvaluator(
        graph=graph,
        link_records=link_records,
        consensus_config=ConsensusConfig(
            quorum_size=fixture.quorum_size,
            success_probability_threshold=fixture.success_probability_threshold,
            deadline_s=fixture.deadline_s,
        ),
    )
    return EvaluationStack(
        fixture=fixture,
        scene=fixture.scene,
        graph=graph,
        evaluator=evaluator,
        oracle=TopologyOracle(evaluator=evaluator, max_exhaustive_edges=10),
    )
