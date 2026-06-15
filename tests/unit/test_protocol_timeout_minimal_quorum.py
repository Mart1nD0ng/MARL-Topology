import pytest

from marl_topology.link import SimpleLinkModel
from marl_topology.protocol import ConsensusConfig
from marl_topology.scenario import make_demo_scene
from marl_topology.topology import CandidateGraph
from marl_topology.topology.evaluator import TopologyEvaluator


def _evaluator(deadline_s: float | None, quorum_size: int = 3) -> TopologyEvaluator:
    scene = make_demo_scene()
    graph = CandidateGraph.from_scene(scene, max_distance_m=80.0)
    links = SimpleLinkModel(reference_distance_m=120.0).evaluate_graph(graph)
    return TopologyEvaluator(
        graph=graph,
        link_records=links,
        consensus_config=ConsensusConfig(
            quorum_size=quorum_size,
            success_probability_threshold=0.45,
            deadline_s=deadline_s,
        ),
    )


def test_consensus_config_rejects_invalid_protocol_parameters() -> None:
    with pytest.raises(ValueError, match="quorum_size must be positive"):
        ConsensusConfig(quorum_size=0)

    with pytest.raises(ValueError, match="success_probability_threshold"):
        ConsensusConfig(quorum_size=1, success_probability_threshold=1.1)

    with pytest.raises(ValueError, match="deadline_s must be nonnegative"):
        ConsensusConfig(quorum_size=1, deadline_s=-0.01)


def test_topology_evaluator_rejects_quorum_larger_than_node_count() -> None:
    scene = make_demo_scene()
    graph = CandidateGraph.from_scene(scene, max_distance_m=80.0)
    links = SimpleLinkModel(reference_distance_m=120.0).evaluate_graph(graph)

    with pytest.raises(ValueError, match="quorum_size cannot exceed node count"):
        TopologyEvaluator(
            graph=graph,
            link_records=links,
            consensus_config=ConsensusConfig(quorum_size=len(graph.node_ids) + 1),
        )


def test_deadline_equality_passes_and_exceedance_fails_binary_success() -> None:
    no_deadline = _evaluator(deadline_s=None)
    selected = set(no_deadline.graph.edge_ids)
    reference = no_deadline.evaluate(selected, topology_id="baseline:full")
    assert reference.consensus.consensus_success is True

    exact_deadline = _evaluator(deadline_s=reference.consensus.latency_s)
    exact = exact_deadline.evaluate(selected, topology_id="baseline:full")
    assert exact.consensus.consensus_success is True
    assert exact.consensus.failure_reason is None

    too_short_deadline = _evaluator(deadline_s=reference.consensus.latency_s - 1e-12)
    timed_out = too_short_deadline.evaluate(selected, topology_id="baseline:full")
    assert timed_out.consensus.consensus_success is False
    assert timed_out.consensus.failure_reason == "deadline_exceeded"


def test_consensus_success_probability_is_not_timeout_gated_in_stage_2() -> None:
    no_deadline = _evaluator(deadline_s=None)
    selected = set(no_deadline.graph.edge_ids)
    reference = no_deadline.evaluate(selected, topology_id="baseline:full")

    too_short_deadline = _evaluator(deadline_s=reference.consensus.latency_s - 1e-12)
    timed_out = too_short_deadline.evaluate(selected, topology_id="baseline:full")

    assert timed_out.metrics["consensus_success"] == 0
    assert timed_out.metrics["consensus_success_probability"] == reference.metrics[
        "consensus_success_probability"
    ]
    assert timed_out.metrics["latency"] == reference.metrics["latency"]


def test_protocol_timeout_does_not_add_metric_names() -> None:
    evaluator = _evaluator(deadline_s=0.01)
    evaluation = evaluator.evaluate(set(evaluator.graph.edge_ids), topology_id="baseline:full")
    metric_names = {row["metric_name"] for row in evaluation.metric_rows()}

    assert metric_names == {
        "consensus_success",
        "consensus_success_probability",
        "latency",
        "energy",
        "topology_diagnostics",
    }
    assert "timeout" not in metric_names
    assert "quorum" not in metric_names
    assert "P_eff" not in metric_names
