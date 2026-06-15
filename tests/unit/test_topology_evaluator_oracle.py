from marl_topology.evaluation import build_demo_stack
from marl_topology.link import SimpleLinkModel
from marl_topology.protocol import ConsensusConfig
from marl_topology.scenario import make_demo_scene
from marl_topology.topology import CandidateGraph
from marl_topology.topology.evaluator import TopologyEvaluator
from marl_topology.topology.oracle import TopologyOracle


def _demo_evaluator() -> TopologyEvaluator:
    scene = make_demo_scene()
    graph = CandidateGraph.from_scene(scene, max_distance_m=80.0)
    links = SimpleLinkModel(reference_distance_m=120.0).evaluate_graph(graph)
    return TopologyEvaluator(
        graph=graph,
        link_records=links,
        consensus_config=ConsensusConfig(
            quorum_size=3,
            success_probability_threshold=0.45,
            deadline_s=0.01,
        ),
    )


def test_consensus_success_responds_to_topology() -> None:
    evaluator = _demo_evaluator()

    empty = evaluator.evaluate(set(), topology_id="baseline:empty")
    full = evaluator.evaluate(set(evaluator.graph.edge_ids), topology_id="baseline:full")

    assert empty.metrics["consensus_success"] == 0
    assert full.metrics["consensus_success"] == 1
    assert empty.metrics["consensus_success_probability"] < full.metrics[
        "consensus_success_probability"
    ]


def test_full_graph_is_baseline_not_oracle() -> None:
    evaluator = _demo_evaluator()
    oracle = TopologyOracle(evaluator=evaluator, max_exhaustive_edges=10)
    result = oracle.solve(random_seed=1)

    assert "full" in result.baseline_evaluations
    assert result.baseline_evaluations["full"].topology_id == "baseline:full"
    assert result.oracle_name != "full"
    assert result.status in {"feasible", "infeasible", "unresolved"}


def test_large_oracle_preserves_unresolved_not_infeasible() -> None:
    evaluator = _demo_evaluator()
    oracle = TopologyOracle(evaluator=evaluator, max_exhaustive_edges=0)

    result = oracle.solve()

    assert result.status == "unresolved"
    assert result.evaluation is None
    assert not result.is_exhaustive


def test_demo_stack_evaluates_required_baselines() -> None:
    _, _, _, oracle = build_demo_stack()
    baselines = oracle.evaluate_baselines(random_seed=7)

    assert {"empty", "full", "greedy_reliability", "random_seed_7"} <= set(baselines)
