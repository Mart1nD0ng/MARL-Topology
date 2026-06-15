from marl_topology.link import SimpleLinkModel
from marl_topology.policies import PolicyBaselines
from marl_topology.scenario import make_demo_scene
from marl_topology.topology import CandidateGraph


def test_policy_baselines_are_non_learning_and_deterministic() -> None:
    graph = CandidateGraph.from_scene(make_demo_scene(), max_distance_m=80.0)
    links = SimpleLinkModel().evaluate_graph(graph)

    empty = PolicyBaselines.empty(graph)
    full = PolicyBaselines.full(graph)
    random_a = PolicyBaselines.random(graph, seed=11)
    random_b = PolicyBaselines.random(graph, seed=11)
    greedy = PolicyBaselines.greedy_reliability(graph, links)

    assert empty.edge_ids == ()
    assert full.edge_ids == graph.edge_ids
    assert full.is_oracle is False
    assert random_a.edge_ids == random_b.edge_ids
    assert set(greedy.edge_ids) <= set(graph.edge_ids)
