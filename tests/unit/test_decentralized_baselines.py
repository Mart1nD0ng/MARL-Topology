from marl_topology.evaluation import build_demo_stack
from marl_topology.env import validate_actor_observation_payload
from marl_topology.policies import (
    DecentralizedPolicyBaselines,
    build_local_observations,
    run_decentralized_baseline,
)


def _observations():
    scene, graph, evaluator, _ = build_demo_stack()
    observations = build_local_observations(scene, graph, evaluator.link_records, time_step=0)
    return graph, observations


def test_local_observations_are_actor_schema_valid() -> None:
    graph, observations = _observations()

    assert {obs.agent_id for obs in observations} == set(graph.node_ids)
    for obs in observations:
        validate_actor_observation_payload(obs.to_payload())
        assert "global_topology" not in obs.to_payload()
        assert "oracle_label" not in obs.to_payload()


def test_decentralized_no_edges_selects_empty_topology() -> None:
    graph, observations = _observations()
    result = run_decentralized_baseline(
        name="decentralized_no_edges",
        observations=observations,
        graph=graph,
        rule=DecentralizedPolicyBaselines.no_edges,
    )

    assert result.joint_action.selected_edge_ids == ()
    assert result.joint_action.proposal_count > 0


def test_decentralized_all_local_edges_selects_full_candidate_graph() -> None:
    graph, observations = _observations()
    result = run_decentralized_baseline(
        name="decentralized_all_local_edges",
        observations=observations,
        graph=graph,
        rule=DecentralizedPolicyBaselines.all_local_edges,
    )

    assert result.joint_action.selected_edge_ids == graph.edge_ids


def test_decentralized_threshold_rule_uses_explicit_threshold() -> None:
    graph, observations = _observations()
    result = run_decentralized_baseline(
        name="decentralized_threshold",
        observations=observations,
        graph=graph,
        rule=lambda obs: DecentralizedPolicyBaselines.reliability_threshold(
            obs,
            min_success_probability=0.65,
        ),
    )

    assert set(result.joint_action.selected_edge_ids) < set(graph.edge_ids)
    assert set(result.joint_action.selected_edge_ids)


def test_decentralized_top_k_reliability_is_local_and_bounded() -> None:
    graph, observations = _observations()
    result = run_decentralized_baseline(
        name="decentralized_top1_reliability",
        observations=observations,
        graph=graph,
        rule=lambda obs: DecentralizedPolicyBaselines.top_k_reliability(obs, k=1),
    )

    assert set(result.joint_action.selected_edge_ids) <= set(graph.edge_ids)
    assert len(result.joint_action.selected_edge_ids) <= len(observations)


def test_decentralized_random_rule_is_seeded_and_local() -> None:
    graph, observations = _observations()
    result_a = run_decentralized_baseline(
        name="random_a",
        observations=observations,
        graph=graph,
        rule=lambda obs: DecentralizedPolicyBaselines.local_random(
            obs,
            seed=13,
            edge_probability=0.4,
        ),
    )
    result_b = run_decentralized_baseline(
        name="random_b",
        observations=observations,
        graph=graph,
        rule=lambda obs: DecentralizedPolicyBaselines.local_random(
            obs,
            seed=13,
            edge_probability=0.4,
        ),
    )

    assert result_a.joint_action.selected_edge_ids == result_b.joint_action.selected_edge_ids


def test_decentralized_rules_validate_parameters() -> None:
    _, observations = _observations()
    obs = observations[0]

    for bad_probability in (-0.1, 1.1):
        try:
            DecentralizedPolicyBaselines.reliability_threshold(obs, bad_probability)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid threshold accepted")

    try:
        DecentralizedPolicyBaselines.top_k_reliability(obs, k=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative k accepted")
