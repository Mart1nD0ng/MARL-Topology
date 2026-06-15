from marl_topology.env import (
    EdgeActionDecision,
    MinimalDecPOMDPEnv,
    SchemaViolation,
    ensure_actor_observations_do_not_contain_metrics,
    validate_actor_observation_payload,
)
from marl_topology.evaluation import build_demo_stack
from marl_topology.policies import DecentralizedPolicyBaselines


def _env(horizon: int = 2) -> MinimalDecPOMDPEnv:
    scene, graph, evaluator, _ = build_demo_stack()
    return MinimalDecPOMDPEnv(scene=scene, graph=graph, evaluator=evaluator, horizon=horizon)


def test_reset_returns_local_actor_observations_only() -> None:
    env = _env()

    reset = env.reset()

    assert reset.time_step == 0
    assert len(reset.observations) == len(env.graph.node_ids)
    ensure_actor_observations_do_not_contain_metrics(reset.observations)
    for observation in reset.observations:
        validate_actor_observation_payload(observation.to_payload())
        assert observation.time_step == 0


def test_step_accepts_local_decisions_and_advances_time() -> None:
    env = _env(horizon=2)
    reset = env.reset()
    decisions = tuple(
        decision
        for observation in reset.observations
        for decision in DecentralizedPolicyBaselines.all_local_edges(observation)
    )

    result = env.step(decisions)

    assert result.time_step == 1
    assert result.terminated is False
    assert result.joint_action.selected_edge_ids == env.graph.edge_ids
    assert result.evaluation.metrics["consensus_success"] == 1
    ensure_actor_observations_do_not_contain_metrics(result.observations)
    assert all(observation.time_step == 1 for observation in result.observations)


def test_step_no_edges_changes_consensus_response() -> None:
    env = _env(horizon=1)
    reset = env.reset()
    decisions = tuple(
        decision
        for observation in reset.observations
        for decision in DecentralizedPolicyBaselines.no_edges(observation)
    )

    result = env.step(decisions)

    assert result.terminated is True
    assert result.joint_action.selected_edge_ids == ()
    assert result.evaluation.metrics["consensus_success"] == 0


def test_step_requires_reset_first() -> None:
    env = _env()

    try:
        env.step(())
    except RuntimeError:
        pass
    else:
        raise AssertionError("step before reset accepted")


def test_step_rejects_non_candidate_edge_decision() -> None:
    env = _env()
    env.reset()

    try:
        env.step((EdgeActionDecision("veh_0", "not_a_node", True),))
    except SchemaViolation:
        pass
    else:
        raise AssertionError("non-candidate edge decision accepted")


def test_env_validates_configuration() -> None:
    scene, graph, evaluator, _ = build_demo_stack()

    try:
        MinimalDecPOMDPEnv(scene=scene, graph=graph, evaluator=evaluator, horizon=0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid horizon accepted")
