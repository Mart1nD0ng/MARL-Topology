from marl_topology.env import (
    CentralizedTrainingView,
    EdgeActionDecision,
    JointTopologyAction,
    LocalMessage,
    SchemaViolation,
    build_actor_observation,
    validate_actor_observation_payload,
)
from marl_topology.evaluation import build_demo_stack


def test_actor_observation_contains_only_local_incident_edges() -> None:
    scene, graph, evaluator, _ = build_demo_stack()
    observation = build_actor_observation(
        scene=scene,
        graph=graph,
        link_records=evaluator.link_records,
        agent_id="veh_0",
        time_step=0,
    )

    payload = observation.to_payload()
    validate_actor_observation_payload(payload)
    assert payload["agent_id"] == "veh_0"
    assert "candidate_edge_ids" not in payload
    assert "global_topology" not in payload
    assert {item.neighbor_id for item in observation.local_neighbor_observations} == {
        "rsu_0",
        "veh_1",
        "veh_2",
    }


def test_actor_schema_rejects_global_and_oracle_fields() -> None:
    for forbidden_field in [
        "global_topology",
        "oracle_label",
        "consensus_success",
        "topology_diagnostics",
        "critic_features",
    ]:
        try:
            validate_actor_observation_payload(
                {
                    "agent_id": "veh_0",
                    "agent_kind": "vehicle",
                    "time_step": 0,
                    "local_position_m": (0.0, 0.0, 1.5),
                    "local_neighbor_observations": (),
                    "local_messages": (),
                    "local_history": {},
                    forbidden_field: "leak",
                }
            )
        except SchemaViolation:
            pass
        else:
            raise AssertionError(f"forbidden field accepted: {forbidden_field}")


def test_local_message_history_rejects_nested_forbidden_fields() -> None:
    try:
        LocalMessage(sender_id="rsu_0", message_type="status", payload={"oracle_label": "good"})
    except SchemaViolation:
        pass
    else:
        raise AssertionError("forbidden oracle label accepted in local message")

    scene, graph, evaluator, _ = build_demo_stack()
    try:
        build_actor_observation(
            scene=scene,
            graph=graph,
            link_records=evaluator.link_records,
            agent_id="veh_0",
            time_step=0,
            local_history={"future_consensus_outcome": 1},
        )
    except SchemaViolation:
        pass
    else:
        raise AssertionError("forbidden future outcome accepted in local history")


def test_centralized_training_view_is_not_actor_payload() -> None:
    _, graph, evaluator, oracle = build_demo_stack()
    oracle_result = oracle.solve()
    training_view = CentralizedTrainingView(
        scenario_id=graph.scenario_id,
        node_ids=graph.node_ids,
        candidate_edge_ids=graph.edge_ids,
        selected_edge_ids=oracle_result.selected_edge_ids,
        metric_names=tuple(evaluator.evaluate(set()).metrics),
        oracle_status=oracle_result.status,
    )

    payload = training_view.to_training_only_payload()
    try:
        validate_actor_observation_payload(payload)
    except SchemaViolation:
        pass
    else:
        raise AssertionError("centralized training view accepted as actor payload")


def test_joint_topology_action_assembles_from_local_decisions() -> None:
    _, graph, _, _ = build_demo_stack()
    decisions = (
        EdgeActionDecision("veh_0", "rsu_0", True),
        EdgeActionDecision("veh_0", "veh_1", False),
        EdgeActionDecision("veh_1", "veh_2", True),
    )

    action = JointTopologyAction.from_local_decisions(graph, decisions)

    assert action.selected_edge_ids == ("rsu_0--veh_0", "veh_1--veh_2")
    assert action.assembly_rule == "any_active_local_proposal"


def test_joint_topology_action_rejects_non_candidate_edge() -> None:
    _, graph, _, _ = build_demo_stack()
    try:
        JointTopologyAction.from_local_decisions(
            graph,
            (EdgeActionDecision("veh_0", "not_a_node", True),),
        )
    except SchemaViolation:
        pass
    else:
        raise AssertionError("non-candidate edge accepted")
