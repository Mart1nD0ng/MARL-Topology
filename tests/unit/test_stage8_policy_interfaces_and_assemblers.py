from marl_topology.data import ACTOR_BATCH_REQUIRED_FIELDS
from marl_topology.policies import (
    ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID,
    ActorPolicyInput,
    ActorPolicyOutput,
    AssemblerConfig,
    AssemblerInputViolation,
    CandidateEdgeConstraint,
    ConflictAwareGreedyAssembler,
    EdgeScoreBatch,
    EdgeScoreRecord,
    FixedTopKPerAgentAssembler,
    PolicyBaselines,
    PolicyInterfaceViolation,
    RejectionReason,
    RoleAwareAdaptiveBudgetAssembler,
    ThresholdAssembler,
    make_directed_edge_id,
    validate_actor_policy_output_payload,
)
from marl_topology.scenario import make_demo_scene
from marl_topology.topology import CandidateGraph
from marl_topology.training import (
    CRITIC_OUTPUT_HEAD_NAMES,
    CentralizedCriticInput,
    CriticInterfaceViolation,
    CriticOutputHeads,
    EdgeDeltaCriticOutput,
    assert_critic_payload_cannot_be_actor_input,
)


def _actor_row() -> dict[str, object]:
    return {
        "agent_id": "veh_0",
        "agent_kind": "vehicle",
        "time_step": 0,
        "local_position_m": (0.0, 0.0, 1.5),
        "local_neighbor_observations": (),
        "local_messages": (),
        "local_history": {},
    }


def _edge_id(tx_id: str, rx_id: str) -> str:
    return "--".join(sorted((tx_id, rx_id)))


def _score(
    tx_id: str,
    rx_id: str,
    score: float,
    *,
    source: str = "unit_test_actor_score",
) -> EdgeScoreRecord:
    return EdgeScoreRecord(
        agent_id=tx_id,
        neighbor_id=rx_id,
        edge_id=_edge_id(tx_id, rx_id),
        directed_edge_id=make_directed_edge_id(tx_id, rx_id),
        score=score,
        probability=None,
        score_source=source,
        time_step=0,
    )


def _constraint(
    tx_id: str,
    rx_id: str,
    *,
    edge_type: str = "vehicle_to_vehicle",
    role_allowed: bool = True,
    channel_slot: str | None = None,
    conflict_group: str | None = None,
    tx_capacity_cost: float = 1.0,
    rx_capacity_cost: float = 1.0,
    valid_candidate: bool = True,
) -> CandidateEdgeConstraint:
    return CandidateEdgeConstraint(
        edge_id=_edge_id(tx_id, rx_id),
        tx_id=tx_id,
        rx_id=rx_id,
        edge_type=edge_type,
        role_allowed=role_allowed,
        channel_slot=channel_slot,
        conflict_group=conflict_group,
        tx_capacity_cost=tx_capacity_cost,
        rx_capacity_cost=rx_capacity_cost,
        valid_candidate=valid_candidate,
    )


def _reason(topology, directed_edge_id: str) -> tuple[RejectionReason, ...]:
    return topology.rejection_reasons[directed_edge_id]


def test_actor_policy_input_matches_actor_safe_schema() -> None:
    policy_input = ActorPolicyInput.from_actor_safe_row(_actor_row())

    assert policy_input.schema_id == "actor_policy_local_input_v1"
    assert policy_input.field_names == ACTOR_BATCH_REQUIRED_FIELDS
    assert set(policy_input.to_actor_safe_row()) == set(ACTOR_BATCH_REQUIRED_FIELDS)


def test_actor_policy_input_rejects_forbidden_fields() -> None:
    row = _actor_row()
    row["global_topology"] = ("veh_0->veh_1",)

    try:
        ActorPolicyInput.from_actor_safe_row(row)
    except PolicyInterfaceViolation as exc:
        assert "forbidden policy input fields" in str(exc)
    else:
        raise AssertionError("forbidden actor input field accepted")


def test_actor_policy_output_contains_edge_scores_not_final_topology() -> None:
    batch = EdgeScoreBatch.from_records([_score("veh_0", "veh_1", 0.75)])
    output = ActorPolicyOutput(agent_id="veh_0", edge_scores=batch)
    payload = output.to_payload()

    assert output.schema_id == ACTOR_EDGE_SCORE_OUTPUT_SCHEMA_ID
    assert payload["contains_final_topology"] is False
    assert payload["contains_oracle_label"] is False
    assert payload["contains_consensus_metric"] is False
    assert "selected_directed_edges" not in payload
    validate_actor_policy_output_payload(payload)


def test_actor_policy_output_rejects_oracle_consensus_and_target_fields() -> None:
    for forbidden in (
        "oracle_label",
        "consensus_success_probability",
        "reward_surrogate",
        "edge_delta_targets",
        "selected_directed_edges",
    ):
        try:
            validate_actor_policy_output_payload({"edge_scores": [], forbidden: object()})
        except PolicyInterfaceViolation as exc:
            assert "forbidden actor output fields" in str(exc)
        else:
            raise AssertionError(f"forbidden actor output field accepted: {forbidden}")


def test_threshold_assembler_selects_score_threshold_as_baseline() -> None:
    scores = (_score("veh_0", "veh_1", 0.9), _score("veh_0", "veh_2", 0.1))
    constraints = (_constraint("veh_0", "veh_1"), _constraint("veh_0", "veh_2"))

    topology = ThresholdAssembler(0.5).assemble(scores, constraints)

    assert topology.selected_directed_edges == ("veh_0->veh_1",)
    assert topology.rejected_edges == ("veh_0->veh_2",)
    assert _reason(topology, "veh_0->veh_2") == (RejectionReason.LOW_SCORE,)
    assert topology.diagnostics["baseline_only"] is True
    assert topology.diagnostics["recommended_for_deployment"] is False


def test_fixed_top_k_per_agent_is_baseline_only() -> None:
    scores = (
        _score("veh_0", "veh_1", 0.5),
        _score("veh_0", "veh_2", 0.8),
        _score("veh_1", "veh_2", 0.7),
    )
    constraints = (
        _constraint("veh_0", "veh_1"),
        _constraint("veh_0", "veh_2"),
        _constraint("veh_1", "veh_2"),
    )

    topology = FixedTopKPerAgentAssembler(k=1).assemble(scores, constraints)

    assert set(topology.selected_directed_edges) == {"veh_0->veh_2", "veh_1->veh_2"}
    assert topology.diagnostics["baseline_only"] is True
    assert "baseline only" in " ".join(topology.diagnostics["notes"])


def test_role_aware_adaptive_budget_respects_role_budgets() -> None:
    scores = (
        _score("rsu_0", "veh_0", 0.9),
        _score("rsu_0", "veh_1", 0.8),
        _score("rsu_0", "veh_2", 0.7),
        _score("veh_0", "rsu_0", 0.6),
        _score("veh_0", "veh_1", 0.5),
    )
    constraints = tuple(_constraint(score.agent_id, score.neighbor_id) for score in scores)
    config = AssemblerConfig(
        assembler_id="role_budget_test",
        mode="role_aware_adaptive_budget",
        role_budget_config={
            "default": 1,
            "rsu": 2,
            "vehicle": 1,
            "dense_neighbor_threshold": 99,
        },
        deterministic=True,
    )

    topology = RoleAwareAdaptiveBudgetAssembler(config).assemble(
        scores,
        constraints,
        metadata={"agent_roles": {"rsu_0": "rsu", "veh_0": "vehicle"}},
    )

    assert set(topology.selected_directed_edges) == {
        "rsu_0->veh_0",
        "rsu_0->veh_1",
        "veh_0->rsu_0",
    }
    assert _reason(topology, "rsu_0->veh_2") == (RejectionReason.PROJECTION_LIMIT,)
    assert _reason(topology, "veh_0->veh_1") == (RejectionReason.PROJECTION_LIMIT,)


def test_conflict_aware_greedy_rejects_invalid_candidate() -> None:
    score = _score("veh_0", "veh_1", 0.9)
    topology = ConflictAwareGreedyAssembler().assemble(
        (score,),
        (_constraint("veh_0", "veh_1", valid_candidate=False),),
    )

    assert topology.selected_directed_edges == ()
    assert _reason(topology, "veh_0->veh_1") == (RejectionReason.INVALID_CANDIDATE,)
    assert topology.diagnostics["recommended_for_deployment"] is True


def test_conflict_aware_greedy_rejects_tx_capacity_violation() -> None:
    scores = (_score("veh_0", "veh_1", 0.9), _score("veh_0", "veh_2", 0.8))
    constraints = (
        _constraint("veh_0", "veh_1", channel_slot="slot_a"),
        _constraint("veh_0", "veh_2", channel_slot="slot_b"),
    )
    config = AssemblerConfig(
        assembler_id="tx_capacity_test",
        mode="conflict_aware_greedy",
        tx_capacity={"veh_0": 1.0},
        deterministic=True,
    )

    topology = ConflictAwareGreedyAssembler(config).assemble(scores, constraints)

    assert topology.selected_directed_edges == ("veh_0->veh_1",)
    assert RejectionReason.TX_BUDGET_EXCEEDED in _reason(topology, "veh_0->veh_2")


def test_conflict_aware_greedy_rejects_rx_capacity_violation() -> None:
    scores = (_score("veh_0", "rsu_0", 0.9), _score("veh_1", "rsu_0", 0.8))
    constraints = (
        _constraint("veh_0", "rsu_0", channel_slot="slot_a"),
        _constraint("veh_1", "rsu_0", channel_slot="slot_b"),
    )
    config = AssemblerConfig(
        assembler_id="rx_capacity_test",
        mode="conflict_aware_greedy",
        rx_capacity={"rsu_0": 1.0},
        deterministic=True,
    )

    topology = ConflictAwareGreedyAssembler(config).assemble(scores, constraints)

    assert topology.selected_directed_edges == ("veh_0->rsu_0",)
    assert RejectionReason.RX_CAPACITY_EXCEEDED in _reason(topology, "veh_1->rsu_0")


def test_conflict_aware_greedy_rejects_channel_and_interference_conflicts() -> None:
    scores = (
        _score("veh_0", "veh_1", 0.9),
        _score("veh_2", "veh_3", 0.8),
        _score("veh_4", "veh_5", 0.7),
    )
    constraints = (
        _constraint("veh_0", "veh_1", channel_slot="slot_a", conflict_group="g1"),
        _constraint("veh_2", "veh_3", channel_slot="slot_a", conflict_group="g2"),
        _constraint("veh_4", "veh_5", channel_slot="slot_b", conflict_group="g1"),
    )

    topology = ConflictAwareGreedyAssembler().assemble(scores, constraints)

    assert topology.selected_directed_edges == ("veh_0->veh_1",)
    assert RejectionReason.CHANNEL_CONFLICT in _reason(topology, "veh_2->veh_3")
    assert RejectionReason.INTERFERENCE_CONFLICT in _reason(topology, "veh_4->veh_5")
    assert topology.diagnostics["rejection_reason_counts"]["channel_conflict"] == 1
    assert topology.diagnostics["rejection_reason_counts"]["interference_conflict"] == 1


def test_deployment_assembler_rejects_objective_oracle_and_reward_metadata() -> None:
    score = _score("veh_0", "veh_1", 0.9)
    constraint = _constraint("veh_0", "veh_1")
    assembler = ConflictAwareGreedyAssembler()

    for forbidden in (
        "consensus_success_probability",
        "oracle_label",
        "reward_surrogate",
        "edge_delta_targets",
        "stage4_pbft_reliability_result",
    ):
        try:
            assembler.assemble((score,), (constraint,), metadata={forbidden: 1.0})
        except AssemblerInputViolation as exc:
            assert "forbidden assembler metadata fields" in str(exc)
        else:
            raise AssertionError(f"forbidden assembler metadata accepted: {forbidden}")


def test_full_graph_baseline_is_not_oracle() -> None:
    graph = CandidateGraph.from_scene(make_demo_scene(), max_distance_m=80.0)

    full = PolicyBaselines.full(graph)

    assert full.edge_ids == graph.edge_ids
    assert full.is_oracle is False


def test_centralized_critic_interface_is_training_only() -> None:
    critic_input = CentralizedCriticInput(
        scenario_id="unit_scene",
        time_step=0,
        node_ids=("veh_0", "veh_1"),
        candidate_directed_edges=("veh_0->veh_1",),
        selected_directed_edges=("veh_0->veh_1",),
        centralized_features={"global_graph": "training_only"},
    )

    assert critic_input.training_only is True
    assert_critic_payload_cannot_be_actor_input(critic_input.to_training_only_payload())

    try:
        CentralizedCriticInput(
            scenario_id="unit_scene",
            time_step=0,
            node_ids=("veh_0",),
            candidate_directed_edges=(),
            selected_directed_edges=(),
            training_only=False,
        )
    except CriticInterfaceViolation as exc:
        assert "training-only" in str(exc)
    else:
        raise AssertionError("critic input was allowed for deployment use")


def test_critic_output_heads_and_edge_delta_heads_exist() -> None:
    heads = CriticOutputHeads(
        value=1.0,
        feasibility=0.5,
        consensus_success_probability=0.75,
        latency=0.1,
        energy=0.2,
        edge_delta_add={"veh_0->veh_1": 0.3},
        edge_delta_remove={"veh_0->veh_1": -0.1},
        edge_delta_keep={"veh_0->veh_1": 0.0},
    )
    edge_delta = EdgeDeltaCriticOutput(
        directed_edge_id="veh_0->veh_1",
        edge_delta_add=0.3,
        edge_delta_remove=-0.1,
        edge_delta_keep=0.0,
    )

    assert heads.head_names == CRITIC_OUTPUT_HEAD_NAMES
    assert set(heads.head_names) == {
        "value",
        "feasibility",
        "consensus_success_probability",
        "latency",
        "energy",
        "edge_delta_add",
        "edge_delta_remove",
        "edge_delta_keep",
    }
    assert edge_delta.training_only is True
