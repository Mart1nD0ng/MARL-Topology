from marl_topology.data.stage22_action_semantics_evidence import (
    build_stage22_action_semantics_evidence,
)
from marl_topology.policies import (
    UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID,
    EdgeScoreRecord,
    PhysicalLinkConflictAwareAssembler,
    aggregate_endpoint_scores_to_physical_links,
    make_directed_edge_id,
)
from marl_topology.policies.topology_assembler import CandidateEdgeConstraint


def _score(tx_id: str, rx_id: str, score: float) -> EdgeScoreRecord:
    edge_id = "--".join(sorted((tx_id, rx_id)))
    return EdgeScoreRecord(
        agent_id=tx_id,
        neighbor_id=rx_id,
        edge_id=edge_id,
        directed_edge_id=make_directed_edge_id(tx_id, rx_id),
        score=score,
        probability=0.9,
        score_source="stage22_unit",
    )


def _constraint(tx_id: str, rx_id: str) -> CandidateEdgeConstraint:
    edge_id = "--".join(sorted((tx_id, rx_id)))
    return CandidateEdgeConstraint(
        edge_id=edge_id,
        tx_id=tx_id,
        rx_id=rx_id,
        edge_type="unit",
        role_allowed=True,
        channel_slot=None,
        conflict_group=f"physical:{edge_id}",
    )


def test_physical_option_aggregates_endpoint_proposals_to_one_link() -> None:
    scores = (_score("veh_0", "veh_1", 0.2), _score("veh_1", "veh_0", 0.9))

    physical_scores = aggregate_endpoint_scores_to_physical_links(scores)
    assembly = PhysicalLinkConflictAwareAssembler().assemble(
        physical_scores,
        (_constraint("veh_0", "veh_1"), _constraint("veh_1", "veh_0")),
    )

    assert len(physical_scores) == 1
    assert physical_scores[0].physical_edge_id == "veh_0--veh_1"
    assert physical_scores[0].score == 0.9
    assert assembly.selected_physical_edges == ("veh_0--veh_1",)
    assert assembly.post_projection_physical_edge_count == 1


def test_stage22_evidence_exposes_only_selected_physical_semantics() -> None:
    build = build_stage22_action_semantics_evidence()

    assert set(build.datasets) == {UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID}
    assert build.report["stage23_preflight_removed_ab_loser_from_executable_code"] is True
    dataset = build.datasets[UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID]
    assert dataset.readiness["uses_final_objective_stack"] is True
    assert dataset.readiness["fallback_used"] is False
    assert dataset.actor_edge_sample_count > 0
    assert any(row.selected_physical_edges for row in dataset.rows)
    for row in dataset.rows:
        assert row.action_semantics_id == UNDIRECTED_PHYSICAL_LINK_ACTION_SEMANTICS_ID
        assert row.selected_directed_edges == ()
        assert "consensus_success_probability" not in repr(row.actor_safe_view)
