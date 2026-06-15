from marl_topology.data.stage22_action_semantics_evidence import (
    build_stage22_action_semantics_evidence,
)
from marl_topology.data.stage22_objective_teacher import (
    actor_target_view_has_no_forbidden_global_fields,
    critic_targets_are_critic_only,
)


def test_stage22_objective_teacher_targets_have_spread_and_rankings() -> None:
    build = build_stage22_action_semantics_evidence()

    for semantics_id, summary in build.report["target_distribution"].items():
        assert summary["uniformly_high"] is False, semantics_id
        assert summary["ranking_pair_count"] > 0, semantics_id
        assert summary["low_priority_abstain_signal_count"] > 0, semantics_id

    selected_summary = build.report["target_distribution"]["undirected_physical_link_v1"]
    assert selected_summary["has_high_mid_low_priority"] is True


def test_stage22_teacher_records_feasibility_and_keeps_actor_inputs_local() -> None:
    dataset = build_stage22_action_semantics_evidence().datasets["undirected_physical_link_v1"]

    assert any(
        row.actor_target_view["teacher_summary"]["teacher_feasible"]
        for row in dataset.rows
    )
    for row in dataset.rows:
        assert actor_target_view_has_no_forbidden_global_fields(row.actor_target_view)
        assert critic_targets_are_critic_only(row.critic_target_view)
        critic_text = repr(row.critic_target_view)
        assert "delta_consensus_success_probability" in critic_text
        assert "objective_value" in critic_text
        actor_text = repr(row.actor_target_view["actor_soft_utility_targets"])
        assert "objective_value" not in actor_text
        assert "delta_consensus_success_probability" not in actor_text
