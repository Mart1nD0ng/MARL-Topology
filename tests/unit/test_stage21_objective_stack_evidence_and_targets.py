from functools import lru_cache

from marl_topology.data.stage21_assembler_aware_targets import (
    actor_target_view_has_no_forbidden_global_fields,
    critic_targets_are_critic_only,
)
from marl_topology.data.stage21_objective_stack_evidence import (
    STAGE21_EVALUATOR_ID,
    STAGE21_PHYSICS_REGIME_ID,
    build_stage21_objective_stack_evidence_report,
)


@lru_cache(maxsize=1)
def _stage21_evidence_report():
    return build_stage21_objective_stack_evidence_report()


def test_stage21_evidence_uses_final_stage3_stage4_objective_stack() -> None:
    build = _stage21_evidence_report()
    report = build.report

    assert report["main_evaluator_id"] == STAGE21_EVALUATOR_ID
    assert report["physics_regime_id"] == STAGE21_PHYSICS_REGIME_ID
    assert report["all_rows_use_final_objective_stack"] is True
    assert report["silent_fallback_to_simple_link_or_min_link"] is False
    assert report["partial_readiness_flag"] is False
    assert report["evidence_readiness"]["all_required_evidence_present"] is True


def test_stage21_rows_expose_required_views_without_actor_leakage() -> None:
    build = _stage21_evidence_report()
    row = build.dataset.rows[0]

    assert row.evidence_id
    assert row.evaluator_id == STAGE21_EVALUATOR_ID
    assert row.actor_safe_view
    assert row.actor_target_view["actor_soft_utility_targets"]
    assert row.critic_target_view["edge_delta_targets"]
    assert actor_target_view_has_no_forbidden_global_fields(row.actor_target_view)
    assert critic_targets_are_critic_only(row.critic_target_view)


def test_stage21_target_distribution_has_high_mid_low_and_rankings() -> None:
    report = _stage21_evidence_report().report
    target = report["target_distribution"]

    assert target["uniformly_high"] is False
    assert target["has_high_mid_low_priority"] is True
    assert target["high_priority_count"] > 0
    assert target["mid_priority_count"] > 0
    assert target["low_priority_count"] > 0
    assert target["ranking_pair_count"] > 0
    assert target["low_priority_abstain_signal_count"] > 0


def test_stage21_prior_stage_audit_marks_old_evaluator_lineage() -> None:
    audit = _stage21_evidence_report().report["source_audit"]

    for stage_name in (
        "stage16_learning_evidence",
        "stage18_evidence_rebuild",
        "stage19_supervised_actor_rerun",
        "stage20_assembler_evaluation",
    ):
        assert audit[stage_name]["uses_simple_link_model"] is True
        assert audit[stage_name]["uses_topology_evaluator_min_link_abstraction"] is True
        assert audit[stage_name]["uses_stage3_urlcc_finite_blocklength_stack"] is False
