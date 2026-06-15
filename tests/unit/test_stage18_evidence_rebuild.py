from marl_topology.data.learning_evidence_stage18 import (
    build_stage18_evidence_rebuild_report,
    build_stage18_learning_evidence_dataset,
)


def test_stage18_evidence_rows_have_four_separated_views() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    row = dataset.rows[0]

    assert row.views == (
        "actor_safe_view",
        "actor_target_view",
        "critic_target_view",
        "diagnostics_view",
    )
    assert row.actor_safe_view
    assert row.actor_target_view["actor_soft_utility_targets"]
    assert row.critic_target_view["edge_delta_targets"]
    assert row.diagnostics_view["forbidden_field_scan_result"]["actor_safe_view_passed"]


def test_stage18_critic_targets_do_not_leak_into_actor_safe_view() -> None:
    dataset = build_stage18_learning_evidence_dataset()
    actor_text = repr([row.actor_safe_view for row in dataset.rows])

    forbidden = [
        "delta_consensus_success_probability",
        "delta_latency",
        "delta_energy",
        "delta_feasibility",
        "oracle_topology_membership",
        "consensus_success_probability",
        "global_topology",
    ]
    assert not [term for term in forbidden if term in actor_text]


def test_stage18_before_after_contradiction_stats_are_computable() -> None:
    build = build_stage18_evidence_rebuild_report()
    before = build.report["before"]
    after = build.report["after"]

    assert before["raw_sample_count"] == 850
    assert before["contradiction_cluster_count"] == 142
    assert before["hard_label_contradiction_rate"] == 1.0
    assert after["rebuilt_sample_count"] == 850
    assert after["hard_label_allowed_subset_contradiction_rate"] == 0.0
    assert after["samples_converted_to_soft_utility_target"] == 850
    assert after["ranking_pair_count"] > 0
    assert after["samples_moved_to_critic_only"] == 850
    assert after["remaining_contradiction_cluster_count"] == 0


def test_stage18_report_allows_only_owner_gated_supervised_actor_rerun() -> None:
    report = build_stage18_evidence_rebuild_report().report

    assert report["stage19_supervised_actor_rerun_allowed"] is True
    assert report["stage19_supervised_actor_stack_rerun_allowed"] is True
    assert report["stage11_to_stage15_full_rerun_allowed"] is False
    assert report["stage15_policy_gradient_rerun_allowed"] is False
    assert report["ppo_mappo_allowed"] is False
    assert report["coma_allowed"] is False
    assert report["transformer_allowed"] is False
    assert report["scale_up_training_allowed"] is False
    assert report["artifact_written"] is False

